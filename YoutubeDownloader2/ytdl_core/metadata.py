"""
Audio metadata embedding (ID3 / MP4 / Vorbis) and optional MusicBrainz enrichment.

Tag mapping:
  Field          MP3 (mutagen.id3)       M4A (mutagen.mp4)
  ─────────────────────────────────────────────────────────
  Title          TIT2                    ©nam
  Artist         TPE1                    ©ART
  Album Artist   TPE2                    aART
  Album          TALB                    ©alb
  Year           TDRC                    ©day
  Genre          TCON                    ©gen
  Track #        TRCK                    trkn
  Source URL     COMM:Source URL:eng     ----:com.apple.iTunes:Source URL
  MusicBrainz ID TXXX:MusicBrainz…      ----:com.apple.iTunes:MusicBrainz Track Id
  Cover art      APIC                    covr
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import mutagen  # type: ignore
import musicbrainzngs
import requests
from mutagen.id3 import (
    APIC,  # type: ignore
    COMM,  # type: ignore
    ID3,
    ID3NoHeaderError,  # type: ignore
    TALB,  # type: ignore
    TCON,  # type: ignore
    TDRC,  # type: ignore
    TIT2,  # type: ignore
    TPE1,  # type: ignore
    TPE2,  # type: ignore
    TRCK,  # type: ignore
    TXXX,  # type: ignore
)
from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm
from mutagen.oggvorbis import OggVorbis
from rapidfuzz import fuzz

from .cache import caches
from .config import Config
from .ratelimit import limiters
from .utils import normalize_title, strip_featuring

# MusicBrainz allows one request per second per client and answers anything
# faster with 503, which ``fetch_musicbrainz`` swallows into a silent None. With
# --workers 4 that silently cost us the reference duration *and* the mb_id the
# iTunes catalogue lookup depends on, so the throttle is load-bearing.
#
# It is now a token bucket rather than a lock held across the sleep: the old
# shape turned every catalogue call in the run into a convoy behind whichever
# thread happened to be inside the critical section.
_mb_useragent_set = False
_mb_lock = threading.Lock()

_session_lock = threading.Lock()
_session: Optional[requests.Session] = None


def _configure_musicbrainz() -> None:
    global _mb_useragent_set
    with _mb_lock:
        if _mb_useragent_set:
            return
        _mb_useragent_set = True
    try:
        musicbrainzngs.set_useragent("YTMusicDownloader", "2.0")
    except Exception:
        pass


def _shared_session() -> requests.Session:
    """One pooled ``requests.Session`` for the whole process.

    A fresh Session per request means a fresh TCP connection and a fresh TLS
    handshake per request, which for a run that fetches cover art and iTunes
    references hundreds of times is pure overhead. ``requests`` Sessions are not
    documented as thread-safe, but urllib3's connection pool underneath one is,
    which is the part that actually matters here.
    """
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
            _session.mount("https://", adapter)
            _session.mount("http://", adapter)
        return _session


def _throttled_search(**kwargs):
    """Run a MusicBrainz query at no more than one request per second."""
    _configure_musicbrainz()
    cfg = Config()
    limiters.bucket(
        "musicbrainz", cfg.MUSICBRAINZ_RATE_PER_SECOND, burst=cfg.MUSICBRAINZ_BURST
    ).acquire()
    return musicbrainzngs.search_recordings(**kwargs)


def _throttled_release(release_id: str, **kwargs):
    _configure_musicbrainz()
    cfg = Config()
    limiters.bucket(
        "musicbrainz", cfg.MUSICBRAINZ_RATE_PER_SECOND, burst=cfg.MUSICBRAINZ_BURST
    ).acquire()
    return musicbrainzngs.get_release_by_id(release_id, **kwargs)


def _joined_artist_credit(value) -> str:
    """Flatten MusicBrainz's artist-credit shape into a plain string.

    The list form is a run of credited artists each carrying its own
    ``joinphrase``. Concatenating the names bare glues the last one onto the
    first ("Banda" + "invitado" -> "Bandainvitado"), which then fails to match
    anything downstream -- artist matching, and the guard that decides whose
    recording a reference duration belongs to.
    """
    if not value:
        return ""
    if isinstance(value, dict):
        nested = value.get("name")
        if isinstance(nested, str):
            return nested
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)

    parts: list[str] = []
    joinphrase = " "
    for part in value:
        if isinstance(part, dict):
            name = part.get("name") or (part.get("artist") or {}).get("name") or ""
            joinphrase = part.get("joinphrase", " ") or " "
        else:
            name = str(part)
            joinphrase = " "
        if not name:
            continue
        if parts:
            parts.append(joinphrase)
        parts.append(name)
    return "".join(parts)


def _catalog_entry(
    *,
    mb_id: Optional[str],
    matched_title: Optional[str],
    matched_artist: Optional[str],
    album: Optional[str],
    year: Optional[str],
    genre: Optional[str],
    track_num: Optional[str],
    release_id: Optional[str],
    duration_seconds: Optional[int],
) -> dict:
    """Build the catalog dict shape the rest of the pipeline consumes."""
    return {
        "album": album,
        "year": year,
        "genre": genre,
        "track_num": track_num,
        "mb_id": mb_id,
        "release_id": release_id,
        "cover_url": f"https://coverartarchive.org/release/{release_id}/front" if release_id else None,
        "duration_seconds": duration_seconds,
        "title": matched_title or None,
        "artist": matched_artist or None,
    }


def fetch_musicbrainz(artist: str, song: str) -> Optional[dict]:
    """
    Query MusicBrainz for recording metadata.

    Returns a dict with keys: album, year, genre, track_num, mb_id,
    release_id, cover_url, duration_seconds, title, artist — or None on any
    failure.

    ``title`` and ``artist`` are the recording MusicBrainz actually matched,
    not what we asked for. Callers compare them against the query to decide
    whether the returned ``duration_seconds`` is a trustworthy reference (see
    ``MusicDownloader._resolve_reference``): MusicBrainz happily returns a
    same-titled recording by a different artist, and scoring every candidate -35
    for "duration mismatch" against someone else's recording is pure noise.

    Results are cached. MusicBrainz is throttled to one request per second
    process-wide, so a catalogue answer that is a month old is worth far more
    than the request it would take to ask again.
    """
    cache = caches.get("musicbrainz")
    if not cache.enabled:
        return _fetch_musicbrainz_uncached(artist, song)
    key = cache.make_key(artist, song)
    hit = cache.get_json(key)
    if hit is not None:
        return hit
    value = _fetch_musicbrainz_uncached(artist, song)
    cache.put_json(key, value, Config().CATALOG_CACHE_TTL)
    return value


def _fetch_musicbrainz_uncached(artist: str, song: str) -> Optional[dict]:
    try:
        res = _throttled_search(
            query=f"{song} {artist}",
            artist=artist,
            recording=song,
            limit=3,
        )
        recordings = res.get("recording-list", [])
        if not recordings:
            return None

        best = recordings[0]
        album = year = release_id = track_num = genre = mb_id = None
        duration_seconds = None

        if "length" in best and best["length"]:
            duration_seconds = int(best["length"]) // 1000

        matched_artist = _joined_artist_credit(best.get("artist-credit") or best.get("artist"))
        matched_title = best.get("title") or ""

        release_list = best.get("release-list", [])
        if release_list:
            rel = release_list[0]
            album = rel.get("title")
            release_id = rel.get("id")
            date_str = rel.get("date", "")
            year = date_str[:4] if date_str else None
            media = rel.get("medium-list", [])
            if media:
                tracks = media[0].get("track-list", [])
                if tracks:
                    track_num = tracks[0].get("number")

        tags = best.get("tag-list", [])
        if tags:
            genre = tags[0].get("name")

        mb_id = best.get("id")

        return _catalog_entry(
            mb_id=mb_id,
            matched_title=matched_title,
            matched_artist=matched_artist,
            album=album,
            year=year,
            genre=genre,
            track_num=track_num,
            release_id=release_id,
            duration_seconds=duration_seconds,
        )

    except (musicbrainzngs.WebServiceError, Exception):
        return None


# ---------------------------------------------------------------------------
# Album-level resolution
# ---------------------------------------------------------------------------


def fetch_release_tracklist(release_id: str) -> Optional[dict]:
    """Fetch one release and its full tracklist.

    One throttled call answers for every track on the album. Resolving fifteen
    songs of an album one search at a time costs fifteen calls against a
    one-per-second budget; this costs two, and the tracklist is *exact* -- no
    search ranking sits between the song list and the label's own track order.

    Returns ``{release_id, album, year, cover_url, tracks: [...]}`` or None.
    """
    cache = caches.get("musicbrainz")
    if cache.enabled:
        hit = cache.get_json(cache.make_key("release", release_id))
        if hit is not None:
            return hit

    try:
        release = _throttled_release(release_id, includes=["recordings", "artist-credits"])
    except Exception:
        return None
    if not isinstance(release, dict):
        return None

    album = release.get("title") or None
    date_str = release.get("date") or ""
    year = date_str[:4] if date_str else None
    album_artist = _joined_artist_credit(release.get("artist-credit"))

    tracks: list[dict] = []
    for medium in release.get("medium-list") or []:
        if not isinstance(medium, dict):
            continue
        for recording in medium.get("track-list") or []:
            if not isinstance(recording, dict) or not recording.get("title"):
                continue
            length = recording.get("length")
            tracks.append(
                {
                    "mb_id": recording.get("id"),
                    "title": recording["title"],
                    "artist": _joined_artist_credit(recording.get("artist-credit")) or album_artist,
                    "duration_seconds": int(length) // 1000 if length else None,
                    "track_num": str(recording.get("position") or "") or None,
                    "genre": _first_tag(recording),
                }
            )
    if not tracks:
        return None

    payload = {
        "release_id": release_id,
        "album": album,
        "year": year,
        "artist": album_artist or None,
        "cover_url": f"https://coverartarchive.org/release/{release_id}/front",
        "tracks": tracks,
    }
    if cache.enabled:
        cache.put_json(cache.make_key("release", release_id), payload, Config().CATALOG_CACHE_TTL)
    return payload


def _first_tag(recording: dict) -> Optional[str]:
    tags = recording.get("tag-list") or []
    if tags and isinstance(tags[0], dict):
        return tags[0].get("name")
    return None


class CatalogContext:
    """Per-run MusicBrainz/iTunes context shared by every song of an artist.

    The song list arrives grouped by artist, and an artist's songs are usually an
    album. Once one of them resolves to a release, the rest can be answered from
    that release's tracklist without touching the network -- which removes the
    per-song MusicBrainz call that the one-per-second budget was serialising, and
    makes the reference duration exact rather than search-ranked.

    Falls back to the per-song search whenever the tracklist has no confident
    match, so a track that is not on the resolved album still behaves exactly as
    it did before.
    """

    def __init__(self, config: Optional[Config] = None, *, musicbrainz: bool = False) -> None:
        self.config = config or Config()
        self.musicbrainz = musicbrainz
        self._by_artist: dict[str, dict] = {}
        self._lock = threading.Lock()

    def resolve(self, artist: str, song: str, fetch=None) -> Optional[dict]:
        """Catalog data for one song, from the album tracklist when possible."""
        if not self.musicbrainz:
            return None

        from_tracklist = self.from_tracklist(artist, song)
        if from_tracklist is not None:
            return from_tracklist

        fetcher = fetch or fetch_musicbrainz
        data = fetcher(artist, song)
        if data and data.get("release_id"):
            self.remember_release(artist, data["release_id"])
        return data

    def remember_release(self, artist: str, release_id: str) -> None:
        """Load *release_id*'s tracklist and file it under *artist*.

        Best-effort: a failure here only means the remaining songs of this artist
        keep using the per-song lookup path, which is what they did anyway.
        """
        if not release_id:
            return
        release = fetch_release_tracklist(release_id)
        if release:
            with self._lock:
                self._by_artist.setdefault(artist.lower(), release)

    def from_tracklist(self, artist: str, song: str) -> Optional[dict]:
        """Catalog data for *song* off the artist's loaded release, or None.

        Public so the pipeline can answer most of an album without issuing any
        catalogue request at all.
        """
        if not self.musicbrainz:
            return None
        with self._lock:
            release = self._by_artist.get(artist.lower())
        if not release:
            return None

        target = normalize_title(strip_featuring(song))
        if not target:
            return None

        best: Optional[dict] = None
        best_ratio = 0
        for track in release.get("tracks") or []:
            ratio = fuzz.token_set_ratio(target, normalize_title(strip_featuring(track["title"])))
            if ratio > best_ratio:
                best_ratio = ratio
                best = track
        if best is None or best_ratio < self.config.MB_REFERENCE_MIN_TITLE_MATCH:
            return None

        # A tracklist entry by a different artist is a different recording; its
        # duration is not a valid reference for ours.
        credited = normalize_title(strip_featuring(artist))
        actual = normalize_title(strip_featuring(best.get("artist") or artist))
        if credited and actual and fuzz.token_set_ratio(credited, actual) < (
            self.config.MB_REFERENCE_MIN_ARTIST_MATCH
        ):
            return None

        return _catalog_entry(
            mb_id=best.get("mb_id"),
            matched_title=best.get("title"),
            matched_artist=best.get("artist"),
            album=release.get("album"),
            year=release.get("year"),
            genre=best.get("genre"),
            track_num=best.get("track_num"),
            release_id=release.get("release_id"),
            duration_seconds=best.get("duration_seconds"),
        )

    def prime(self, artist: str, song: str, fetch=None) -> None:
        """Eagerly load the album for *artist* so later songs never wait on a call.

        Called once per artist at the head of a batch, before its songs are
        dispatched, so the throttled lookup overlaps with other artists' work
        instead of sitting on one song's critical path.
        """
        if not self.musicbrainz:
            return
        with self._lock:
            if artist.lower() in self._by_artist:
                return
        fetcher = fetch or fetch_musicbrainz
        data = fetcher(artist, song)
        if not data or not data.get("release_id"):
            return
        self.remember_release(artist, data["release_id"])


def fetch_itunes_reference(artist: str, song: str, timeout: float = 8.0) -> Optional[dict]:
    """
    Look up a song in the iTunes catalogue for a trustworthy duration reference.

    MusicBrainz coverage of Latin American cumbia/vallenato is patchy, and when
    it *does* answer it frequently returns a same-titled recording by a different
    artist -- whose duration then penalises every correct candidate. Apple's
    catalogue carries most of this material and answers by title+artist, so it
    serves as both the reference duration and an independent check on whether
    the song is really attributed to the artist named in the song list.

    This is the ``search`` endpoint, not ``lookup``: ``lookup`` does not accept
    MusicBrainz IDs (it answers 400), and ``trackViewUrl`` points at the Apple
    Music store rather than at a YouTube video, so it cannot resolve an official
    upload directly.

    Returns ``{title, artist, album, duration, track_id, view_url}`` or None.
    """
    if not artist or not song:
        return None

    cache = caches.get("itunes")
    if cache.enabled:
        hit = cache.get_json(cache.make_key(artist, song))
        if hit is not None:
            return hit

    payload = _fetch_itunes_uncached(artist, song, timeout)
    if cache.enabled:
        cache.put_json(cache.make_key(artist, song), payload, Config().CATALOG_CACHE_TTL)
    return payload


def _fetch_itunes_uncached(artist: str, song: str, timeout: float = 8.0) -> Optional[dict]:
    term = f"{artist} {song}".strip()
    url = "https://itunes.apple.com/search?" + urlencode(
        {"term": term, "entity": "song", "limit": 5}
    )
    try:
        cfg = Config()
        limiters.bucket("itunes", cfg.ITUNES_RATE_PER_SECOND, burst=cfg.ITUNES_BURST).acquire()
        response = _shared_session().get(
            url,
            timeout=timeout,
            headers={"User-Agent": "YTMusicDownloader/2.0"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    target = normalize_title(strip_featuring(song))
    for track in payload.get("results") or []:
        if not isinstance(track, dict) or track.get("kind") != "song":
            continue
        title = track.get("trackName") or ""
        if not title:
            continue
        if fuzz.token_set_ratio(target, normalize_title(strip_featuring(title))) < 85:
            continue
        track_time = track.get("trackTimeMillis")
        return {
            "title": title,
            "artist": track.get("artistName") or "",
            "album": track.get("collectionName") or "",
            "duration": int(track_time) // 1000 if track_time else 0,
            "track_id": track.get("trackId"),
            "view_url": track.get("trackViewUrl") or "",
        }
    return None


def _fetch_image(url: str) -> Optional[bytes]:
    """Download raw image bytes for *url*, returning None on error.

    Cached by URL, because the dominant case is an album: every track resolves to
    the same ``coverartarchive`` release front, and re-downloading byte-identical
    cover art once per track is a request per track for no new information.
    """
    if not url:
        return None

    cache = caches.get("cover")
    if cache.enabled:
        hit = cache.get_bytes(url)
        if hit is not None:
            return hit

    try:
        response = _shared_session().get(
            url,
            timeout=10,
            headers={"User-Agent": "YTMusicDownloader/2.0"},
        )
        response.raise_for_status()
        data = response.content
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

    if cache.enabled and data:
        cache.put_bytes(url, data, Config().COVER_CACHE_TTL)
    return data


def _embed_mp3(
    path: Path,
    title: str,
    artist: str,
    album: str,
    year: str,
    genre: str,
    track_num: str,
    mb_id: str,
    source_url: str,
    image: Optional[bytes],
) -> None:
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()

    tags.add(TIT2(encoding=3, text=title))
    tags.add(TPE1(encoding=3, text=artist))
    tags.add(TPE2(encoding=3, text=artist))
    if album:
        tags.add(TALB(encoding=3, text=album))
    if year:
        tags.add(TDRC(encoding=3, text=str(year)))
    if genre:
        tags.add(TCON(encoding=3, text=genre))
    if track_num:
        tags.add(TRCK(encoding=3, text=str(track_num)))
    if source_url:
        tags.add(COMM(encoding=3, lang="eng", desc="Source URL", text=source_url))
    if mb_id:
        tags.add(TXXX(encoding=3, desc="MusicBrainz Track Id", text=mb_id))
    if image:
        tags.delall("APIC")
        tags.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=image))
    tags.save(str(path), v2_version=3)


def _embed_m4a(
    path: Path,
    title: str,
    artist: str,
    album: str,
    year: str,
    genre: str,
    track_num: str,
    mb_id: str,
    source_url: str,
    image: Optional[bytes],
) -> None:
    tags = MP4(str(path))
    tags["©nam"] = [title]
    tags["©ART"] = [artist]
    tags["aART"] = [artist]
    if album:
        tags["©alb"] = [album]
    if year:
        tags["©day"] = [str(year)]
    if genre:
        tags["©gen"] = [genre]
    if track_num:
        try:
            tags["trkn"] = [(int(str(track_num).split("/")[0]), 0)]
        except (ValueError, TypeError):
            pass
    if source_url:
        tags["----:com.apple.iTunes:Source URL"] = [MP4FreeForm(source_url.encode("utf-8"))]
    if mb_id:
        tags["----:com.apple.iTunes:MusicBrainz Track Id"] = [MP4FreeForm(mb_id.encode("utf-8"))]
    if image:
        tags["covr"] = [MP4Cover(image, imageformat=MP4Cover.FORMAT_JPEG)]
    tags.save()


def _embed_opus(
    path: Path,
    title: str,
    artist: str,
    album: str,
    year: str,
    genre: str,
    track_num: str,
    console_warn_fn=None,
) -> None:
    """Embed Vorbis comments. Cover art is not supported in OGG Vorbis."""
    if console_warn_fn:
        console_warn_fn("[yellow]⚠ Cover art embedding is not supported for OPUS files.[/yellow]")
    try:
        tags = OggVorbis(str(path))
        tags["title"] = [title]
        tags["artist"] = [artist]
        if album:
            tags["album"] = [album]
        if year:
            tags["date"] = [str(year)]
        if genre:
            tags["genre"] = [genre]
        if track_num:
            tags["tracknumber"] = [str(track_num)]
        tags.save()
    except Exception:
        pass


def embed_metadata(
    file_path: Path,
    title: str,
    artist: str,
    extra: dict,
    thumbnail_url: Optional[str],
    fmt: str,
    console_warn_fn=None,
) -> bool:
    """
    Embed tags into *file_path*.

    Args:
        extra: dict with optional keys: album, year, genre, track_num,
               mb_id, source_url, cover_url.
        thumbnail_url: fallback image URL if extra["cover_url"] is absent.
        fmt:   "mp3" | "m4a" | "mp4" | "opus".
        console_warn_fn: callable(str) for Rich warnings (optional).

    Returns:
        True on success and integrity check pass; False otherwise.
    """
    album = extra.get("album") or ""
    year = extra.get("year") or ""
    genre = extra.get("genre") or ""
    track_num = extra.get("track_num") or ""
    mb_id = extra.get("mb_id") or ""
    source_url = extra.get("source_url") or ""
    cover_url = extra.get("cover_url") or thumbnail_url

    image: Optional[bytes] = _fetch_image(cover_url) if cover_url else None

    try:
        if fmt == "mp3":
            _embed_mp3(
                file_path,
                title,
                artist,
                album,
                year,
                genre,
                track_num,
                mb_id,
                source_url,
                image,
            )
        elif fmt == "m4a":
            _embed_m4a(
                file_path,
                title,
                artist,
                album,
                year,
                genre,
                track_num,
                mb_id,
                source_url,
                image,
            )
        elif fmt == "mp4":
            _embed_m4a(
                file_path,
                title,
                artist,
                album,
                year,
                genre,
                track_num,
                mb_id,
                source_url,
                image,
            )
        elif fmt == "opus":
            _embed_opus(
                file_path,
                title,
                artist,
                album,
                year,
                genre,
                track_num,
                console_warn_fn,
            )

        # Post-embed integrity check
        probe = mutagen.File(str(file_path))  # type: ignore
        return probe is not None

    except (mutagen.MutagenError, Exception):  # type: ignore
        return False
