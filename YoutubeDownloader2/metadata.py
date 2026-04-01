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

from pathlib import Path
from typing import Optional

import mutagen # type: ignore
import musicbrainzngs
import requests
from mutagen.id3 import (
    APIC,  # type: ignore
    COMM,  # type: ignore
    ID3,
    ID3NoHeaderError,  # type: ignore
    TALB,  # type: ignore
    TDRC,  # type: ignore
    TCON,  # type: ignore
    TIT2,  # type: ignore
    TPE1,  # type: ignore
    TPE2,  # type: ignore
    TRCK,  # type: ignore
    TXXX,  # type: ignore
)
from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm
from mutagen.oggvorbis import OggVorbis


def fetch_musicbrainz(artist: str, song: str) -> Optional[dict]:
    """
    Query MusicBrainz for recording metadata.

    Returns a dict with keys: album, year, genre, track_num, mb_id,
    release_id, cover_url — or None on any failure.
    """
    try:
        res = musicbrainzngs.search_recordings(
            query=f"{song} {artist}",
            artist=artist,
            recording=song,
            limit=3,
        )
        recordings = res.get("recording-list", [])
        if not recordings:
            return None

        best = recordings[0]
        album = year = release_id = track_num = genre = mb_id = cover_url = None

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

        if release_id:
            caa = f"https://coverartarchive.org/release/{release_id}/front"
            try:
                r = requests.head(caa, timeout=5, allow_redirects=True)
                if r.status_code == 200:
                    cover_url = caa
            except requests.exceptions.RequestException:
                pass

        return {
            "album": album,
            "year": year,
            "genre": genre,
            "track_num": track_num,
            "mb_id": mb_id,
            "release_id": release_id,
            "cover_url": cover_url,
        }

    except (musicbrainzngs.WebServiceError, Exception):
        return None


def _fetch_image(url: str) -> Optional[bytes]:
    """Download raw image bytes from *url*, returning None on error."""
    try:
        session = requests.Session()
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException:
        return None


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
        tags.add(
            APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=image)
        )
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
        tags["----:com.apple.iTunes:Source URL"] = [
            MP4FreeForm(source_url.encode("utf-8"))
        ]
    if mb_id:
        tags["----:com.apple.iTunes:MusicBrainz Track Id"] = [
            MP4FreeForm(mb_id.encode("utf-8"))
        ]
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
        console_warn_fn(
            "[yellow]⚠ Cover art embedding is not supported for OPUS files.[/yellow]"
        )
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
        fmt:   "mp3" | "m4a" | "opus".
        console_warn_fn: callable(str) for Rich warnings (optional).

    Returns:
        True on success and integrity check pass; False otherwise.
    """
    album     = extra.get("album") or ""
    year      = extra.get("year") or ""
    genre     = extra.get("genre") or ""
    track_num = extra.get("track_num") or ""
    mb_id     = extra.get("mb_id") or ""
    source_url = extra.get("source_url") or ""
    cover_url  = extra.get("cover_url") or thumbnail_url

    image: Optional[bytes] = _fetch_image(cover_url) if cover_url else None

    try:
        if fmt == "mp3":
            _embed_mp3(
                file_path, title, artist, album, year,
                genre, track_num, mb_id, source_url, image,
            )
        elif fmt == "m4a":
            _embed_m4a(
                file_path, title, artist, album, year,
                genre, track_num, mb_id, source_url, image,
            )
        elif fmt == "opus":
            _embed_opus(
                file_path, title, artist, album, year, genre,
                track_num, console_warn_fn,
            )

        # Post-embed integrity check
        probe = mutagen.File(str(file_path)) # type: ignore
        return probe is not None

    except (mutagen.MutagenError, Exception): # type: ignore
        return False