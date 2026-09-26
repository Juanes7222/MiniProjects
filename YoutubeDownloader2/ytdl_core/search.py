"""YouTube, YouTube Music API, iTunes catalog, SoundCloud, and Bandcamp search
orchestration, candidate scoring heuristic, and selection logic."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote_plus

import yt_dlp
from rapidfuzz import fuzz

from .config import Config
from .scorer import rank_results, score_youtube_result  # noqa: F401 — re-exported
from .utils import normalize_title, strip_featuring

if TYPE_CHECKING:
    from rich.console import Console

    from .channels import ChannelTrust


def build_search_query(artist: str, song: str, source: str) -> str:
    """
    Builds a structured search query tailored for specific streaming platforms.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        source: The target platform identifier (e.g., 'youtube').

    Returns:
        A formatted query string.
    """
    if source == "youtube":
        return f"{artist} {song} official audio"
    return f"{song} {artist}"


def search_ytmusic_official(artist: str, song: str, opts: dict) -> list[dict]:
    """
    Queries the official YouTube Music catalog for verified audio tracks.

    Bypasses traditional user-generated videos (covers, remixes, speed-ups)
    by fetching directly from the internal YouTube Music songs library.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        opts: Configuration options dictionary containing performance constraints.

    Returns:
        A list of standardized metadata dictionaries representing official tracks.
    """
    try:
        from ytmusicapi import YTMusic

        ytmusic = YTMusic()
        max_r = min(opts.get("max_results", 5), 5)
        query = f"{artist} {song}"
        search_results = ytmusic.search(query, filter="songs", limit=max_r)
        # Fall back to video catalog when the songs index returns fewer hits than expected.
        if len(search_results) < max_r:
            try:
                video_results = ytmusic.search(query, filter="videos", limit=max_r)
                search_results = search_results + video_results
            except Exception:
                pass
    except Exception:
        return []

    structured_results = []
    for track in search_results:
        video_id = track.get("videoId")
        if not video_id:
            continue

        artists = [a.get("name", "") for a in track.get("artists", []) if a.get("name")]
        channel_name = artists[0] if artists else "YouTube Music"

        structured_results.append(
            {
                "id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": track.get("title", ""),
                "channel": channel_name,
                "uploader": channel_name,
                "artists": artists,
                "duration": track.get("duration_seconds") or 0,
                "thumbnail": track.get("thumbnails", [{}])[0].get("url")
                if track.get("thumbnails")
                else None,
                "view_count": 0,
                "_source": "ytmusic_api",
            }
        )
    return structured_results


def search_source(query: str, source: str, opts: dict) -> list[dict]:
    """
    Extracts flat metadata entries from a specific scraper source using yt_dlp.

    Args:
        query: The raw text search query.
        source: The target source platform identifier ('youtube', 'soundcloud', 'bandcamp').
        opts: Network, proxy, and authentication options.

    Returns:
        A list of unverified candidate metadata entries.
    """
    max_r = opts.get("max_results", 5)
    prefix = {
        "youtube": f"ytsearch{max_r}",
        "soundcloud": f"scsearch{max_r}",
        "bandcamp": f"bcsearch{max_r}",
    }.get(source, f"ytsearch{max_r}")

    ydl_opts: Any = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": True,
    }
    if opts.get("cookies_browser"):
        ydl_opts["cookiesfrombrowser"] = (opts["cookies_browser"],)
    if opts.get("cookies_file"):
        ydl_opts["cookiefile"] = str(opts["cookies_file"])
    if opts.get("proxy"):
        ydl_opts["proxy"] = opts["proxy"]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info: Any = ydl.extract_info(f"{prefix}:{query}", download=False)
            if info and "entries" in info:
                return [e for e in info.get("entries", []) if e is not None]
    except Exception:
        pass
    return []


def build_query_variants(
    artist: str,
    song: str,
    source: str,
    canonical_song: Optional[str] = None,
    extra_queries: Optional[list[str]] = None,
) -> list[str]:
    """
    Generates string variations of query inputs to increase search coverage.

    ``canonical_song`` is the recording title as spelled by MusicBrainz. When it
    differs from the requested title (diacritics, word order, ``"Cariñito Sin Mí"``
    vs ``"Cariñito Sin Mi"``) it is worth its own query: the user's list is
    hand-written from memory and organic YouTube search frequently ranks the
    correct upload below unrelated same-artist tracks.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        source: The target platform identifier.
        canonical_song: Catalogue spelling of the title, when known.
        extra_queries: Additional caller-supplied query strings.

    Returns:
        A list of query string permutations, de-duplicated, order preserved.
    """
    if source != "youtube":
        queries = [f"{song} {artist}"]
    else:
        queries = [
            f"{artist} {song} official audio",
            f"{artist} - {song}",
            f"{artist} {song}",
        ]

    if canonical_song and canonical_song.strip() and canonical_song.strip() != song.strip():
        queries.append(f"{artist} {canonical_song}")
        queries.append(f"{artist} - {canonical_song}")

    for extra in extra_queries or []:
        cleaned = (extra or "").strip()
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    seen: set[str] = set()
    ordered: list[str] = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            ordered.append(query)
    return ordered


def search_with_variants(
    artist: str,
    song: str,
    source: str,
    opts: dict,
    queries: Optional[list[str]] = None,
) -> list[dict]:
    """
    Iterates over multiple query permutations to gather candidate tracks.

    Every query is fetched to completion before returning: the caller's
    ``max_results`` is a *presentation* budget, not a fetch budget. Truncating
    mid-fan-out lets the first variant ("official audio") consume the entire
    allowance, which is precisely the failure mode that hid the real recording
    of obscure catalogue tracks behind four organic results.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        source: The target platform identifier.
        opts: Configuration options dictionary.
        queries: Pre-built query list; defaults to ``build_query_variants``.

    Returns:
        A list of unique track results, capped at the fetch budget.
    """
    max_results = max(1, int(opts.get("max_results", 5)))
    fetch_multiplier = max(1, int(opts.get("fetch_multiplier", 4)))
    min_per_query = max(1, int(opts.get("min_fetch_per_query", 10)))
    hard_cap = max(max_results, max_results * fetch_multiplier)

    query_list = queries or build_query_variants(artist, song, source)
    per_query_limit = max(min_per_query, -(-max_results // max(1, len(query_list))))
    variant_options = {**opts, "max_results": per_query_limit}

    seen_ids: set[str] = set()
    all_results: list[dict] = []

    for query in query_list:
        for result in search_source(query, source, variant_options):
            video_id = result.get("id") or result.get("url")
            if video_id and video_id in seen_ids:
                continue
            if video_id:
                seen_ids.add(video_id)
            all_results.append(result)
        if len(all_results) >= hard_cap:
            break

    return all_results[:hard_cap]


def search_channel_tabs(
    song: str,
    channels: list[dict],
    opts: dict,
) -> list[dict]:
    """
    Search *inside* channels that previously delivered this artist.

    Channel upload listings are curated and stable, unlike organic search
    ranking, so for catalogue back-catalogue this reliably surfaces the label's
    own upload when keyword search returns nothing but the artist's other songs.

    Args:
        song: The title of the song.
        channels: ``{name, url}`` records from :class:`ChannelTrust`.
        opts: Network, proxy, and authentication options.

    Returns:
        A list of candidate metadata dictionaries (``_source="channel"``).
    """
    if not channels:
        return []

    max_r = max(1, int(opts.get("max_results", 5)))
    per_channel = max(3, -(-max_r // max(1, len(channels))))

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": True,
        "playlistend": per_channel,
    }
    if opts.get("cookies_browser"):
        ydl_opts["cookiesfrombrowser"] = (opts["cookies_browser"],)
    if opts.get("cookies_file"):
        ydl_opts["cookiefile"] = str(opts["cookies_file"])
    if opts.get("proxy"):
        ydl_opts["proxy"] = opts["proxy"]

    results: list[dict] = []
    seen: set[str] = set()
    for channel in channels:
        base = (channel.get("url") or "").rstrip("/")
        if not base:
            continue
        target = f"{base}/search?query={quote_plus(song)}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info: Any = ydl.extract_info(target, download=False)
                entries = (info or {}).get("entries") or []
                for entry in entries:
                    if not entry:
                        continue
                    video_id = entry.get("id") or entry.get("url")
                    if video_id and video_id in seen:
                        continue
                    if video_id:
                        seen.add(video_id)
                    entry = dict(entry)
                    entry["_source"] = "channel"
                    results.append(entry)
        except Exception:
            continue
    return results


def search_ytmusic_album(artist: str, song: str, album: str, opts: dict) -> list[dict]:
    """
    Find the song by browsing its album in the official YouTube Music catalogue.

    Album tracklists are exact: no keyword ranking sits between us and the
    label's own upload. This is the only source that can answer "which video is
    the record label's master" for catalogue material, and it is what recovers
    the tracks that organic search ranks below the artist's *other* songs.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        album: Album name from the catalogue reference.
        opts: Unused; kept for signature symmetry with the other sources.

    Returns:
        A list of candidate dictionaries (``_source="itunes"``).
    """
    if not album:
        return []
    try:
        from ytmusicapi import YTMusic

        ytmusic = YTMusic()
        albums = ytmusic.search(f"{album} {artist}", filter="albums", limit=3)
    except Exception:
        return []

    target = normalize_title(strip_featuring(song))
    best_entry = None
    best_ratio = 0
    for album_result in albums or []:
        browse_id = album_result.get("browseId")
        if not browse_id:
            continue
        try:
            details = ytmusic.get_album(browse_id)
        except Exception:
            continue
        for track in (details or {}).get("tracks") or []:
            title = track.get("title") or ""
            if not title:
                continue
            ratio = fuzz.token_set_ratio(target, normalize_title(strip_featuring(title)))
            if ratio > best_ratio and track.get("videoId"):
                best_ratio = ratio
                best_entry = track

    if best_entry is None or best_ratio < 85:
        return []

    video_id = best_entry["videoId"]
    artists = [a.get("name", "") for a in best_entry.get("artists", []) if a.get("name")]
    channel = artists[0] if artists else "YouTube Music"
    return [
        {
            "id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": best_entry.get("title") or "",
            "channel": channel,
            "uploader": channel,
            "artists": artists,
            "duration": best_entry.get("duration_seconds") or 0,
            "view_count": 0,
            "_source": "itunes",
        }
    ]


def _dedup_results(results: list[dict]) -> list[dict]:
    """
    Deduplicates candidates by ID, merging repeated entries into a source count.

    When the same video ID appears from multiple sources, the first occurrence
    is kept and its ``_source_count`` is incremented. If a later occurrence
    comes from ``ytmusic_api``, the stored entry is promoted to that source so
    the scorer applies the API fast-path regardless of which source resolved first.
    The ``artists`` list from the API entry is also merged in, since yt_dlp
    scraping does not populate that field.
    """
    seen_ids: dict[str, int] = {}  # id -> index in deduped
    deduped: list[dict] = []
    sig_set: set = set()

    for r in results:
        vid = r.get("id") or r.get("url")
        if vid:
            if vid in seen_ids:
                existing = deduped[seen_ids[vid]]
                existing["_source_count"] = existing.get("_source_count", 1) + 1
                # Promote to ytmusic_api source and merge artist metadata if available.
                if r.get("_source") == "ytmusic_api":
                    existing["_source"] = "ytmusic_api"
                    if r.get("artists"):
                        existing["artists"] = r["artists"]
                continue
            seen_ids[vid] = len(deduped)

        sig = (r.get("title"), r.get("channel"), r.get("duration"))
        if sig in sig_set:
            continue
        sig_set.add(sig)
        entry = dict(r)
        entry.setdefault("_source_count", 1)
        deduped.append(entry)

    return deduped


def search_all_sources(
    artist: str,
    song: str,
    sources: list[str],
    opts: dict,
    mb_data: Optional[dict] = None,
    channel_trust: "Optional[ChannelTrust]" = None,
    config: Optional[Config] = None,
) -> list[dict]:
    """
    Executes concurrent cross-platform lookups across all requested streams.

    Injects, in addition to the requested sources:

    - the official YouTube Music API catalog (when YouTube is requested);
    - the iTunes catalogue lookup, resolved from the MusicBrainz recording ID,
      which returns the label's own upload without any keyword search;
    - a channel-scoped search of channels that previously delivered this artist.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        sources: List of target scraper backends requested by the pipeline.
        opts: Shared network, proxy, and operational configurations.
        mb_data: MusicBrainz metadata for this song, when available.
        channel_trust: Learned channel trust model, when available.
        config: Active configuration, used for fetch-budget defaults.

    Returns:
        A unified, deduplicated list of candidate dictionaries.
    """
    cfg = config or Config()
    all_results = []
    active_sources = list(sources)
    if "youtube" in active_sources and "ytmusic_api" not in active_sources:
        active_sources.append("ytmusic_api")

    canonical_song = None
    catalog_album = None
    if isinstance(mb_data, dict):
        canonical_song = mb_data.get("title") or None
        catalog_album = mb_data.get("album") or None

    youtube_queries = build_query_variants(artist, song, "youtube", canonical_song)

    def _run_variant_scrape(src: str) -> list[dict]:
        queries = youtube_queries if src == "youtube" else None
        return search_with_variants(artist, song, src, opts, queries=queries)

    with ThreadPoolExecutor(max_workers=max(1, len(active_sources))) as executor:
        futures = {}
        for src in active_sources:
            if src == "ytmusic_api":
                futures[executor.submit(search_ytmusic_official, artist, song, opts)] = src
            else:
                futures[executor.submit(_run_variant_scrape, src)] = src

        if "youtube" in active_sources and catalog_album:
            futures[
                executor.submit(search_ytmusic_album, artist, song, catalog_album, opts)
            ] = "itunes"

        if channel_trust is not None and "youtube" in active_sources:
            known = channel_trust.channels_for_artist(artist, cfg.MAX_CHANNEL_SEARCHES)
            if known:
                futures[executor.submit(search_channel_tabs, song, known, opts)] = "channel"

        for future in as_completed(futures):
            source = futures[future]
            try:
                source_results = future.result()
            except Exception:
                continue
            for result in source_results:
                result.setdefault("_source", source)
            all_results.extend(source_results)

    return _dedup_results(all_results)


def print_candidates_table(
    scored: list[tuple[dict, int, dict]],
    artist: str,
    song: str,
    console: "Console",
    reject_threshold: int,
) -> None:
    from .cli.candidate_table import print_candidate_table

    print_candidate_table(scored, artist, song, console, reject_threshold)


def select_best_result(
    results: list[dict],
    artist: str,
    song: str,
    mb_duration_seconds: int | None,
    config: Config,
    console: Optional[Console],
    console_lock: Optional[threading.Lock],
    min_duration: int | None = None,
    max_duration: int | None = None,
    score_threshold: int | None = None,
    channel_trust: "Optional[ChannelTrust]" = None,
) -> tuple[dict | None, list[tuple[dict, int, dict]]]:
    """
    Evaluates candidates and isolates the highest-scoring matching track.

    Args:
        results: Unified inputs pool collected from all platforms.
        artist: Verified target artist string.
        song: Verified target song title string.
        mb_duration_seconds: Reference track duration, or None when the
            MusicBrainz reference is not trustworthy for this song.
        config: Shared parameters settings module instance.
        console: Rich text engine connection object.
        console_lock: Thread block lock object protecting stdout streams.
        min_duration: Minimum duration constraints parameter override.
        max_duration: Maximum duration constraints parameter override.
        score_threshold: Target matching floor scoring filter cutoff value.
        channel_trust: Learned channel trust model applied during scoring.

    Returns:
        A tuple with the best tracking map candidate (or None if disqualified) and the total scored list.
    """
    reject_threshold = (
        score_threshold if score_threshold is not None else config.SCORE_THRESHOLD_REJECT
    )

    scored = rank_results(
        results,
        artist,
        song,
        mb_duration_seconds,
        config,
        min_duration,
        max_duration,
        channel_trust,
    )

    if console is not None and scored:
        if console_lock:
            with console_lock:
                print_candidates_table(scored, artist, song, console, reject_threshold)
        else:
            print_candidates_table(scored, artist, song, console, reject_threshold)

    if not scored or scored[0][1] < reject_threshold:
        return None, scored

    return scored[0][0], scored
