"""YouTube, YouTube Music API, SoundCloud, and Bandcamp search orchestration,
candidate scoring heuristic, and selection logic."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import yt_dlp
from rich import box
from rich.console import Console
from rich.table import Table

from .config import Config
from .scorer import rank_results, score_youtube_result  # noqa: F401 — re-exported
from .utils import format_duration


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

        structured_results.append({
            "id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": track.get("title", ""),
            "channel": channel_name,
            "uploader": channel_name,
            "artists": artists,                  # nuevo
            "duration": track.get("duration_seconds") or 0,
            "thumbnail": track.get("thumbnails", [{}])[0].get("url") if track.get("thumbnails") else None,
            "view_count": 0,
            "_source": "ytmusic_api",
        })
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
        "extract_flat": False,
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


def build_query_variants(artist: str, song: str, source: str) -> list[str]:
    """
    Generates string variations of query inputs to increase search coverage.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        source: The target platform identifier.

    Returns:
        A list of query string permutations.
    """
    if source != "youtube":
        return [f"{song} {artist}"]
    return [
        f"{artist} {song} official audio",
        f"{artist} - {song}",
        f"{artist} {song}",
    ]


def search_with_variants(
    artist: str, song: str, source: str, opts: dict
) -> list[dict]:
    """
    Iterates over multiple query permutations to gather candidate tracks.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        source: The target platform identifier.
        opts: Configuration options dictionary.

    Returns:
        An aggregated list of unique track results from the given source.
    """
    seen_ids: set[str] = set()
    all_results: list[dict] = []

    for query in build_query_variants(artist, song, source):
        for result in search_source(query, source, opts):
            video_id = result.get("id") or result.get("url")
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                all_results.append(result)

    return all_results


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


def search_all_sources(artist: str, song: str, sources: list[str], opts: dict) -> list[dict]:
    """
    Executes concurrent cross-platform lookups across all requested streams.

    Automatically injects the official YouTube Music API catalog query if 
    the standard YouTube search source is listed in the parameters.

    Args:
        artist: The name of the artist.
        song: The title of the song.
        sources: List of target scraper backends requested by the pipeline.
        opts: Shared network, proxy, and operational configurations.

    Returns:
        A unified, deduplicated list of candidate dictionaries.
    """
    all_results = []
    active_sources = list(sources)
    if "youtube" in active_sources and "ytmusic_api" not in active_sources:
        active_sources.append("ytmusic_api")

    with ThreadPoolExecutor(max_workers=max(1, len(active_sources))) as executor:
        futures = {}
        for src in active_sources:
            if src == "ytmusic_api":
                futures[executor.submit(search_ytmusic_official, artist, song, opts)] = src
            else:
                futures[executor.submit(search_with_variants, artist, song, src, opts)] = src
            
        for future in as_completed(futures):
            src = futures[future]
            res = future.result()
            for r in res:
                if '_source' not in r:
                    r['_source'] = src
            all_results.extend(res)
            
    return _dedup_results(all_results)





def print_candidates_table(
    scored: list[tuple[dict, int, dict]],
    artist: str,
    song: str,
    console: Console,
    reject_threshold: int,
) -> None:
    """
    Renders a formatted table summarizing all evaluated candidates to the output terminal.

    Args:
        scored: Sorted tracks dataset containing scores and metrics evaluation maps.
        artist: Reference artist name.
        song: Reference song title.
        console: Targeted rich display rendering context.
        reject_threshold: Minimum score required to avoid rejection highlighting.
    """
    tbl = Table(title=f"Candidates for: {artist} -- {song}", box=box.SIMPLE)
    tbl.add_column("#", width=3, style="dim")
    tbl.add_column("Title", max_width=55)
    tbl.add_column("Channel", max_width=30)
    tbl.add_column("Duration", width=10, style="yellow")
    tbl.add_column("Score", width=7)
    tbl.add_column("Top signals", min_width=30, style="dim")

    best_idx = 0 if scored and scored[0][1] >= reject_threshold else None

    for i, (entry, sc, bd) in enumerate(scored):
        dur = int(entry.get("duration") or 0)
        top = sorted(bd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        signals = ", ".join(f"{'+' if v >= 0 else ''}{v} {k}" for k, v in top)
        score_markup = (
            f"[green]{sc}[/green]"
            if sc >= 70
            else f"[yellow]{sc}[/yellow]"
            if sc >= 30
            else f"[red]{sc}[/red]"
        )
        tbl.add_row(
            f"{'>' if i == best_idx else ' '}{i + 1}",
            (entry.get("title") or "")[:55],
            (entry.get("channel") or entry.get("uploader") or "")[:30],
            format_duration(dur),
            score_markup,
            signals,
        )

    console.print(tbl)


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
) -> tuple[dict | None, list[tuple[dict, int, dict]]]:
    """
    Evaluates candidates and isolates the highest-scoring matching track.

    Args:
        results: Unified inputs pool collected from all platforms.
        artist: Verified target artist string.
        song: Verified target song title string.
        mb_duration_seconds: Reference track duration.
        config: Shared parameters settings module instance.
        console: Rich text engine connection object.
        console_lock: Thread block lock object protecting stdout streams.
        min_duration: Minimum duration constraints parameter override.
        max_duration: Maximum duration constraints parameter override.
        score_threshold: Target matching floor scoring filter cutoff value.

    Returns:
        A tuple with the best tracking map candidate (or None if disqualified) and the total scored list.
    """
    reject_threshold = (
        score_threshold if score_threshold is not None else config.SCORE_THRESHOLD_REJECT
    )

    scored = rank_results(
        results, artist, song, mb_duration_seconds, config, min_duration, max_duration
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