"""
YouTube, YouTube Music API, SoundCloud, and Bandcamp search orchestration,
candidate scoring heuristic, and selection logic.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import yt_dlp
from rapidfuzz import fuzz
from rich import box
from rich.console import Console
from rich.table import Table

from .config import Config
from .utils import format_duration, normalize_title, strip_featuring, remove_matching_noise, contains_forbidden_phrase


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
        return f'"{song}" "{artist}" official audio'
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
        max_r = min(opts.get("max_results", 5), 3)
        query = f"{artist} {song}"
        search_results = ytmusic.search(query, filter="songs", limit=max_r)
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
        f'"{song}" "{artist}" official audio',
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
    Deduplicates tracking lists by using content IDs and metadata signatures.

    Args:
        results: A dirty list containing overlapping source entries.

    Returns:
        A deduplicated clean list of track entries.
    """
    seen_ids = set()
    deduped = []
    for r in results:
        vid = r.get("id") or r.get("url")
        if vid:
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
        sig = (r.get("title"), r.get("channel"), r.get("duration"))
        if sig in seen_ids:
            continue
        seen_ids.add(sig)
        deduped.append(r)
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


def score_youtube_result(
    result: dict,
    artist: str,
    song: str,
    mb_duration_seconds: int | None,
    config: Config,
) -> tuple[int, dict[str, int]]:
    """
    Analyzes and ranks a track candidate based on matching metrics and validation filters.

    Employs strict hard-rejection thresholds for forbidden patterns (e.g., covers, live sets, 
    remixes), checks precise duration alignments in absolute seconds, and evaluates channel authority.

    Args:
        result: The metadata dictionary of the track candidate.
        artist: The verified target artist name.
        song: The verified target song title.
        mb_duration_seconds: Expected duration from MusicBrainz metadata registry.
        config: Central parameter definitions and weight allocations.

    Returns:
        A tuple containing the composite integer score and the metric breakdown dictionary.
    """
    entry = dict(result)
    raw_title = entry.get("title") or ""
    title: str = normalize_title(raw_title)
    channel: str = (entry.get("channel") or entry.get("uploader") or "").lower()
    view_count: int = int(entry.get("view_count") or 0)
    result_duration: int = int(entry.get("duration") or 0)
    breakdown: dict[str, int] = {}
        
    artist_clean = normalize_title(strip_featuring(artist.lower()))
    song_clean = normalize_title(strip_featuring(song.lower()))

    if entry.get("_source") == "ytmusic_api":
        song_match = int(fuzz.token_set_ratio(song_clean, title))

        ytmusic_artists = " ".join(entry.get("artists") or [])
        artist_match = max(
            int(fuzz.token_set_ratio(artist_clean, normalize_title(ytmusic_artists))),
            int(fuzz.token_set_ratio(artist_clean, normalize_title(channel))),
        )

        if song_match >= 75 and artist_match >= 60:
            breakdown["official_ytmusic_api"] = 120
            breakdown["catalog_match"] = song_match
            breakdown["artist_match"] = artist_match
            if mb_duration_seconds is not None and result_duration > 0:
                if abs(result_duration - mb_duration_seconds) <= 4:
                    breakdown["duration_perfect"] = config.DURATION_MATCH_BONUS
            return sum(breakdown.values()), breakdown
    
    title_tokens = set(title.split())
    query_song_tokens = set(song_clean.split())
    
    bad_found = contains_forbidden_phrase(raw_title, config.FORBIDEN_TERMS)

    query_text = f"{artist} {song}"
    bad_in_query = contains_forbidden_phrase(query_text, config.FORBIDEN_TERMS)

    if bad_found and not bad_in_query:
        return -9999, {
            f"hard_reject_{bad_found}": -9999
        }

    if "album" in title_tokens and "album" not in query_song_tokens:
         return -9999, {"hard_reject_full_album": -9999}
     
    title_no_noise = remove_matching_noise(title)

    song_match = int(fuzz.token_set_ratio(song_clean, title_no_noise))
    artist_in_title = int(fuzz.token_set_ratio(artist_clean, title_no_noise))
    artist_in_channel = int(fuzz.partial_ratio(artist_clean, channel))
    
    if song_match < 55:
        return -9999, {"hard_reject_song_absent": -9999}
        
    artist_presence = max(artist_in_title, artist_in_channel)
    if artist_presence < 50:
        return -9999, {"hard_reject_artist_absent": -9999}
        
    base_score = song_match + int(artist_presence * 0.4)
    breakdown["base_match"] = base_score

    raw_title_lower = raw_title.lower()
    if "official audio" in raw_title_lower:
        breakdown["official_audio"] = config.OFFICIAL_AUDIO_BONUS
    elif "official video" in raw_title_lower or "official music video" in raw_title_lower:
        breakdown["official_video"] = 15
    elif "official" in raw_title_lower:
        breakdown["official_signal"] = 10

    if channel.endswith("- topic") and artist_in_channel > 85:
        breakdown["topic_channel"] = config.TOPIC_CHANNEL_BONUS
    elif channel.endswith("vevo") and artist_clean in channel.replace("vevo", ""):
        breakdown["vevo_channel"] = config.VEVO_CHANNEL_BONUS
    elif artist_in_channel > 85:
        breakdown["artist_in_channel"] = 25

    if mb_duration_seconds is not None and result_duration > 0:
        diff_seconds = abs(result_duration - mb_duration_seconds)
        if diff_seconds <= 4:
            breakdown["duration_exact"] = config.DURATION_MATCH_BONUS
        elif diff_seconds <= 12:
            breakdown["duration_close"] = 10
        elif diff_seconds <= 25:
            breakdown["duration_acceptable"] = 0
        else:
            breakdown["duration_mismatch"] = -35

    if view_count > 1_000_000:
        breakdown["high_views"] = 5

    if any(t in channel for t in ["dj", "mix", "bootleg", "edits"]):
        breakdown["dj_channel_penalty"] = -25
    if any(t in title for t in ["lyrics", "letra", "lyric video"]):
        breakdown["lyrics_penalty"] = -20

    total = sum(breakdown.values())
    return total, breakdown


def rank_results(
    results: list[dict],
    artist: str,
    song: str,
    mb_duration_seconds: int | None,
    config: Config,
    min_duration: int | None = None,
    max_duration: int | None = None,
) -> list[tuple[dict, int, dict]]:
    """
    Filters, assesses, and sorts multiple search candidates descending by total score.

    Args:
        results: List of raw candidate dictionaries.
        artist: Target artist name.
        song: Target song title.
        mb_duration_seconds: Expected duration from Registry.
        config: Config settings module instance.
        min_duration: Override minimum runtime constraint.
        max_duration: Override maximum runtime constraint.

    Returns:
        A sorted list of tuples holding the complete metadata, raw score, and breakdown data.
    """
    min_dur = min_duration if min_duration is not None else config.MIN_DURATION_SECONDS
    max_dur = max_duration if max_duration is not None else config.MAX_DURATION_SECONDS

    valid = [r for r in results if r.get("duration") and min_dur <= int(r["duration"]) <= max_dur]
    if not valid:
        return []

    scored = []
    for raw in valid:
        entry = dict(raw)
        score, breakdown = score_youtube_result(entry, artist, song, mb_duration_seconds, config)
        entry["_composite_score"] = score
        entry["_score_breakdown"] = breakdown
        scored.append((entry, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


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