"""
YouTube / SoundCloud / Bandcamp search, candidate scoring, and best-result selection.
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
from .utils import format_duration, normalize_title, strip_featuring


def build_search_query(artist: str, song: str, source: str) -> str:
    if source == "youtube":
        return f'"{song}" "{artist}" official audio'
    return f"{song} {artist}"


def search_source(query: str, source: str, opts: dict) -> list[dict]:
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
    seen_ids = set()
    deduped = []
    for r in results:
        vid = r.get("id") or r.get("url")
        if vid:
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
        # also deduplicate by title, channel and duration signature
        sig = (r.get("title"), r.get("channel"), r.get("duration"))
        if sig in seen_ids:
            continue
        seen_ids.add(sig)
        deduped.append(r)
    return deduped


def search_all_sources(artist: str, song: str, sources: list[str], opts: dict) -> list[dict]:
    all_results = []
    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as executor:
        futures = {}
        for src in sources:
            futures[executor.submit(search_with_variants, artist, song, src, opts)] = src
            
        for future in as_completed(futures):
            src = futures[future]
            res = future.result()
            # Inject source to know where it came from if needed
            for r in res:
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
    entry = dict(result)
    raw_title = entry.get("title") or ""
    title: str = normalize_title(raw_title)
    channel: str = (entry.get("channel") or entry.get("uploader") or "").lower()
    view_count: int = int(entry.get("view_count") or 0)
    result_duration: int = int(entry.get("duration") or 0)
    breakdown: dict[str, int] = {}
    
    artist_clean = normalize_title(strip_featuring(artist.lower()))
    song_clean = normalize_title(strip_featuring(song.lower()))

    # 1. Channel scoring
    # Exact or near exact topic channel
    if channel.endswith("- topic") and fuzz.partial_ratio(artist_clean, channel) > 85:
        breakdown["topic_channel"] = config.TOPIC_CHANNEL_BONUS

    # VEVO channel
    if channel.endswith("vevo") and artist_clean in channel.replace("vevo", ""):
        breakdown["vevo_channel"] = config.VEVO_CHANNEL_BONUS
        
    # Artist in channel name
    if fuzz.partial_ratio(artist_clean, channel) > 85:
        breakdown["artist_in_channel"] = 25

    # 2. Title scoring (Official content)
    raw_title_lower = raw_title.lower()
    if "official audio" in raw_title_lower:
        breakdown["official_audio"] = config.OFFICIAL_AUDIO_BONUS
    elif "official video" in raw_title_lower or "official music video" in raw_title_lower:
        breakdown["official_video"] = 15

    # 3. Fuzzy matching (Avoid overlapping scores)
    ref = f"{artist_clean} {song_clean}"
    fuzzy_ratio = int(fuzz.token_sort_ratio(ref, title))
    song_partial = int(fuzz.partial_ratio(song_clean, title))
    song_only_ratio = int(fuzz.token_sort_ratio(song_clean, title))
    
    length_coverage = len(song_clean) / max(len(title), 1)

    # Choose the best matching method to avoid double counting
    if fuzzy_ratio >= 85:
        breakdown["high_fuzzy"] = config.HIGH_FUZZY_BONUS
    elif fuzzy_ratio >= 70:
        breakdown["medium_fuzzy"] = 15
    elif song_partial >= 90:
        if length_coverage >= 0.35:
            breakdown["song_in_title"] = 35
        elif length_coverage >= 0.15:
            breakdown["song_in_title"] = 10   
        # breakdown["song_exact_in_title"] = 30
        if fuzz.partial_ratio(artist_clean, title) > 80:
            breakdown["artist_in_title"] = 15
    elif song_only_ratio >= 80:
        breakdown["song_title_match"] = 20
        if fuzz.partial_ratio(artist_clean, title) > 80:
            breakdown["artist_in_title"] = 15
            
    song_presence = max(
        fuzz.partial_ratio(song_clean, title),
        fuzz.token_sort_ratio(song_clean, title),
    )

    if song_presence < 45:
        breakdown["song_absent_penalty"] = -80
    elif song_presence < 60:
        breakdown["song_weak_match_penalty"] = -50

    if song_presence >= 45:
        ref = f"{artist_clean} {song_clean}"
        fuzzy_ratio = int(fuzz.token_sort_ratio(ref, title))
        
    artist_in_title_ratio   = fuzz.partial_ratio(artist_clean, title)
    artist_in_channel_ratio = fuzz.partial_ratio(artist_clean, channel)
    artist_presence = max(artist_in_title_ratio, artist_in_channel_ratio)

    if artist_presence < 50:
        breakdown["artist_absent_penalty"] = -50
    elif artist_presence < 65:
        breakdown["artist_weak_penalty"] = -20

    # 4. Duration scoring
    if mb_duration_seconds is not None and result_duration > 0:
        ratio = abs(result_duration - mb_duration_seconds) / mb_duration_seconds
        if ratio <= 0.08:
            breakdown["duration_exact"] = config.DURATION_MATCH_BONUS
        elif ratio <= 0.20:
            breakdown["duration_close"] = 10
        elif ratio <= 0.40:
            breakdown["duration_far"] = -10
        else:
            breakdown["duration_mismatch"] = -35

    if view_count > 1_000_000:
        breakdown["high_views"] = 5

    # Penalties
    if any(t in title for t in ["live", "en vivo", "concert", "concierto", "tour"]):
        breakdown["live_penalty"] = config.LIVE_PENALTY

    if any(t in channel for t in ["dj", "mix", "bootleg", "edits"]):
        breakdown["dj_channel_penalty"] = -25

    if any(t in title for t in ["cover", "karaoke", "tribute"]):
        breakdown["cover_karaoke_penalty"] = config.COVER_KARAOKE_PENALTY

    if any(t in title for t in ["reaction", "reacts to", "reaccion"]):
        breakdown["reaction_penalty"] = config.REACTION_REMIX_PENALTY

    if any(t in title for t in ["remix", "mashup", "bootleg"]):
        breakdown["remix_penalty"] = -30

    if any(t in title for t in ["slowed", "reverb", "sped up", "nightcore", "lofi", "lo-fi"]):
        breakdown["altered_playback_penalty"] = -40

    if any(t in title for t in ["lyrics", "letra", "lyric video"]):
        breakdown["lyrics_penalty"] = -20

    if any(t in title for t in ["full album", "album completo", "compilation"]):
        breakdown["album_penalty"] = -60

    if any(t in title for t in ["10 hours", "1 hour", "hora", "extended"]):
        breakdown["extended_penalty"] = -60

    total = sum(breakdown.values())
    entry["_score_breakdown"] = breakdown
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
