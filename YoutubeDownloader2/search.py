"""
YouTube / SoundCloud / Bandcamp search, candidate scoring, and best-result selection.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import yt_dlp
from rapidfuzz import fuzz
from rich import box
from rich.console import Console
from rich.table import Table

from config import Config
from utils import format_duration


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
        "quiet": True, "no_warnings": True,
        "skip_download": True, "extract_flat": False, "noplaylist": True,
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


def score_youtube_result(
    result: dict,
    artist: str,
    song: str,
    mb_duration_seconds: int | None,
    config: Config,
) -> tuple[int, dict[str, int]]:
    entry = dict(result)
    title: str = (entry.get("title") or "").lower()
    channel: str = (entry.get("channel") or entry.get("uploader") or "").lower()
    view_count: int = int(entry.get("view_count") or 0)
    result_duration: int = int(entry.get("duration") or 0)
    breakdown: dict[str, int] = {}

    if channel.endswith("- topic") and fuzz.partial_ratio(artist.lower(), channel) > 70:
        breakdown["topic_channel"] = config.TOPIC_CHANNEL_BONUS

    if channel.endswith("vevo") and artist.lower() in channel:
        breakdown["vevo_channel"] = config.VEVO_CHANNEL_BONUS

    if "official audio" in title:
        breakdown["official_audio"] = config.OFFICIAL_AUDIO_BONUS
    elif "official video" in title:
        breakdown["official_video"] = 10

    ref = f"{artist} {song}"
    fuzzy_ratio = int(fuzz.token_sort_ratio(ref.lower(), title))
    if fuzzy_ratio >= 80:
        breakdown["high_fuzzy"] = config.HIGH_FUZZY_BONUS
    elif fuzzy_ratio >= 65:
        breakdown["medium_fuzzy"] = 10
        
    song_partial = int(fuzz.partial_ratio(song.lower(), title))
    if song_partial >= 88:
        breakdown["song_exact_in_title"] = 30
    elif song_partial >= 75:
        breakdown["song_exact_in_title"] = 15

    song_only_ratio = int(fuzz.token_sort_ratio(song.lower(), title))
    if song_only_ratio >= 85 and "high_fuzzy" not in breakdown and "song_exact_in_title" not in breakdown:
        breakdown["song_title_match"] = 18
    elif song_only_ratio >= 72 and "high_fuzzy" not in breakdown and "medium_fuzzy" not in breakdown and "song_exact_in_title" not in breakdown:
        breakdown["song_title_match"] = 8

    if fuzz.partial_ratio(artist.lower(), title) > 75:
        breakdown["artist_in_title"] = 15

    if mb_duration_seconds is not None and result_duration > 0:
        ratio = abs(result_duration - mb_duration_seconds) / mb_duration_seconds
        if ratio <= 0.10:
            breakdown["duration_match"] = config.DURATION_MATCH_BONUS
        elif ratio <= 0.20:
            breakdown["duration_close"] = 10
        elif ratio > 0.40:
            breakdown["duration_mismatch"] = -20

    if fuzz.partial_ratio(artist.lower(), channel) > 80:
        breakdown["artist_in_channel"] = 25

    if view_count > 1_000_000:
        breakdown["high_views"] = 5
        

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
    min_dur = min_duration if min_duration is not None else config.MIN_DURATION_SECONDS
    max_dur = max_duration if max_duration is not None else config.MAX_DURATION_SECONDS
    reject_threshold = score_threshold if score_threshold is not None else config.SCORE_THRESHOLD_REJECT

    valid = [r for r in results if r.get("duration") and min_dur <= int(r["duration"]) <= max_dur]
    if not valid:
        return None, []

    scored = []
    for raw in valid:
        entry = dict(raw)
        score, breakdown = score_youtube_result(entry, artist, song, mb_duration_seconds, config)
        entry["_composite_score"] = score
        entry["_score_breakdown"] = breakdown
        scored.append((entry, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Tabla Rich solo en modo CLI (console != None)
    if console is not None:
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
                f"[green]{sc}[/green]" if sc >= 70
                else f"[yellow]{sc}[/yellow]" if sc >= 30
                else f"[red]{sc}[/red]"
            )
            tbl.add_row(
                f"{'>' if i == best_idx else ' '}{i+1}",
                (entry.get("title") or "")[:55],
                (entry.get("channel") or entry.get("uploader") or "")[:30],
                format_duration(dur), score_markup, signals,
            )

        if console_lock:
            with console_lock:
                console.print(tbl)
        else:
            console.print(tbl)

    if not scored or scored[0][1] < reject_threshold:
        return None, scored

    return scored[0][0], scored