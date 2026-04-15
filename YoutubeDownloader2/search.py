"""
YouTube / SoundCloud / Bandcamp search, candidate scoring, and best-result selection.

Scoring pipeline (replaces the old fuzzy-only filter):
1. Duration gate: results outside [min_duration, max_duration] are discarded.
2. Composite scoring via score_youtube_result() — channel, title, duration, penalties.
3. Results below SCORE_THRESHOLD_REJECT are rejected outright.
4. The highest-scoring surviving result is returned together with the full ranked list.
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
    """Construct a source-appropriate search query string."""
    if source == "youtube":
        return f'"{song}" "{artist}" official audio'
    return f"{song} {artist}"


def search_source(query: str, source: str, opts: dict) -> list[dict]:
    """
    Run a yt-dlp search against *source* and return raw entry dicts.

    Args:
        query: The search string.
        source: One of "youtube", "soundcloud", "bandcamp".
        opts: Dict with keys: max_results, cookies_browser, proxy.
    """
    max_r = opts.get("max_results", 5)
    prefix = {
        "youtube": f"ytsearch{max_r}",
        "soundcloud": f"scsearch{max_r}",
        "bandcamp": f"bcsearch{max_r}",
    }.get(source, f"ytsearch{max_r}")

    ydl_opts: dict[str, Any] = {
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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
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
    channel: str = (
        entry.get("channel") or entry.get("uploader") or ""
    ).lower()
    view_count: int = int(entry.get("view_count") or 0)
    result_duration: int = int(entry.get("duration") or 0)

    breakdown: dict[str, int] = {}

    # --- POSITIVE SIGNALS ---

    # Topic channel
    if channel.endswith("- topic") and fuzz.partial_ratio(artist.lower(), channel) > 70:
        breakdown["topic_channel"] = config.TOPIC_CHANNEL_BONUS

    # VEVO channel
    if channel.endswith("vevo") and artist.lower() in channel:
        breakdown["vevo_channel"] = config.VEVO_CHANNEL_BONUS

    # Official audio / official video
    if "official audio" in title:
        breakdown["official_audio"] = config.OFFICIAL_AUDIO_BONUS
    elif "official video" in title:
        breakdown["official_video"] = 10

    # Fuzzy match: referencia completa "{artist} {song}" vs título
    ref = f"{artist} {song}"
    fuzzy_ratio = int(fuzz.token_sort_ratio(ref.lower(), title))
    if fuzzy_ratio >= 80:
        breakdown["high_fuzzy"] = config.HIGH_FUZZY_BONUS
    elif fuzzy_ratio >= 65:
        breakdown["medium_fuzzy"] = 10

    # [NUEVO] Fuzzy match solo con el nombre de la canción.
    # Cubre el caso frecuente donde el canal oficial sube "El Gozo del Señor"
    # sin incluir el nombre del artista en el título.
    song_only_ratio = int(fuzz.token_sort_ratio(song.lower(), title))
    if song_only_ratio >= 85 and "high_fuzzy" not in breakdown:
        breakdown["song_title_match"] = 18
    elif song_only_ratio >= 72 and "high_fuzzy" not in breakdown and "medium_fuzzy" not in breakdown:
        breakdown["song_title_match"] = 8

    # [NUEVO] Artista mencionado en el título (ej. "El Gozo del Señor - Abel Zavala").
    # Señal fuerte: el uploader incluyó explícitamente el nombre del artista.
    if fuzz.partial_ratio(artist.lower(), title) > 75:
        breakdown["artist_in_title"] = 15

    # Duración cross-reference con MusicBrainz
    if mb_duration_seconds is not None and result_duration > 0:
        ratio = abs(result_duration - mb_duration_seconds) / mb_duration_seconds
        if ratio <= 0.10:
            breakdown["duration_match"] = config.DURATION_MATCH_BONUS
        elif ratio <= 0.20:
            breakdown["duration_close"] = 10
        elif ratio > 0.40:
            breakdown["duration_mismatch"] = -20

    # Artista en el nombre del canal — aumentado de +15 a +25
    if fuzz.partial_ratio(artist.lower(), channel) > 85:
        breakdown["artist_in_channel"] = 25

    # View count
    if view_count > 1_000_000:
        breakdown["high_views"] = 5

    # --- NEGATIVE SIGNALS ---

    live_terms = ["live", "en vivo", "concert", "concierto", "tour"]
    if any(t in title for t in live_terms):
        breakdown["live_penalty"] = config.LIVE_PENALTY

    cover_terms = ["cover", "karaoke", "tribute"]
    if any(t in title for t in cover_terms):
        breakdown["cover_karaoke_penalty"] = config.COVER_KARAOKE_PENALTY

    reaction_terms = ["reaction", "reacts to", "reaccion"]
    if any(t in title for t in reaction_terms):
        breakdown["reaction_penalty"] = config.REACTION_REMIX_PENALTY

    remix_terms = ["remix", "mashup", "bootleg"]
    if any(t in title for t in remix_terms):
        breakdown["remix_penalty"] = -30

    altered_terms = ["slowed", "reverb", "sped up", "nightcore", "lofi", "lo-fi"]
    if any(t in title for t in altered_terms):
        breakdown["altered_playback_penalty"] = -40

    lyrics_terms = ["lyrics", "letra", "lyric video"]
    if any(t in title for t in lyrics_terms):
        breakdown["lyrics_penalty"] = -20

    album_terms = ["full album", "album completo", "compilation"]
    if any(t in title for t in album_terms):
        breakdown["album_penalty"] = -60

    extended_terms = ["10 hours", "1 hour", "hora", "extended"]
    if any(t in title for t in extended_terms):
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
    console: Console,
    console_lock: threading.Lock | None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    score_threshold: int | None = None,
) -> tuple[dict | None, list[tuple[dict, int, dict]]]:
    """
    Score all duration-valid results, log the candidate table, and return
    the highest-scoring result that passes the score threshold.

    Returns:
        (best_result, ranked_list) where ranked_list is sorted by score desc
        and each element is (result_dict, score, breakdown_dict).
        best_result is None if every candidate was rejected.
    """
    min_dur = min_duration if min_duration is not None else config.MIN_DURATION_SECONDS
    max_dur = max_duration if max_duration is not None else config.MAX_DURATION_SECONDS
    reject_threshold = score_threshold if score_threshold is not None else config.SCORE_THRESHOLD_REJECT

    valid: list[dict] = []
    for raw in results:
        dur = raw.get("duration")
        if dur is None:
            continue
        dur_int = int(dur)
        if min_dur <= dur_int <= max_dur:
            valid.append(raw)

    if not valid:
        return None, []

    scored: list[tuple[dict, int, dict]] = []
    for raw in valid:
        entry = dict(raw)
        score, breakdown = score_youtube_result(entry, artist, song, mb_duration_seconds, config)
        entry["_composite_score"] = score
        entry["_score_breakdown"] = breakdown
        scored.append((entry, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)

    tbl = Table(
        title=f"Candidates for: {artist} -- {song}",
        box=box.SIMPLE,
        show_lines=False,
        expand=False,
    )
    tbl.add_column("#", width=3, style="dim")
    tbl.add_column("Title", max_width=55)
    tbl.add_column("Channel", max_width=30)
    tbl.add_column("Duration", width=10, style="yellow")
    tbl.add_column("Score", width=7)
    tbl.add_column("Top signals", min_width=30, style="dim")

    best_idx = None
    if scored and scored[0][1] >= reject_threshold:
        best_idx = 0

    for i, (entry, sc, bd) in enumerate(scored):
        dur = int(entry.get("duration") or 0)
        dur_str = format_duration(dur)
        title_str = (entry.get("title") or "")[:55]
        channel_str = (entry.get("channel") or entry.get("uploader") or "")[:30]

        top_signals = sorted(bd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        signals_str = ", ".join(
            f"{'+' if v >= 0 else ''}{v} {k}" for k, v in top_signals
        )

        if sc >= 70:
            score_markup = f"[green]{sc}[/green]"
        elif sc >= 30:
            score_markup = f"[yellow]{sc}[/yellow]"
        else:
            score_markup = f"[red]{sc}[/red]"

        row_prefix = ">" if i == best_idx else " "

        tbl.add_row(
            f"{row_prefix}{i + 1}",
            title_str,
            channel_str,
            dur_str,
            score_markup,
            signals_str,
        )

    if console_lock is not None:
        with console_lock:
            console.print(tbl)
    else:
        console.print(tbl)

    if not scored or scored[0][1] < reject_threshold:
        return None, scored

    return scored[0][0], scored