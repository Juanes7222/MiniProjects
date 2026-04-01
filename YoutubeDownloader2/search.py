"""
YouTube / SoundCloud / Bandcamp search and result-filtering logic.

Filtering pipeline per candidate result:
  1. Duration gate  (min_duration <= dur <= max_duration)
  2. Fuzzy score    (rapidfuzz token_sort_ratio vs "{artist} {song}")
     - score >= threshold  → ACCEPT
     - score <  threshold  → soft-reject, kept as fallback
  3. If no accepted result, return the first duration-valid entry as fallback.
"""

from __future__ import annotations

from typing import Any, Optional

import yt_dlp
from rapidfuzz import fuzz

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
        query:   The search string.
        source:  One of "youtube", "soundcloud", "bandcamp".
        opts:    Dict with keys: max_results, cookies_browser, proxy.
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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
            info: Any = ydl.extract_info(f"{prefix}:{query}", download=False)
            if info and "entries" in info:
                return [e for e in info.get("entries", []) if e is not None]
    except Exception:
        pass
    return []


def filter_results(
    results: list[dict],
    artist: str,
    song: str,
    max_duration: int,
    min_duration: int,
    fuzzy_threshold: int,
) -> tuple[Optional[dict], list[dict]]:
    """
    Apply duration and fuzzy-match filters to a list of search results.

    Returns:
        (best_result, all_candidates_annotated)
        best_result is None if every candidate was rejected.
    """
    ref = f"{artist} {song}"
    candidates: list[dict] = []

    for idx, raw in enumerate(results):
        entry = dict(raw)
        entry["_index"] = idx
        duration = entry.get("duration")
        title = entry.get("title") or ""

        if duration is None:
            entry.update(_fuzzy_score=0, _accepted=False, _reason="no duration info")
            candidates.append(entry)
            continue

        dur_int = int(duration)
        if not (min_duration <= dur_int <= max_duration):
            lo = format_duration(min_duration)
            hi = format_duration(max_duration)
            entry.update(
                _fuzzy_score=0,
                _accepted=False,
                _reason=f"duration {format_duration(dur_int)} outside [{lo}–{hi}]",
            )
            candidates.append(entry)
            continue

        score = int(fuzz.token_sort_ratio(ref.lower(), title.lower()))
        if score >= fuzzy_threshold:
            entry.update(
                _fuzzy_score=score,
                _accepted=True,
                _reason=f"score={score} >= threshold={fuzzy_threshold}",
            )
        else:
            entry.update(
                _fuzzy_score=score,
                _accepted=False,
                _reason=f"score={score} < threshold={fuzzy_threshold} (soft-reject)",
            )
        candidates.append(entry)

    # Best accepted result (highest fuzzy score)
    accepted = [c for c in candidates if c.get("_accepted")]
    if accepted:
        return max(accepted, key=lambda x: x["_fuzzy_score"]), candidates

    # Fallback: first duration-valid entry
    valid = [
        c
        for c in candidates
        if c.get("duration") is not None
        and min_duration <= int(c["duration"]) <= max_duration
    ]
    if valid:
        fallback = valid[0]
        fallback["_reason"] = "fallback: no fuzzy match — using first duration-valid result"
        return fallback, candidates

    return None, candidates