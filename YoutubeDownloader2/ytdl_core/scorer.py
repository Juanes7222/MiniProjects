"""
Candidate scoring and ranking heuristics.

Extracted from ``search.py`` so the scoring logic can be tested and
maintained independently of the search orchestration.

Two public functions:
- ``score_youtube_result`` — score a single candidate against a query
- ``rank_results`` — filter, score, and sort a list of candidates
"""

from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz

from .config import Config
from .utils import contains_forbidden_phrase, normalize_title, remove_matching_noise, strip_featuring


def score_youtube_result(
    result: dict,
    artist: str,
    song: str,
    mb_duration_seconds: Optional[int],
    config: Config,
) -> tuple[int, dict[str, int]]:
    """
    Score a single search candidate against the target artist + song.

    Uses a composite heuristic with hard-rejection gates for forbidden
    patterns (covers, live, remix), fuzzy title/artist matching, channel
    authority signals, duration alignment, and cross-source consensus.

    Returns (composite_score, breakdown_dict).
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

    # ------------------------------------------------------------------
    # Fast-path: YouTube Music API catalog entries
    # ------------------------------------------------------------------
    if entry.get("_source") == "ytmusic_api":
        title_clean = normalize_title(strip_featuring(raw_title.lower()))
        song_match = int(
            fuzz.token_set_ratio(song_clean, title_clean) * 0.3
            + fuzz.token_sort_ratio(song_clean, title_clean) * 0.3
            + fuzz.ratio(song_clean, title_clean) * 0.4
        )

        ytmusic_artist_names = [
            normalize_title(a) for a in (entry.get("artists") or []) if a
        ]
        ytmusic_artist_names.append(normalize_title(channel))
        artist_match = max(
            (
                int(
                    fuzz.token_set_ratio(artist_clean, name) * 0.3
                    + fuzz.token_sort_ratio(artist_clean, name) * 0.3
                    + fuzz.ratio(artist_clean, name) * 0.4
                )
                for name in ytmusic_artist_names
                if name
            ),
            default=0,
        )

        if song_match >= 80 and artist_match >= 80:
            artist_factor = artist_match / 100.0
            api_bonus = int(25 + (song_match * artist_match) ** 0.5 * 0.3 * artist_factor)

            breakdown["official_ytmusic_api"] = api_bonus
            breakdown["catalog_match"] = song_match
            breakdown["artist_match"] = artist_match

            if mb_duration_seconds is not None and result_duration > 0:
                diff_seconds = abs(result_duration - mb_duration_seconds)
                if diff_seconds <= 4:
                    breakdown["duration_perfect"] = config.DURATION_MATCH_BONUS
                elif diff_seconds <= 12:
                    breakdown["duration_close"] = 10
                elif diff_seconds > 25:
                    breakdown["duration_mismatch"] = -35

            source_count = entry.get("_source_count", 1)
            if source_count > 1:
                breakdown["cross_source_consensus"] = min(20, (source_count - 1) * 10)

            return sum(breakdown.values()), breakdown

    # ------------------------------------------------------------------
    # Standard yt-dlp scraping path
    # ------------------------------------------------------------------
    title_tokens = set(title.split())
    query_song_tokens = set(song_clean.split())

    bad_found = contains_forbidden_phrase(raw_title, config.FORBIDEN_TERMS)
    query_text = f"{artist} {song}"
    bad_in_query = contains_forbidden_phrase(query_text, config.FORBIDEN_TERMS)

    if bad_found and not bad_in_query:
        return -9999, {f"hard_reject_{bad_found}": -9999}

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

    source_count = entry.get("_source_count", 1)
    if source_count > 1:
        breakdown["cross_source_consensus"] = min(20, (source_count - 1) * 10)

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
    mb_duration_seconds: Optional[int],
    config: Config,
    min_duration: Optional[int] = None,
    max_duration: Optional[int] = None,
) -> list[tuple[dict, int, dict]]:
    """
    Filter, score, and sort candidates descending by composite score.

    Only candidates within [min_duration, max_duration] are considered.

    Returns a list of ``(entry_dict, score, breakdown_dict)`` tuples.
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
