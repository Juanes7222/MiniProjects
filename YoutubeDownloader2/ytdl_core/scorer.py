"""
Candidate scoring and ranking heuristics.

Extracted from ``search.py`` so the scoring logic can be tested and
maintained independently of the search orchestration.

Two public functions:
- ``score_youtube_result`` — score a single candidate against a query
- ``rank_results`` — filter, score, and sort a list of candidates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from rapidfuzz import fuzz

from .config import (
    DEFAULT_FORBIDDEN_TERMS,
    DEFAULT_LIVE_TERMS,
    DEFAULT_SOFT_TERMS,
    Config,
)
from .utils import (
    PhraseMatcher,
    normalize_title,
    phrase_matcher,
    remove_matching_noise,
    strip_featuring,
)

if TYPE_CHECKING:
    from .channels import ChannelTrust

_CATALOG_SOURCES = ("ytmusic_api", "itunes")


def _terms(explicit, default) -> frozenset[str]:
    return frozenset(explicit if explicit is not None else default)


@dataclass(frozen=True)
class QueryContext:
    """Everything about the *query* that does not vary between candidates.

    Ranking a song means running the same fuzzy normalisations and the same
    query-side forbidden/live checks once per candidate, even though they depend
    only on the artist and title being looked for. With forty candidates that is
    forty identical computations of four values. Built once per song instead.
    """

    artist: str
    song: str
    artist_clean: str
    song_clean: str
    query_forbidden: frozenset[str]
    query_live: frozenset[str]
    forbidden_matcher: PhraseMatcher
    live_matcher: PhraseMatcher
    soft_matcher: PhraseMatcher


def build_query_context(
    artist: str,
    song: str,
    config: Optional[Config] = None,
) -> QueryContext:
    """Precompute the per-song invariants used by :func:`score_youtube_result`."""
    cfg = config or Config()
    forbidden = _terms(getattr(cfg, "FORBIDDEN_TERMS", DEFAULT_FORBIDDEN_TERMS), DEFAULT_FORBIDDEN_TERMS)
    live = _terms(getattr(cfg, "LIVE_TERMS", DEFAULT_LIVE_TERMS), DEFAULT_LIVE_TERMS)
    soft = _terms(getattr(cfg, "SOFT_TERMS", DEFAULT_SOFT_TERMS), DEFAULT_SOFT_TERMS)

    forbidden_matcher = phrase_matcher(forbidden)
    live_matcher = phrase_matcher(live)
    query_text = f"{artist} {song}"
    return QueryContext(
        artist=artist,
        song=song,
        artist_clean=normalize_title(strip_featuring(artist.lower())),
        song_clean=normalize_title(strip_featuring(song.lower())),
        query_forbidden=frozenset(forbidden_matcher.find(query_text)),
        query_live=frozenset(live_matcher.find(query_text)),
        forbidden_matcher=forbidden_matcher,
        live_matcher=live_matcher,
        soft_matcher=phrase_matcher(soft),
    )


def _is_official_channel(channel: str) -> bool:
    """True for auto-generated Topic channels and VEVO-style label channels."""
    return channel.endswith("- topic") or channel.endswith("vevo")


def score_youtube_result(
    result: dict,
    artist: str,
    song: str,
    mb_duration_seconds: Optional[int],
    config: Config,
    channel_trust: "Optional[ChannelTrust]" = None,
    context: Optional[QueryContext] = None,
) -> tuple[int, dict[str, int]]:
    """
    Score a single search candidate against the target artist + song.

    Uses a composite heuristic with hard-rejection gates for forbidden
    patterns (covers and remixes), fuzzy title/artist matching, channel
    authority signals, learned channel trust, duration alignment, and
    cross-source consensus.

    ``channel_trust`` is an optional :class:`~ytdl_core.channels.ChannelTrust`.
    When supplied, candidates published on channels that previously delivered
    verified downloads for this artist earn a bonus.

    ``context`` is an optional prebuilt :class:`QueryContext`; supply it when
    scoring many candidates for the same song. It is pure memoisation -- the
    result is identical either way.

    Returns (composite_score, breakdown_dict).
    """
    if context is None:
        context = build_query_context(artist, song, config)

    entry = result
    raw_title = entry.get("title") or ""
    title: str = normalize_title(raw_title)
    channel: str = (entry.get("channel") or entry.get("uploader") or "").lower()
    view_count: int = int(entry.get("view_count") or 0)
    result_duration: int = int(entry.get("duration") or 0)
    breakdown: dict[str, int] = {}
    artist_clean = context.artist_clean
    song_clean = context.song_clean
    title_forbidden = context.forbidden_matcher.find(raw_title)
    query_forbidden = context.query_forbidden
    title_live = context.live_matcher.find(raw_title)
    query_live = context.query_live
    is_live_version = bool(title_live and not query_live)
    live_descriptor_allowed = is_live_version and title_forbidden.issubset({"version", "extended"})
    if title_forbidden and not query_forbidden and not live_descriptor_allowed:
        return -9999, {f"hard_reject_{min(title_forbidden)}": -9999}

    if is_live_version:
        breakdown["live_version"] = config.LIVE_PENALTY

    # Learned channel trust. A channel that has already delivered this artist is
    # the single most reliable signal available, so it is resolved once here and
    # applied on both the catalog fast path and the generic path below.
    trust_bonus = 0
    if channel_trust is not None and channel:
        trust_bonus = channel_trust.bonus_for(channel, artist)
    official_channel = _is_official_channel(channel)

    if entry.get("_source") in _CATALOG_SOURCES:
        title_clean = normalize_title(strip_featuring(raw_title.lower()))
        song_match = int(
            fuzz.token_set_ratio(song_clean, title_clean) * 0.3
            + fuzz.token_sort_ratio(song_clean, title_clean) * 0.3
            + fuzz.ratio(song_clean, title_clean) * 0.4
        )

        ytmusic_artist_names = [normalize_title(a) for a in (entry.get("artists") or []) if a]
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
            if entry.get("_source") == "itunes":
                # Resolved from the label's own catalogue entry via MusicBrainz,
                # so the provenance is already proven -- score it as a flat
                # bonus rather than compounding a fuzzy-match formula.
                api_bonus = config.CATALOG_SOURCE_BONUS
                breakdown["official_catalog_match"] = api_bonus
            else:
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

            if trust_bonus:
                breakdown["trusted_channel"] = trust_bonus
            return sum(breakdown.values()), breakdown

    title_tokens = set(title.split())
    query_song_tokens = set(song_clean.split())

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

    if trust_bonus:
        breakdown["trusted_channel"] = trust_bonus

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

    # Remasters are legitimate on official channels and suspicious everywhere
    # else, so the soft penalty is waived when the publisher is trusted.
    title_soft = context.soft_matcher.find(raw_title)
    if title_soft and not (trust_bonus or official_channel):
        breakdown["remaster_penalty"] = config.SOFT_TERM_PENALTY

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
    channel_trust: "Optional[ChannelTrust]" = None,
) -> list[tuple[dict, int, dict]]:
    """
    Filter, score, and sort candidates descending by composite score.

    Only candidates within [min_duration, max_duration] are considered.

    The per-song :class:`QueryContext` is built once here and threaded through
    every candidate, so the query-side work is done once per song instead of once
    per candidate.

    Returns a list of ``(entry_dict, score, breakdown_dict)`` tuples.
    """
    min_dur = min_duration if min_duration is not None else config.MIN_DURATION_SECONDS
    max_dur = max_duration if max_duration is not None else config.MAX_DURATION_SECONDS

    valid = [r for r in results if r.get("duration") and min_dur <= int(r["duration"]) <= max_dur]
    if not valid:
        return []

    context = build_query_context(artist, song, config)
    scored = []
    for raw in valid:
        entry = dict(raw)
        score, breakdown = score_youtube_result(
            entry,
            artist,
            song,
            mb_duration_seconds,
            config,
            channel_trust,
            context=context,
        )
        entry["_composite_score"] = score
        entry["_score_breakdown"] = breakdown
        scored.append((entry, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
