from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .config import Config
from .json_io import write_json_atomic

if TYPE_CHECKING:
    from .result import DownloadResult

_DEFAULT_STATE_FILENAME = Config().STATE_FILE


def load_state(
    output_dir: Path,
    state_filename: str | None = None,
) -> dict[str, Any]:
    path = Path(output_dir) / (state_filename or _DEFAULT_STATE_FILENAME)
    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"downloads": {}}
    if not isinstance(state, dict) or not isinstance(state.get("downloads"), dict):
        return {"downloads": {}}
    state["downloads"] = {
        key: entry for key, entry in state["downloads"].items() if isinstance(entry, dict)
    }
    return state


def save_state(
    state: dict[str, Any],
    output_dir: Path,
    state_filename: str | None = None,
) -> None:
    path = Path(output_dir) / (state_filename or _DEFAULT_STATE_FILENAME)
    write_json_atomic(path, state)


# Fields that only become known once a candidate has actually been chosen. They
# are written only when truthy so an attempt that dies before the selection
# stage does not erase what a previous run already learned.
PROVENANCE_FIELDS = frozenset(
    {
        "source",
        "source_title",
        "decision_probability",
        "fingerprint_matched_title",
        "album",
        "year",
        "genre",
    }
)

# Telemetry a --verify run cannot possibly know: it rebuilds the result from the
# file on disk and never re-runs the search, the decider or the post-download
# checks, so its defaults are "not measured", not "measured and found zero".
VERIFY_UNKNOWN_FIELDS = frozenset(
    {
        "fuzzy_score",
        "heuristic_score",
        "heuristic_breakdown",
        "composite_score",
        "decision_provider",
        "decision_samples",
        "decision_runs",
        "decision_threshold",
        "selection_method",
        "candidates_ranked",
        "fallback_used",
        "silence_ratio",
        "duration_verified",
    }
)


def state_detail(result: DownloadResult) -> dict[str, Any]:
    """Project a DownloadResult onto the extra fields kept in the state file.

    These are the same numbers the run report exports, kept next to the
    download so a run can be audited later without the report: which title
    actually won, what the heuristic scored it, and what Jev/Kev decided.
    """
    return {
        # Which candidate won, and what it claimed to be.
        "source": result.source,
        "source_title": result.matched_title,
        "fuzzy_score": result.fuzzy_score,
        # The heuristic ranking, before any model overwrote the composite score.
        "heuristic_score": result.heuristic_score,
        "heuristic_breakdown": dict(result.score_breakdown or {}),
        "composite_score": result.composite_score,
        # The Jev/Kev verdict, and the run configuration that produced it.
        "decision_provider": result.decision_provider,
        "decision_probability": result.decision_probability,
        "decision_samples": [round(float(s), 4) for s in (result.decision_samples or [])],
        "decision_runs": result.decision_runs,
        "decision_threshold": result.decision_threshold,
        "selection_method": result.selection_method,
        "candidates_ranked": result.candidates_ranked,
        "fallback_used": result.fallback_used,
        # Fingerprint, including the title AcoustID thinks it actually is.
        "fingerprint_matched_title": result.fingerprint_matched_title,
        # What the downloaded audio turned out to be.
        "duration_seconds": result.duration_seconds,
        "duration_verified": result.duration_verified,
        "silence_ratio": round(float(result.silence_ratio), 4),
        "file_size_bytes": result.file_size_bytes,
        # Reference metadata, when the lookup succeeded.
        "musicbrainz_enriched": result.musicbrainz_enriched,
        "album": result.album,
        "year": result.year,
        "genre": result.genre,
        # Why this attempt ended the way it did; None means "no failure".
        "reason": result.reason,
    }


def merge_state_detail(
    entry: dict[str, Any],
    previous: dict[str, Any] | None,
    detail: dict[str, Any],
    preserve_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fold a ``state_detail`` projection into a fresh state entry, in place.

    ``previous`` is the entry as it stood before this write. It supplies the
    values this attempt never learned: unknown provenance, and -- for callers
    that pass ``preserve_fields`` -- telemetry this attempt could not measure.
    """
    known = previous or {}
    carried = frozenset(preserve_fields or ())
    for name, value in detail.items():
        if name in carried:
            if name in known:
                entry[name] = known[name]
            continue
        if name in PROVENANCE_FIELDS and not value and known.get(name):
            value = known[name]
        entry[name] = value
    return entry

