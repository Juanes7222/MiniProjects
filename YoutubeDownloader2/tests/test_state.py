from __future__ import annotations

from ytdl_core.result import DownloadResult
from ytdl_core.state import (
    VERIFY_UNKNOWN_FIELDS,
    load_state,
    merge_state_detail,
    save_state,
    state_detail,
)


def _judged_result() -> DownloadResult:
    return DownloadResult(
        artist="Ana Gabriel",
        song="Tú lo decidiste",
        status="downloaded",
        source="youtube",
        matched_title="Ana Gabriel - Tú lo decidiste (official video)",
        fuzzy_score=91,
        heuristic_score=88,
        composite_score=88,
        score_breakdown={"title_match": 40, "duration": 20, "channel_trust": 28},
        decision_provider="kev",
        decision_probability=0.87,
        decision_samples=[0.87, 0.91, 0.83],
        decision_runs=3,
        decision_threshold=0.6,
        selection_method="kev",
        candidates_ranked=12,
        duration_seconds=214,
        file_size_bytes=3_400_000,
        silence_ratio=0.0412,
        fingerprint_verified=True,
        fingerprint_matched_title="Ana Gabriel -- Tú lo decidiste",
    )


def test_custom_state_filename_round_trip(tmp_path) -> None:
    state = {"downloads": {"Artist::Song": {"status": "downloaded"}}}

    save_state(state, tmp_path, "custom.json")

    assert load_state(tmp_path, "custom.json") == state
    assert not (tmp_path / ".download_state.json").exists()


def test_invalid_state_returns_empty_downloads(tmp_path) -> None:
    state_path = tmp_path / ".download_state.json"
    for content in ("not-json", '{"downloads": null}', '{"downloads": []}'):
        state_path.write_text(content, encoding="utf-8")
        assert load_state(tmp_path) == {"downloads": {}}


def test_malformed_entries_are_discarded(tmp_path) -> None:
    state_path = tmp_path / ".download_state.json"
    state_path.write_text(
        '{"downloads": {"Artist::Song": {"status": "downloaded"}, "Broken::Song": null}}',
        encoding="utf-8",
    )

    assert load_state(tmp_path) == {"downloads": {"Artist::Song": {"status": "downloaded"}}}


class TestStateDetail:
    def test_captures_jev_kev_and_heuristic_verdicts(self):
        detail = state_detail(_judged_result())

        assert detail["source_title"] == "Ana Gabriel - Tú lo decidiste (official video)"
        assert detail["heuristic_score"] == 88
        assert detail["heuristic_breakdown"]["channel_trust"] == 28
        assert detail["decision_provider"] == "kev"
        assert detail["decision_probability"] == 0.87
        assert detail["decision_samples"] == [0.87, 0.91, 0.83]
        assert detail["decision_runs"] == 3
        assert detail["decision_threshold"] == 0.6
        assert detail["selection_method"] == "kev"
        assert detail["candidates_ranked"] == 12
        assert detail["fingerprint_matched_title"] == "Ana Gabriel -- Tú lo decidiste"

    def test_unknown_provenance_does_not_erase_a_previous_run(self):
        previous = {"source": "youtube", "source_title": "Tú lo decidiste", "album": "Alma"}
        # Jev cleared every candidate, so the attempt dies before it can learn
        # anything about a winning candidate.
        attempt = DownloadResult(
            "A",
            "S",
            status="failed",
            reason="Jev: no candidate at or above 0.60",
            decision_provider="jev",
        )

        entry = merge_state_detail({"status": "failed"}, previous, state_detail(attempt))

        assert entry["source"] == "youtube"
        assert entry["source_title"] == "Tú lo decidiste"
        assert entry["album"] == "Alma"
        # The reason always describes the current attempt, so it is never inherited.
        assert entry["reason"] == "Jev: no candidate at or above 0.60"
        assert entry["decision_provider"] == "jev"

    def test_reason_is_cleared_on_success(self):
        entry = {"status": "failed", "reason": "Fingerprint did not confirm the song"}

        merge_state_detail(entry, {"reason": "Fingerprint did not confirm"}, state_detail(_judged_result()))

        assert entry["reason"] is None

    def test_preserve_fields_leaves_recorded_run_untouched(self):
        entry = {"status": "verified"}
        previous = {
            "heuristic_score": 88,
            "composite_score": 88,
            "selection_method": "kev",
            "silence_ratio": 0.0412,
            "duration_verified": True,
            "fingerprint_matched_title": "Ana Gabriel -- Tú lo decidiste",
        }
        # What --verify knows: the file, its duration, and the fingerprint verdict.
        verify_pass = DownloadResult(
            "Ana Gabriel",
            "Tú lo decidiste",
            status="verified",
            duration_seconds=214,
            file_size_bytes=3_400_000,
            fingerprint_verified=True,
        )

        merge_state_detail(entry, previous, state_detail(verify_pass), VERIFY_UNKNOWN_FIELDS)

        assert entry["heuristic_score"] == 88
        assert entry["composite_score"] == 88
        assert entry["selection_method"] == "kev"
        assert entry["silence_ratio"] == 0.0412
        assert entry["duration_verified"] is True
        # Provenance the verify pass also never learns survives for the same reason.
        assert entry["fingerprint_matched_title"] == "Ana Gabriel -- Tú lo decidiste"
        # What the pass actually measured is written.
        assert entry["duration_seconds"] == 214
        assert entry["file_size_bytes"] == 3_400_000

