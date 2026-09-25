from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ytdl_core.jev import JevClassifier, JevEvaluationError


def _candidate(title: str, score: int) -> dict:
    return {
        "title": title,
        "channel": "Official Artist",
        "duration": 200,
        "_composite_score": score,
        "_score_breakdown": {"base_match": score},
    }


def test_selects_candidate_with_highest_probability(monkeypatch):
    classifier = JevClassifier(threshold=0.75)
    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda payload: {
            "candidate_0": {"type": "boolean", "probability": 0.62},
            "candidate_1": {"type": "boolean", "probability": 0.93},
        },
    )

    selected, ranked = classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song (Live)", 120), _candidate("Artist - Song", 110)],
    )

    assert selected is not None
    assert selected["title"] == "Artist - Song"
    assert selected["_decision_selected"] is True
    assert selected["_decision_probability"] == 0.93
    assert selected["_composite_score"] == 93
    assert ranked[0][0]["_heuristic_score"] == 110


def test_repeated_runs_average_probabilities(monkeypatch):
    classifier = JevClassifier(threshold=0.65)
    responses = iter(
        [
            {"candidate_0": {"type": "boolean", "probability": 0.50}},
            {"candidate_0": {"type": "boolean", "probability": 0.70}},
            {"candidate_0": {"type": "boolean", "probability": 0.90}},
        ]
    )
    monkeypatch.setattr(classifier, "_evaluate", lambda payload: next(responses))

    selected, ranked = classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120)],
        runs=3,
    )

    assert selected is not None
    assert selected["_decision_probability"] == pytest.approx(0.70)
    assert selected["_decision_samples"] == [0.50, 0.70, 0.90]
    assert selected["_decision_runs"] == 3
    assert selected["_decision_min"] == 0.50
    assert selected["_decision_max"] == 0.90
    assert ranked[0][0]["_composite_score"] == 70


def test_rejects_invalid_run_count():
    classifier = JevClassifier()

    with pytest.raises(JevEvaluationError, match="at least 1"):
        classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)], runs=0)


def test_candidate_payload_includes_release_metadata():
    candidate = _candidate("Artist - Song", 120)
    candidate.update(
        {
            "description": "Official album recording",
            "upload_date": "20240102",
            "album": "Album",
            "year": "2024",
            "genre": "Gospel",
            "view_count": 1000,
            "is_live": False,
        }
    )

    payload = JevClassifier._candidate_payload(candidate, 0)

    assert payload["description"] == "Official album recording"
    assert payload["upload_date"] == "20240102"
    assert payload["album"] == "Album"
    assert payload["year"] == "2024"
    assert payload["genre"] == "Gospel"
    assert payload["view_count"] == 1000


def test_select_includes_reference_metadata(monkeypatch):
    classifier = JevClassifier()
    captured = {}

    def fake_evaluate(payload):
        captured.update(payload)
        return {"candidate_0": {"type": "boolean", "probability": 0.9}}

    monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
    classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120)],
        reference_metadata={"album": "Reference Album", "year": "2024"},
    )

    reference = captured["state"]["target"]["reference_metadata"]
    assert reference == {"album": "Reference Album", "year": "2024"}


def test_rejects_when_probability_is_below_threshold(monkeypatch):
    classifier = JevClassifier(threshold=0.75)
    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda payload: {
            "candidate_0": {"type": "boolean", "probability": 0.74},
        },
    )

    selected, ranked = classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])

    assert selected is None
    assert ranked[0][0]["_decision_probability"] == 0.74


def test_rejects_invalid_probability(monkeypatch):
    classifier = JevClassifier()
    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda payload: {
            "candidate_0": {"type": "boolean", "probability": 1.5},
        },
    )

    with pytest.raises(JevEvaluationError, match="invalid probability"):
        classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])


def test_bridge_error_identifies_gateway_cause():
    assert "credit card" in JevClassifier._bridge_error("valid credit card required")
    assert "authentication" in JevClassifier._bridge_error("Unauthenticated request")
    assert "rate limit" in JevClassifier._bridge_error("HTTP 429 rate limit")
    assert "temporarily failed" in JevClassifier._bridge_error("HTTP 503 internal server error")


def test_evaluate_uses_utf8_for_candidate_text(monkeypatch):
    classifier = JevClassifier(project_root=Path(__file__).resolve().parents[1])

    def fake_run(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return subprocess.CompletedProcess(command, 0, '{"answers": {}}', "")

    monkeypatch.setattr("ytdl_core.jev.subprocess.run", fake_run)

    assert classifier._evaluate({"text": "♡"}) == {}
