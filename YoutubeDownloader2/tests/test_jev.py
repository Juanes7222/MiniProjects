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
    assert selected["_jev_selected"] is True
    assert selected["_jev_probability"] == 0.93
    assert selected["_composite_score"] == 93
    assert ranked[0][0]["_heuristic_score"] == 110


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
    assert ranked[0][0]["_jev_probability"] == 0.74


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


def test_evaluate_uses_utf8_for_candidate_text(monkeypatch):
    classifier = JevClassifier(project_root=Path(__file__).resolve().parents[1])

    def fake_run(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return subprocess.CompletedProcess(command, 0, '{"answers": {}}', "")

    monkeypatch.setattr("ytdl_core.jev.subprocess.run", fake_run)

    assert classifier._evaluate({"text": "♡"}) == {}
