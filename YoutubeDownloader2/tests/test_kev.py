from __future__ import annotations

import pytest
import requests

from ytdl_core import decision_questions as dq
from ytdl_core.jev import JevEvaluationError
from ytdl_core.kev import KevClassifier


def _answers(choice: dict[str, float] | None = None) -> dict:
    answers: dict = {}
    for index, key in enumerate(("candidate_0", "candidate_1")):
        value = 0.95 - index * 0.30
        for dimension in dq.DIMENSIONS:
            answers[dq.question_id(key, dimension.key)] = {
                "type": "noul",
                "noul": max(0.0, value),
            }
    if choice is not None:
        answers[dq.choice_id()] = {
            "type": "choice",
            "choice": max(choice, key=choice.get),
            "probabilities": choice,
            "confidence": 0.77,
        }
    return answers


class _Response:
    status_code = 200
    headers: dict = {}

    def __init__(self, answers: dict) -> None:
        self._answers = answers

    def json(self):
        return {
            "answers": self._answers,
            "latency_ms": 41.5,
            "usage": {"input_tokens": 101, "output_tokens": 161},
        }


def test_kev_uses_local_systemone_api(monkeypatch):
    classifier = KevClassifier(model="kev-4b", threshold=0.60)
    captured = {}

    def fake_post(_session, url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return _Response(_answers({"candidate_0": 0.9, "candidate_1": 0.1}))

    monkeypatch.setattr(requests.Session, "post", fake_post)
    selected, ranked = classifier.select(
        "Artist",
        "Song",
        [
            {
                "title": "Artist - Song",
                "channel": "Official Artist",
                "duration": 200,
                "_composite_score": 100,
                "_score_breakdown": {"base_match": 100},
            }
        ],
    )

    assert selected is not None
    assert selected["_decision_provider"] == "Kev"
    assert selected["_composite_score"] == 95
    assert captured["url"] == "http://127.0.0.1:8009/v1/systemone"
    assert captured["json"]["model"] == "kev-4b"
    assert ranked[0][0]["_decision_selected"] is True


def test_kev_sends_the_same_question_contract_as_jev(monkeypatch):
    """Kev is transport only: the payload must come from decision_questions."""
    captured = {}

    def fake_post(_session, url, json, timeout):
        captured.update(json)
        return _Response(_answers())

    monkeypatch.setattr(requests.Session, "post", fake_post)
    classifier = KevClassifier(threshold=0.60)
    classifier.select(
        "Artist",
        "Song",
        [
            {"title": "Artist - Song", "channel": "Official", "duration": 200},
            {"title": "Artist - Song (Live)", "channel": "Official", "duration": 220},
        ],
        reference_metadata={"album": "Album"},
    )

    assert captured["state"] == {
        "target": {"artist": "Artist", "song": "Song", "reference": {"album": "Album"}}
    }
    for dimension in dq.DIMENSIONS:
        question = captured["questions"][dq.question_id("candidate_0", dimension.key)]
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}
        assert question["instructions"]["candidate"]["key"] == "candidate_0"


def test_kev_does_not_restate_the_prompt(monkeypatch):
    captured = {}

    def fake_post(_session, url, json, timeout):
        captured.update(json)
        return _Response(_answers())

    monkeypatch.setattr(requests.Session, "post", fake_post)
    KevClassifier().select(
        "Artist",
        "Song",
        [{"title": "Artist - Song", "channel": "Official", "duration": 200}],
    )

    blob = str(captured)
    assert "official live version" not in blob
    assert "Prefer studio recordings" not in blob


def test_kev_connection_error_is_actionable(monkeypatch):
    classifier = KevClassifier(url="http://127.0.0.1:9000")

    def fake_post(_session, *args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests.Session, "post", fake_post)

    with pytest.raises(JevEvaluationError, match="server is unavailable"):
        classifier.select(
            "Artist",
            "Song",
            [
                {
                    "title": "Artist - Song",
                    "channel": "Official Artist",
                    "duration": 200,
                    "_composite_score": 100,
                    "_score_breakdown": {},
                }
            ],
        )


def test_kev_retries_on_server_error(monkeypatch):
    statuses = iter([500, 503, 200])
    calls = {"n": 0}

    class Flaky:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.headers: dict = {}

        def json(self):
            return {"answers": _answers()}

    def fake_post(_session, url, json, timeout):
        calls["n"] += 1
        return Flaky(next(statuses))

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr("ytdl_core.kev.time.sleep", lambda *_: None)

    classifier = KevClassifier(threshold=0.60)
    selected, _ = classifier.select(
        "Artist",
        "Song",
        [{"title": "Artist - Song", "channel": "Official", "duration": 200}],
    )

    assert calls["n"] == 3
    assert selected is not None
