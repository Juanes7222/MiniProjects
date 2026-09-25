from __future__ import annotations

import pytest
import requests

from ytdl_core.jev import JevEvaluationError
from ytdl_core.kev import KevClassifier


class _Response:
    status_code = 200

    def json(self):
        return {
            "answers": {
                "candidate_0": {"type": "noul", "noul": 0.82},
            }
        }


def test_kev_uses_local_systemone_api(monkeypatch):
    classifier = KevClassifier(model="kev-4b", threshold=0.60)
    captured = {}

    def fake_post(url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
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
    assert selected["_decision_probability"] == 0.82
    assert captured["url"] == "http://127.0.0.1:8009/v1/systemone"
    assert captured["json"]["model"] == "kev-4b"
    assert captured["json"]["questions"]["candidate_0"]["type"] == "noul"
    instructions = captured["json"]["questions"]["candidate_0"]["instructions"]
    assert "official live version" in instructions
    assert "Prefer studio recordings" in instructions
    assert ranked[0][0]["_decision_selected"] is True


def test_kev_connection_error_is_actionable(monkeypatch):
    classifier = KevClassifier(url="http://127.0.0.1:9000")

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)

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
