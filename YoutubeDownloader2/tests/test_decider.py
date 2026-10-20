"""Decider as a decision provider.

The point of most of these is that Decider is *not* a special case. It serves the
same ``/v1/systemone`` endpoint as Kev, is scored by the same questions, and
inherits the same transport. A test that pins that is worth more than one that
pins Decider's own behaviour, because the failure it prevents is a second copy
of the scoring logic drifting away from the first.
"""

from __future__ import annotations

import pytest
import requests

from ytdl_core import decision_questions as dq
from ytdl_core.decider import DeciderClassifier
from ytdl_core.jev import JevClassifier, JevEvaluationError
from ytdl_core.kev import KevClassifier
from ytdl_core.systemone import SystemOneClassifier


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
            "latency_ms": 12.5,
            "usage": {"input_tokens": 80, "output_tokens": 40},
        }


def _candidate(title: str = "Artist - Song") -> dict:
    return {"title": title, "channel": "Official", "duration": 200, "_composite_score": 100}


class TestSharedContract:
    """What Decider must have in common with Kev to be interchangeable."""

    def test_it_shares_the_system_one_transport(self):
        assert issubclass(DeciderClassifier, SystemOneClassifier)
        assert issubclass(KevClassifier, SystemOneClassifier)

    def test_it_speaks_the_typesafe_dialect_not_the_gateway_one(self):
        # Both local servers answer `noul`. Only the hosted gateway, reached
        # through Vercel, spells the same thing `boolean`.
        assert DeciderClassifier().dialect == dq.TYPESAFE_DIALECT
        assert DeciderClassifier().dialect.noul == "noul"
        assert JevClassifier().dialect.noul == "boolean"

    def test_it_is_not_a_kev(self):
        # A subclass would inherit Kev's URL, model name and error strings, so
        # every Decider failure would be reported as a Kev one.
        assert not issubclass(DeciderClassifier, KevClassifier)
        assert DeciderClassifier.provider_name == "Decider"
        assert DeciderClassifier.start_flag == "--decider"


class TestRequest:
    def test_it_posts_to_systemone_with_the_same_question_contract(self, monkeypatch):
        captured: dict = {}

        def fake_post(_session, url, json, timeout):
            captured.update({"url": url, "json": json})
            return _Response(_answers())

        monkeypatch.setattr(requests.Session, "post", fake_post)
        classifier = DeciderClassifier(url="http://127.0.0.1:8010", threshold=0.60)
        classifier.select("Artist", "Song", [_candidate(), _candidate("Artist - Song (Live)")])

        assert captured["url"] == "http://127.0.0.1:8010/v1/systemone"
        question = captured["json"]["questions"][dq.question_id("candidate_0", "song")]
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}

    def test_the_state_never_carries_the_candidates(self, monkeypatch):
        captured: dict = {}

        def fake_post(_session, url, json, timeout):
            captured.update(json)
            return _Response(_answers())

        monkeypatch.setattr(requests.Session, "post", fake_post)
        DeciderClassifier().select("Artist", "Song", [_candidate(), _candidate("Other - Song")])

        assert "candidate" not in captured["state"]
        assert captured["state"]["target"]["artist"] == "Artist"

    def test_it_does_not_restate_the_prompt(self, monkeypatch):
        captured: dict = {}

        def fake_post(_session, url, json, timeout):
            captured.update(json)
            return _Response(_answers())

        monkeypatch.setattr(requests.Session, "post", fake_post)
        DeciderClassifier().select("Artist", "Song", [_candidate()])

        blob = str(captured)
        assert "Prefer studio recordings" not in blob
        assert "official live version" not in blob


class TestModelNameDiscovery:
    """decider.serve names the checkpoint after the version in its own config.

    There is no stable string to hardcode the way `kev-latest` is, so the client
    asks. Getting this wrong is a 404 that reads like a missing endpoint.
    """

    def test_it_omits_the_model_field_when_the_server_cannot_be_asked(self, monkeypatch):
        captured: dict = {}

        def fake_get(self, url, timeout=None):
            raise requests.ConnectionError("no server")

        def fake_post(_session, url, json, timeout):
            captured.update(json)
            return _Response(_answers())

        monkeypatch.setattr(requests.Session, "get", fake_get)
        monkeypatch.setattr(requests.Session, "post", fake_post)
        DeciderClassifier().select("Artist", "Song", [_candidate()])

        assert "model" not in captured

    def test_it_asks_the_server_and_remembers_the_answer(self, monkeypatch):
        gets = {"n": 0}

        class Info:
            status_code = 200

            def json(self):
                return {"model": "decider-v2.1", "device": "cuda:0", "temperature": 1.36}

        def fake_get(self, url, timeout=None):
            gets["n"] += 1
            return Info()

        def fake_post(_session, url, json, timeout):
            return _Response(_answers())

        monkeypatch.setattr(requests.Session, "get", fake_get)
        monkeypatch.setattr(requests.Session, "post", fake_post)
        classifier = DeciderClassifier()

        assert classifier._request_model() == "decider-v2.1"
        # A second song must not pay for the lookup again.
        classifier._request_model()
        assert gets["n"] == 1

    def test_an_explicit_model_wins_and_skips_the_lookup(self, monkeypatch):
        def fail(self, url, timeout=None):
            raise AssertionError("must not look up a model that was configured")

        monkeypatch.setattr(requests.Session, "get", fail)
        assert DeciderClassifier(model="decider-v9")._request_model() == "decider-v9"

    def test_it_reads_a_models_list_when_that_is_the_shape(self, monkeypatch):
        class Info:
            status_code = 200

            def json(self):
                return {"models": [{"name": "decider-v11"}, {"name": "other"}]}

        monkeypatch.setattr(requests.Session, "get", lambda self, url, timeout=None: Info())
        assert DeciderClassifier()._discover_model() == "decider-v11"


class TestFailures:
    def test_connection_error_names_decider_and_the_flag_that_starts_it(self, monkeypatch):
        classifier = DeciderClassifier(url="http://127.0.0.1:9000")

        def fake_post(_session, *args, **kwargs):
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(requests.Session, "post", fake_post)

        with pytest.raises(JevEvaluationError, match="Decider server is unavailable"):
            classifier.select("Artist", "Song", [_candidate()])

        with pytest.raises(JevEvaluationError, match="--decider"):
            classifier.select("Artist", "Song", [_candidate()])

    def test_a_timeout_points_at_the_merged_flag(self, monkeypatch):
        def fake_post(_session, *args, **kwargs):
            raise requests.Timeout("slow")

        monkeypatch.setattr(requests.Session, "post", fake_post)

        with pytest.raises(JevEvaluationError, match="--decision-timeout"):
            DeciderClassifier()._evaluate({}, {})

    def test_413_answers_with_the_knob_that_fixes_it(self, monkeypatch):
        class Rejected:
            status_code = 413
            headers: dict = {}

            def json(self):
                return {}

        monkeypatch.setattr(requests.Session, "post", lambda *a, **k: Rejected())

        with pytest.raises(JevEvaluationError, match="--decision-questions"):
            DeciderClassifier()._evaluate({}, {})

    def test_there_is_no_auth_hint_to_invent(self):
        # decider.serve binds to 127.0.0.1 and ships without authentication.
        assert DeciderClassifier.auth_env == ""
        assert KevClassifier.auth_env == "KEV_API_KEY"


class TestTelemetry:
    def test_it_accumulates_what_the_server_reported(self, monkeypatch):
        monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Response(_answers()))
        classifier = DeciderClassifier()
        classifier._evaluate({}, {})
        classifier._evaluate({}, {})

        telemetry = classifier.telemetry()
        assert telemetry["requests"] == 2
        assert telemetry["tokens"] == 240
        assert telemetry["max_latency_ms"] == 12.5
