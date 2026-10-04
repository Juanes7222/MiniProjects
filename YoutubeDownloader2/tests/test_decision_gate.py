"""Tests for decision-provider robustness: admission control and giving up."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import requests

from unittest.mock import patch

import pytest

from ytdl_core.config import Config
from ytdl_core.jev import DecisionGate, JevEvaluationError
from ytdl_core.kev import KevClassifier


class TestDecisionGate:
    def test_admits_up_to_the_limit(self):
        gate = DecisionGate(max_in_flight=2)
        assert gate.admit() is True
        assert gate.admit() is True
        # A third has to wait for a slot; it must not be refused outright.
        refused = []

        def try_admit():
            if not gate.admit():
                refused.append(True)

        thread = threading.Thread(target=try_admit, daemon=True)
        thread.start()
        time.sleep(0.05)
        assert refused == []
        gate.release()
        thread.join(timeout=5)
        assert refused == []
        gate.release()

    def test_gives_up_after_consecutive_failures(self):
        gate = DecisionGate(failure_threshold=3)
        assert gate.record_failure() is False
        assert gate.record_failure() is False
        assert gate.record_failure() is True
        # Further attempts are refused immediately rather than timing out.
        assert gate.admit() is False

    def test_success_resets_the_failure_run(self):
        gate = DecisionGate(failure_threshold=3)
        gate.record_failure()
        gate.record_failure()
        gate.record_success()
        assert gate.record_failure() is False
        assert gate.admit() is True
        gate.release()

    def test_notice_is_offered_once(self):
        gate = DecisionGate(failure_threshold=2)
        gate.record_failure()
        assert gate.give_up_notice() is None
        gate.record_failure()
        first = gate.give_up_notice()
        assert first is not None and "heuristic" in first
        # Asking again must not repeat the warning for every remaining song.
        assert gate.give_up_notice() is None

    def test_no_notice_while_healthy(self):
        gate = DecisionGate()
        gate.record_success()
        assert gate.give_up_notice() is None

    def test_release_returns_the_slot(self):
        gate = DecisionGate(max_in_flight=1)
        assert gate.admit() is True
        gate.release()
        assert gate.admit() is True
        gate.release()


class TestKevTimeoutHandling:
    def _classifier(self, **kwargs):
        return KevClassifier(
            url="http://127.0.0.1:8009",
            timeout_seconds=kwargs.pop("timeout_seconds", 1),
            max_attempts=kwargs.pop("max_attempts", 3),
            **kwargs,
        )

    def test_timeout_is_not_retried(self):
        """A slow model needs a longer timeout, not three recomputes.

        Retrying a timeout re-runs the same forward pass from scratch and throws
        away the GPU work already done -- which is how one answer became three
        minutes of work before the first download.
        """
        classifier = self._classifier()
        with patch("ytdl_core.kev.requests.Session.post", side_effect=requests.Timeout("slow")) as post:
            with pytest.raises(JevEvaluationError, match="kev-timeout"):
                classifier._evaluate({}, {})
        assert post.call_count == 1

    def test_timeout_is_not_retried_even_with_attempts_available(self):
        classifier = self._classifier(max_attempts=3)
        with patch("ytdl_core.kev.requests.Session.post", side_effect=requests.Timeout("slow")) as post:
            with pytest.raises(JevEvaluationError):
                classifier._evaluate({}, {})
        assert post.call_count == 1

    def test_slots_are_always_returned(self):
        """With one slot, a leaked admission deadlocks every later song."""
        classifier = self._classifier(max_attempts=1)

        class Response:
            status_code = 200

            def json(self):
                return {"answers": {"q": {"noul": 0.9}}}

        for _ in range(5):
            with patch("ytdl_core.kev.requests.Session.post", return_value=Response()):
                classifier._evaluate({}, {})
        # Would hang here if the slot leaked.
        assert classifier.gate.successes == 5

    def test_gives_up_after_repeated_failures_then_stops_calling(self):
        classifier = self._classifier(max_attempts=1)
        with patch("ytdl_core.kev.requests.Session.post", side_effect=TimeoutError("slow")) as post:
            for _ in range(4):
                with pytest.raises(JevEvaluationError):
                    classifier._evaluate({}, {})
        # The gate trips after the threshold and the server stops being called.
        assert post.call_count < 4

    def test_connection_error_is_reported_immediately(self):
        import requests

        classifier = self._classifier(max_attempts=3)
        with patch("ytdl_core.kev.requests.Session.post", side_effect=requests.ConnectionError("down")):
            with pytest.raises(JevEvaluationError, match="unavailable"):
                classifier._evaluate({}, {})

    def test_success_is_recorded(self):
        classifier = self._classifier()

        class Response:
            status_code = 200

            def json(self):
                return {"answers": {"q": {"noul": 0.9}}}

        with patch("ytdl_core.kev.requests.Session.post", return_value=Response()):
            assert classifier._evaluate({}, {}) == {"q": {"noul": 0.9}}
        assert classifier.gate.successes == 1

    def test_auth_error_is_not_retried(self):
        classifier = self._classifier()

        class Response:
            status_code = 401

            def json(self):
                return {}

        with patch("ytdl_core.kev.requests.Session.post", return_value=Response()) as post:
            with pytest.raises(JevEvaluationError, match="authentication"):
                classifier._evaluate({}, {})
        assert post.call_count == 1

    def test_server_errors_are_retried_then_reported(self):
        """Transient server faults *are* worth retrying, unlike a timeout."""
        classifier = self._classifier(max_attempts=2)

        class Response:
            status_code = 503

            def json(self):
                return {}

        slept: list[float] = []
        with (
            patch("ytdl_core.kev.requests.Session.post", return_value=Response()) as post,
            patch("ytdl_core.kev.time.sleep", side_effect=slept.append),
        ):
            with pytest.raises(JevEvaluationError, match="temporarily failed"):
                classifier._evaluate({}, {})
        assert post.call_count == 2
        assert len(slept) == 1


class TestProviderWiring:
    """Every option core.py passes must actually exist on the classifier.

    A keyword the provider does not accept is a TypeError at construction, which
    surfaces as a crash before the run starts -- after the model has already been
    downloaded and loaded.
    """

    def test_music_downloader_builds_a_kev_provider(self, config):
        from ytdl_core.core import MusicDownloader
        from ytdl_core.events import DownloaderEvents

        downloader = MusicDownloader(
            config=config,
            events=DownloaderEvents(),
            workers=1,
            delay=(0, 0),
            use_kev=True,
        )
        assert isinstance(downloader.decision_classifier, KevClassifier)
        assert downloader.decision_classifier.gate is not None

    def test_configured_limits_reach_the_provider(self):
        from ytdl_core.core import MusicDownloader
        from ytdl_core.events import DownloaderEvents

        config = Config()
        config.DECISION_TIMEOUT_SECONDS = 123
        config.DECISION_MAX_IN_FLIGHT = 3
        config.DECISION_FAILURE_THRESHOLD = 7
        config.DECISION_MAX_QUESTIONS = 19
        config.DECISION_HEADROOM = 2

        downloader = MusicDownloader(
            config=config, events=DownloaderEvents(), workers=1, delay=(0, 0), use_kev=True
        )
        classifier = downloader.decision_classifier
        assert classifier.timeout_seconds == 123
        assert classifier.gate.max_in_flight == 3
        assert classifier.gate.failure_threshold == 7
        assert classifier.max_questions == 19
        assert classifier.eval_headroom == 2

    def test_injected_provider_is_used_verbatim(self, config):
        from ytdl_core.core import MusicDownloader
        from ytdl_core.events import DownloaderEvents

        provided = KevClassifier()
        downloader = MusicDownloader(
            config=config,
            events=DownloaderEvents(),
            workers=1,
            delay=(0, 0),
            kev_classifier=provided,
        )
        assert downloader.decision_classifier is provided


class TestProbePayload:
    def test_probe_sends_real_questions(self):
        """An empty request is answered instantly and measures nothing.

        This is the bug that reported "0.0s -- within budget" while every real
        evaluation took minutes.
        """
        from ytdl_core.kev_server import KevServerManager

        manager = KevServerManager(Path(tempfile.mkdtemp()) / "kev")
        payload = manager._probe_payload()
        # Real question contract, not an empty dict the server can answer for free.
        assert len(payload["questions"]) > 0
        assert any("::" in key for key in payload["questions"])
        assert payload["state"]["target"]["artist"] == "Probe Artist"

    def test_probe_survives_a_contract_failure(self):
        from ytdl_core.kev_server import KevServerManager

        steps: list[str] = []
        manager = KevServerManager(Path(tempfile.mkdtemp()) / "kev", on_step=steps.append)

        class Ok:
            status_code = 200

        with patch.object(manager, "_probe_payload", side_effect=RuntimeError("contract broke")):
            with patch("ytdl_core.kev_server.requests.post", return_value=Ok()):
                manager._report_latency(budget_seconds=1.0)
        # A broken probe must never block startup or raise.
        assert any("latency" in step for step in steps)

    def test_a_rejected_probe_is_not_reported_as_healthy(self):
        from ytdl_core.kev_server import KevServerManager

        steps: list[str] = []
        manager = KevServerManager(Path(tempfile.mkdtemp()) / "kev", on_step=steps.append)

        class Rejected:
            status_code = 422

        monkey = patch("ytdl_core.kev_server.requests.post", return_value=Rejected())
        with monkey:
            manager._report_latency(budget_seconds=20.0)
        assert any("rejected" in step for step in steps)
        assert not any("within budget" in step for step in steps)


class TestCoreSurfacesGiveUpOnce:
    def test_notice_reaches_the_user_once(self):
        """A dead provider should cost one message, not one per song."""
        from ytdl_core.core import MusicDownloader
        from ytdl_core.events import DownloaderEvents

        warnings: list[str] = []

        class Recorder(DownloaderEvents):
            def on_warn(self, message):
                warnings.append(message)

        classifier = KevClassifier(max_attempts=1)
        downloader = MusicDownloader(
            config=Config(), events=Recorder(), workers=1, delay=(0, 0), kev_classifier=classifier
        )

        with patch("ytdl_core.kev.requests.Session.post", side_effect=TimeoutError("slow")):
            for _ in range(6):
                try:
                    classifier._evaluate({}, {})
                except JevEvaluationError:
                    notice = classifier.gate.give_up_notice()
                    if notice:
                        downloader.events.on_warn(notice)

        assert len([w for w in warnings if "heuristic" in w]) == 1