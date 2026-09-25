"""Tests for ytdl_core.verifier module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from ytdl_core.fingerprint import AcoustIDCircuitBreaker
from ytdl_core.verifier import _verify_single, verify_library


class TestVerifySingle:
    def test_returns_skipped_when_stop_event_set(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        stop = threading.Event()
        stop.set()

        result = _verify_single(
            "Artist",
            "Song",
            tmp_path,
            "mp3",
            None,
            config,
            cb,
            sem,
            False,
            spy_events,
            stop,
        )
        assert result.status == "skipped"
        assert result.reason == "Interrupted"

    def test_returns_failed_when_file_missing(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        stop = threading.Event()

        result = _verify_single(
            "Artist",
            "Song",
            tmp_path,
            "mp3",
            None,
            config,
            cb,
            sem,
            False,
            spy_events,
            stop,
        )
        assert result.status == "failed"
        assert "does not exist" in result.reason

    def test_returns_failed_when_file_too_small(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        stop = threading.Event()

        # Create tiny file
        small = tmp_path / "Artist" / "Song.mp3"
        small.parent.mkdir(parents=True, exist_ok=True)
        small.write_bytes(b"x" * 100)  # 100 bytes < 50KB

        result = _verify_single(
            "Artist",
            "Song",
            tmp_path,
            "mp3",
            None,
            config,
            cb,
            sem,
            False,
            spy_events,
            stop,
        )
        assert result.status == "failed"
        assert "small" in result.reason.lower()

    def test_verifies_valid_file_without_fingerprint(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        stop = threading.Event()

        # Create a file large enough to pass the size check
        dest = tmp_path / "Artist" / "Song.mp3"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 60000)  # 60KB > 50KB threshold

        with patch("ytdl_core.verifier.verify_duration", return_value=(True, 180)):
            result = _verify_single(
                "Artist",
                "Song",
                tmp_path,
                "mp3",
                None,
                config,
                cb,
                sem,
                False,
                spy_events,
                stop,
            )
        assert result.status == "verified"
        assert result.file_path is not None
        assert result.duration_seconds == 180
        assert result.file_size_bytes == 60000

    def test_strict_mode_fails_without_fingerprint_match(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        stop = threading.Event()

        dest = tmp_path / "Artist" / "Song.mp3"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 60000)

        with patch("ytdl_core.verifier.verify_duration", return_value=(True, 180)):
            with patch("ytdl_core.verifier.verify_fingerprint", return_value=(False, 0.0, "")):
                result = _verify_single(
                    "Artist",
                    "Song",
                    tmp_path,
                    "mp3",
                    "KEY",
                    config,
                    cb,
                    sem,
                    False,
                    spy_events,
                    stop,
                    require_fingerprint=True,
                )
        assert result.status == "failed"
        assert "Fingerprint did not confirm" in result.reason


class TestVerifyLibrary:
    def test_returns_empty_for_empty_songs(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()

        results = verify_library(
            {},
            tmp_path,
            "mp3",
            1,
            None,
            config,
            cb,
            sem,
            False,
            spy_events,
            MagicMock(),
            {"downloads": {}},
            lock,
        )
        assert results == []
        assert any(c[0] == "on_session_start" for c in spy_events.calls)

    def test_skips_valid_cached_verification(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()
        file_path = tmp_path / "Artist" / "Song.mp3"
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(b"\x00" * 60000)
        md5 = "abc123"

        state = {
            "downloads": {
                "Artist::Song": {
                    "status": "verified",
                    "file_path": str(file_path),
                    "md5": md5,
                    "fingerprint_verified": False,
                }
            }
        }

        with patch("ytdl_core.verifier.verify_duration", return_value=(True, 180)):
            with patch("ytdl_core.verifier.compute_md5", return_value=md5):
                results = verify_library(
                    {"Artist": ["Song"]},
                    tmp_path,
                    "mp3",
                    1,
                    None,
                    config,
                    cb,
                    sem,
                    False,
                    spy_events,
                    MagicMock(),
                    state,
                    lock,
                )

        assert len(results) == 1
        assert results[0].status == "verified"
        assert results[0].md5 == md5

    def test_rechecks_missing_cached_verification(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()
        state = {
            "downloads": {
                "Artist::Song": {
                    "status": "verified",
                    "file_path": str(tmp_path / "Artist" / "Song.mp3"),
                    "fingerprint_verified": False,
                }
            }
        }

        results = verify_library(
            {"Artist": ["Song"]},
            tmp_path,
            "mp3",
            1,
            None,
            config,
            cb,
            sem,
            False,
            spy_events,
            MagicMock(),
            state,
            lock,
        )

        assert len(results) == 1
        assert results[0].status == "failed"
        assert "does not exist" in results[0].reason

    def test_rechecks_cached_verification_without_md5(self, tmp_path, spy_events, config):
        file_path = tmp_path / "Artist" / "Song.mp3"
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(b"\x00" * 60000)
        state = {
            "downloads": {
                "Artist::Song": {
                    "status": "verified",
                    "file_path": str(file_path),
                }
            }
        }

        with patch("ytdl_core.verifier.verify_duration", return_value=(True, 180)):
            results = verify_library(
                {"Artist": ["Song"]},
                tmp_path,
                "mp3",
                1,
                None,
                config,
                AcoustIDCircuitBreaker(),
                threading.Semaphore(2),
                False,
                spy_events,
                MagicMock(),
                state,
                threading.Lock(),
            )

        assert results[0].status == "verified"
        assert any(call[0] == "on_result" for call in spy_events.calls)

    def test_processes_downloaded_songs(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()

        # Create a file large enough to pass the size check
        dest = tmp_path / "Artist" / "Song.mp3"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 60000)  # 60KB > 50KB threshold

        state = {"downloads": {"Artist::Song": {"status": "downloaded"}}}
        persist = MagicMock()

        with patch("ytdl_core.verifier.verify_duration", return_value=(True, 180)):
            results = verify_library(
                {"Artist": ["Song"]},
                tmp_path,
                "mp3",
                1,
                None,
                config,
                cb,
                sem,
                False,
                spy_events,
                persist,
                state,
                lock,
            )
        assert len(results) == 1
        assert results[0].status == "verified"
        persist.assert_called()

    def test_legacy_persist_callback_signature_is_supported(self, tmp_path, spy_events, config):
        file_path = tmp_path / "Artist" / "Song.mp3"
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(b"\x00" * 60000)
        calls = []

        def persist(
            state,
            lock,
            key,
            status,
            url,
            stored_path,
            md5,
            output_dir,
            fingerprint_verified=False,
            fingerprint_confidence=0.0,
            fingerprint_label=None,
            preserve_timestamp=False,
        ):
            calls.append(key)

        with patch("ytdl_core.verifier.verify_duration", return_value=(True, 180)):
            verify_library(
                {"Artist": ["Song"]},
                tmp_path,
                "mp3",
                1,
                None,
                config,
                AcoustIDCircuitBreaker(),
                threading.Semaphore(2),
                False,
                spy_events,
                persist,
                {"downloads": {"Artist::Song": {"status": "downloaded"}}},
                threading.Lock(),
            )

        assert calls == ["Artist::Song"]

    def test_persists_failed_state(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()

        state = {"downloads": {"Artist::Song": {"status": "downloaded"}}}
        persist = MagicMock()

        # No file on disk → should fail
        results = verify_library(
            {"Artist": ["Song"]},
            tmp_path,
            "mp3",
            1,
            None,
            config,
            cb,
            sem,
            False,
            spy_events,
            persist,
            state,
            lock,
        )
        assert len(results) == 1
        assert results[0].status == "failed"
        assert "does not exist" in results[0].reason
