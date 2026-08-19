"""Tests for ytdl_core.verifier module."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ytdl_core.config import Config
from ytdl_core.fingerprint import AcoustIDCircuitBreaker
from ytdl_core.result import DownloadResult
from ytdl_core.verifier import _verify_single, verify_library


class TestVerifySingle:
    def test_returns_skipped_when_stop_event_set(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        stop = threading.Event()
        stop.set()

        result = _verify_single(
            "Artist", "Song", tmp_path, "mp3", None, config, cb, sem,
            False, spy_events, stop,
        )
        assert result.status == "skipped"
        assert result.reason == "Interrupted"

    def test_returns_failed_when_file_missing(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        stop = threading.Event()

        result = _verify_single(
            "Artist", "Song", tmp_path, "mp3", None, config, cb, sem,
            False, spy_events, stop,
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
            "Artist", "Song", tmp_path, "mp3", None, config, cb, sem,
            False, spy_events, stop,
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
                "Artist", "Song", tmp_path, "mp3", None, config, cb, sem,
                False, spy_events, stop,
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
            with patch(
                "ytdl_core.verifier.verify_fingerprint", return_value=(False, 0.0, "")
            ):
                result = _verify_single(
                    "Artist", "Song", tmp_path, "mp3", "KEY", config, cb, sem,
                    False, spy_events, stop, require_fingerprint=True,
                )
        assert result.status == "failed"
        assert "Fingerprint did not confirm" in result.reason


class TestVerifyLibrary:
    def test_returns_empty_for_empty_songs(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()

        results = verify_library(
            {}, tmp_path, "mp3", 1, None, config, cb, sem,
            False, spy_events, MagicMock(), {"downloads": {}}, lock,
        )
        assert results == []
        assert any(c[0] == "on_session_start" for c in spy_events.calls)

    def test_skips_already_verified_songs(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()

        state = {
            "downloads": {
                "Artist::Song": {
                    "status": "verified",
                    "file_path": "/some/path.mp3",
                    "md5": "abc123",
                    "fingerprint_verified": False,
                }
            }
        }

        results = verify_library(
            {"Artist": ["Song"]}, tmp_path, "mp3", 1, None, config, cb, sem,
            False, spy_events, MagicMock(), state, lock,
        )
        assert len(results) == 1
        assert results[0].status == "verified"
        assert results[0].md5 == "abc123"

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
                {"Artist": ["Song"]}, tmp_path, "mp3", 1, None, config, cb, sem,
                False, spy_events, persist, state, lock,
            )
        assert len(results) == 1
        assert results[0].status == "verified"
        persist.assert_called()

    def test_persists_failed_state(self, tmp_path, spy_events, config):
        cb = AcoustIDCircuitBreaker()
        sem = threading.Semaphore(2)
        lock = threading.Lock()

        state = {"downloads": {"Artist::Song": {"status": "downloaded"}}}
        persist = MagicMock()

        # No file on disk → should fail
        results = verify_library(
            {"Artist": ["Song"]}, tmp_path, "mp3", 1, None, config, cb, sem,
            False, spy_events, persist, state, lock,
        )
        assert len(results) == 1
        assert results[0].status == "failed"
        assert "does not exist" in results[0].reason
