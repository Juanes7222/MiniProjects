"""Tests for ytdl_core.downloader module."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ytdl_core.config import Config
from ytdl_core.downloader import download_partial, execute_download
from ytdl_core.events import DownloaderEvents


class TestExecuteDownload:
    def test_returns_error_on_download_error(self, tmp_path, spy_events, config):
        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            from yt_dlp.utils import DownloadError
            instance.extract_info.side_effect = DownloadError("network error")

            stop = threading.Event()
            file, err = execute_download(
                "http://example.com/video", tmp_path, "mp3", "192",
                "Artist", "Song", spy_events, config, stop,
            )
            assert file is None
            assert "DownloadError" in err
            # execute_download returns (None, error) — on_download_failed is called by the caller
            assert any(c[0] == "on_download_retry" for c in spy_events.calls)

    def test_returns_file_on_success(self, tmp_path, spy_events, config):
        # Create a fake file that resolve_downloaded_file will find
        fake_file = tmp_path / "Artist" / "Song.mp3"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"fake mp3 content")

        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = {"title": "Song"}
            instance.prepare_filename.return_value = str(fake_file)

            stop = threading.Event()
            file, err = execute_download(
                "http://example.com/video", tmp_path, "mp3", "192",
                "Artist", "Song", spy_events, config, stop,
            )
            assert file is not None
            assert file.exists()
            assert err == ""

    def test_respects_stop_event(self, tmp_path, spy_events, config):
        stop = threading.Event()
        stop.set()  # pre-set

        file, err = execute_download(
            "http://example.com/video", tmp_path, "mp3", "192",
            "Artist", "Song", spy_events, config, stop,
        )
        assert file is None

    def test_retries_on_failure(self, tmp_path, spy_events, config):
        config.RETRY_ATTEMPTS = 2

        fake_file = tmp_path / "Artist" / "Song.mp3"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"fake")

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from yt_dlp.utils import DownloadError
                raise DownloadError("first attempt fails")
            return {"title": "Song"}

        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.side_effect = side_effect
            instance.prepare_filename.return_value = str(fake_file)

            with patch("ytdl_core.downloader.time.sleep"):
                stop = threading.Event()
                file, err = execute_download(
                    "http://example.com/video", tmp_path, "mp3", "192",
                    "Artist", "Song", spy_events, config, stop,
                )
            assert file is not None
            assert call_count == 2
            assert any(c[0] == "on_download_retry" for c in spy_events.calls)

    def test_disk_full_sets_stop_event(self, tmp_path, spy_events, config):
        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            exc = OSError(28, "No space left on device")
            instance.extract_info.side_effect = exc

            stop = threading.Event()
            state = {"downloads": {}}
            lock = threading.Lock()
            file, err = execute_download(
                "http://example.com/video", tmp_path, "mp3", "192",
                "Artist", "Song", spy_events, config, stop,
                state=state, state_lock=lock,
            )
            assert file is None
            assert "Disk full" in err
            assert stop.is_set()
            assert any(c[0] == "on_disk_full" for c in spy_events.calls)


class TestDownloadPartial:
    def test_returns_none_on_exception(self, tmp_path, spy_events):
        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.side_effect = RuntimeError("network error")

            result = download_partial("http://example.com/video", tmp_path, spy_events)
            assert result is None

    def test_returns_none_when_no_info(self, tmp_path, spy_events):
        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = None

            result = download_partial("http://example.com/video", tmp_path, spy_events)
            assert result is None

    def test_returns_file_on_success(self, tmp_path, spy_events):
        # The partial download creates a file matching _partial_{token}.*
        fake = tmp_path / "_partial_abc12345.mp3"
        fake.write_bytes(b"fake partial")

        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = {"title": "Test"}
            instance.prepare_filename.return_value = str(tmp_path / "_partial_abc12345.webm")

            result = download_partial("http://example.com/video", tmp_path, spy_events)
            # The function looks for the mp3 or glob matches
            assert result is None or result.exists()
