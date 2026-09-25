"""Tests for ytdl_core.downloader module."""

from __future__ import annotations

import threading
from unittest.mock import patch

from ytdl_core.downloader import download_partial, execute_download


class TestExecuteDownload:
    def test_returns_error_on_download_error(self, tmp_path, spy_events, config):
        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            from yt_dlp.utils import DownloadError

            instance.extract_info.side_effect = DownloadError("network error")

            stop = threading.Event()
            file, err = execute_download(
                "http://example.com/video",
                tmp_path,
                "mp3",
                "192",
                "Artist",
                "Song",
                spy_events,
                config,
                stop,
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
                "http://example.com/video",
                tmp_path,
                "mp3",
                "192",
                "Artist",
                "Song",
                spy_events,
                config,
                stop,
            )
            assert file is not None
            assert file.exists()
            assert err == ""
            assert MockYDL.call_args.args[0]["outtmpl"].endswith("Song.%(ext)s")

    def test_migrates_legacy_double_extension(self, tmp_path, spy_events, config):
        legacy_file = tmp_path / "Artist" / "Song.mp3.mp3"
        legacy_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_bytes(b"converted audio")
        target_file = tmp_path / "Artist" / "Song.mp3"

        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = {"title": "Song"}
            instance.prepare_filename.return_value = str(target_file)

            file, err = execute_download(
                "http://example.com/video",
                tmp_path,
                "mp3",
                "192",
                "Artist",
                "Song",
                spy_events,
                config,
                threading.Event(),
            )

        assert err == ""
        assert file == target_file
        assert not legacy_file.exists()

    def test_respects_stop_event(self, tmp_path, spy_events, config):
        stop = threading.Event()
        stop.set()  # pre-set

        file, err = execute_download(
            "http://example.com/video",
            tmp_path,
            "mp3",
            "192",
            "Artist",
            "Song",
            spy_events,
            config,
            stop,
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
                    "http://example.com/video",
                    tmp_path,
                    "mp3",
                    "192",
                    "Artist",
                    "Song",
                    spy_events,
                    config,
                    stop,
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
                "http://example.com/video",
                tmp_path,
                "mp3",
                "192",
                "Artist",
                "Song",
                spy_events,
                config,
                stop,
                state=state,
                state_lock=lock,
            )
            assert file is None
            assert "Disk full" in err
            assert stop.is_set()
            assert any(c[0] == "on_disk_full" for c in spy_events.calls)

    def test_disk_full_persists_while_holding_state_lock(self, tmp_path, spy_events, config):
        class ObservableLock:
            def __init__(self):
                self.entered = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, *_args):
                self.entered = False

        observed = []
        lock = ObservableLock()
        with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.side_effect = OSError(28, "No space left on device")
            with patch(
                "ytdl_core.downloader.save_state",
                side_effect=lambda *_args: observed.append(lock.entered),
            ):
                execute_download(
                    "http://example.com/video",
                    tmp_path,
                    "mp3",
                    "192",
                    "Artist",
                    "Song",
                    spy_events,
                    config,
                    threading.Event(),
                    state={"downloads": {}},
                    state_lock=lock,
                )

        assert observed == [True]


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
        partial = tmp_path / "_partial_abc12345.mp3"
        partial.write_bytes(b"fake partial")

        class FixedUUID:
            hex = "abc12345"

        with patch("ytdl_core.downloader.uuid4", return_value=FixedUUID()):
            with patch("ytdl_core.downloader.yt_dlp.YoutubeDL") as MockYDL:
                instance = MockYDL.return_value.__enter__.return_value
                instance.extract_info.return_value = {"title": "Test"}
                instance.prepare_filename.return_value = str(tmp_path / "_partial_abc12345.webm")

                result = download_partial(
                    "http://example.com/video",
                    tmp_path,
                    spy_events,
                )

        options = MockYDL.call_args.args[0]
        assert result == partial
        assert options["download_ranges"]({}, MockYDL) == ({"start_time": 0.0, "end_time": 90.0},)
        assert options["force_keyframes_at_cuts"] is True
        assert options["postprocessor_args"] == {"ExtractAudio+ffmpeg": ["-t", "90"]}
