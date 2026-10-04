from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

from ytdl_core.config import Config
from ytdl_core.downloader import execute_download
from ytdl_core.events import DownloaderEvents
from ytdl_core.ytdlp_options import build_ytdlp_base_opts


def _postprocessor_keys(options: dict) -> list[str]:
    return [postprocessor["key"] for postprocessor in options["postprocessors"]]


def test_thumbnail_embedding_can_be_disabled(tmp_path: Path) -> None:
    options = build_ytdlp_base_opts(
        tmp_path,
        "mp3",
        "192",
        quiet=True,
        no_warnings=True,
        embed_thumbnail=False,
    )

    assert "writethumbnail" not in options
    assert "EmbedThumbnail" not in _postprocessor_keys(options)


def test_thumbnail_embedding_is_enabled_for_url_downloads(tmp_path: Path) -> None:
    options = build_ytdlp_base_opts(
        tmp_path,
        "mp3",
        "192",
        quiet=True,
        no_warnings=True,
        embed_thumbnail=True,
    )

    assert options["writethumbnail"] is True
    assert "EmbedThumbnail" in _postprocessor_keys(options)


# --- ffmpeg thread cap ------------------------------------------------------
#
# ffmpeg defaults to one thread per core. This transcode runs in a pool beside
# every other ffmpeg in the batch, so N songs meant N transcodes each taking all
# six cores -- which does not make any of them finish sooner and starves the
# silence checks beside them.


def test_ffmpeg_thread_cap_is_a_top_level_option(tmp_path: Path) -> None:
    """yt-dlp passes the postprocessor dict straight to the postprocessor class.

    ``postprocessor_args`` nested inside it raises
    ``FFmpegExtractAudioPP.__init__() got an unexpected keyword argument
    'postprocessor_args'`` and fails every download in the batch. That was found
    by running the real pipeline rather than by a test, which is why the shape is
    pinned here now.
    """
    options = build_ytdlp_base_opts(tmp_path, "mp3", "192", True, True, config=Config())
    extract = next(pp for pp in options["postprocessors"] if pp["key"] == "FFmpegExtractAudio")
    assert "postprocessor_args" not in extract
    assert options["postprocessor_args"]["ExtractAudio+ffmpeg"] == ["-threads", "1"]


def test_a_scan_carries_no_postprocessors_at_all(tmp_path: Path) -> None:
    scan = build_ytdlp_base_opts(
        tmp_path, "mp3", "192", True, True, config=Config(), for_scan=True
    )
    assert "postprocessors" not in scan
    assert "postprocessor_args" not in scan


# --- fragment concurrency ---------------------------------------------------


def test_fragments_are_fetched_concurrently(tmp_path: Path) -> None:
    config = Config()
    config.FRAGMENT_CONCURRENCY = 4
    options = build_ytdlp_base_opts(tmp_path, "mp3", "192", True, True, config=config)
    assert options["concurrent_fragment_downloads"] == 4


def test_a_scan_does_not_fetch_fragments_concurrently(tmp_path: Path) -> None:
    """A scan downloads nothing, so there are no fragments to fetch."""
    config = Config()
    config.FRAGMENT_CONCURRENCY = 4
    scan = build_ytdlp_base_opts(
        tmp_path, "mp3", "192", True, True, config=config, for_scan=True
    )
    assert "concurrent_fragment_downloads" not in scan


def test_the_downloader_honours_the_configured_fragment_concurrency(tmp_path: Path) -> None:
    """The builder falls back to a module-level Config when given none.

    ``execute_download`` used to omit ``config``, so every override -- fragment
    concurrency, aria2c, retry counts -- was silently ignored for the one call
    that actually moves bytes.
    """
    config = Config()
    config.FRAGMENT_CONCURRENCY = 7
    captured: dict = {}

    class FakeYDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            raise RuntimeError("stop here")

    with patch("ytdl_core.downloader.yt_dlp.YoutubeDL", FakeYDL):
        execute_download(
            "https://example.invalid/x",
            tmp_path,
            "mp3",
            "192",
            "A",
            "S",
            DownloaderEvents(),
            config,
            threading.Event(),
        )
    assert captured["concurrent_fragment_downloads"] == 7


# --- aria2c -----------------------------------------------------------------


def test_aria2c_is_opt_in(tmp_path: Path) -> None:
    options = build_ytdlp_base_opts(tmp_path, "mp3", "192", True, True, config=Config())
    assert "external_downloader" not in options


def test_aria2c_is_wired_when_present(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "ytdl_core.ytdlp_options.find_aria2c", lambda: str(tmp_path / "aria2c")
    )
    config = Config()
    config.USE_ARIA2C = True
    options = build_ytdlp_base_opts(tmp_path, "mp3", "192", True, True, config=config)
    assert options["external_downloader"]["default"].endswith("aria2c")
    assert "-x" in options["external_downloader_args"]["default"]


def test_a_missing_aria2c_is_reported_rather_than_silent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ytdl_core.ytdlp_options.find_aria2c", lambda: None)
    said: list[str] = []
    config = Config()
    config.USE_ARIA2C = True
    options = build_ytdlp_base_opts(
        tmp_path, "mp3", "192", True, True, config=config, on_step=said.append
    )
    assert "external_downloader" not in options
    assert any("aria2c" in message for message in said)


def test_a_scan_never_hands_transfers_to_aria2c(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ytdl_core.ytdlp_options.find_aria2c", lambda: "aria2c")
    config = Config()
    config.USE_ARIA2C = True
    options = build_ytdlp_base_opts(
        tmp_path, "mp3", "192", True, True, config=config, for_scan=True
    )
    assert "external_downloader" not in options