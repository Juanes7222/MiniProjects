"""
yt-dlp download execution with retry logic and partial downloads.

Provides the low-level ``execute_download`` function that wraps yt-dlp's
``extract_info`` with configurable retries, disk-full detection, and
progress hooks.  Also contains ``download_partial`` for the short clips
used by fingerprint verification.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

from .config import Config
from .events import DownloaderEvents
from .state import save_state
from .utils import sanitize_filename
from .ytdlp_options import build_ytdlp_base_opts, make_progress_hook, resolve_downloaded_file


def execute_download(
    url: str,
    output_dir: Path,
    fmt: str,
    quality: str,
    artist: str,
    song: str,
    events: DownloaderEvents,
    config: Config,
    stop_event: threading.Event,
    state: Optional[dict] = None,
    state_lock: Optional[threading.Lock] = None,
    cookies_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    proxy: Optional[str] = None,
    needs_fp: bool = False,
) -> tuple[Optional[Path], str]:
    """
    Download a single URL via yt-dlp with retry and post-download resolution.

    Returns
    -------
    (downloaded_file_or_None, last_error_string)

    On disk-full (errno 28) the caller should persist state and return
    early — this function sets *stop_event* but does NOT persist state
    itself (to avoid double-persists).
    """
    safe_artist = sanitize_filename(artist)
    safe_song = sanitize_filename(song)
    is_video = fmt == "mp4"
    output_template = str(
        output_dir / safe_artist / f"{safe_song}.mp4"
        if is_video
        else f"{safe_song}.%(ext)s"
    )

    progress_hook = make_progress_hook(events, artist, song)

    if is_video:
        ydl_opts: Any = {
            "format": f"bestvideo[height<={quality}]+bestaudio/bestvideo[height<={quality}]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [progress_hook],
            "postprocessors": [
                {"key": "FFmpegMetadata"},
                {"key": "EmbedThumbnail", "already_have_thumbnail": False},
            ],
            "noplaylist": True,
            "writethumbnail": True,
            "extractor_args": {"youtube": {"player_client": ["web"]}},
        }
    else:
        ydl_opts: Any = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [progress_hook],
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": fmt, "preferredquality": quality},
                {"key": "FFmpegMetadata"},
                {"key": "EmbedThumbnail", "already_have_thumbnail": False},
            ],
            "noplaylist": True,
            "writethumbnail": True,
            "extractor_args": {"youtube": {"player_client": ["web"]}},
        }

    node_path = shutil.which("node")
    if node_path:
        ydl_opts["js_runtimes"] = {"node": {"path": node_path}}

    ydl_opts["remote_components"] = ["ejs:github"]

    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)
    if proxy:
        ydl_opts["proxy"] = proxy

    downloaded_file: Optional[Path] = None
    last_error = ""

    for attempt in range(1, config.RETRY_ATTEMPTS + 1):
        if stop_event.is_set():
            break
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                if info_dict:
                    source_file = Path(ydl.prepare_filename(info_dict))
                    downloaded_file = resolve_downloaded_file(source_file, fmt)

            if downloaded_file and downloaded_file.exists():
                break

        except DownloadError as exc:
            last_error = f"DownloadError: {exc}"
        except ExtractorError as exc:
            last_error = f"ExtractorError: {exc}"
        except OSError as exc:
            if exc.errno == 28:
                events.on_disk_full()
                stop_event.set()
                if state is not None and state_lock is not None:
                    save_state(state, output_dir)
                last_error = "Disk full"
                return None, last_error
            last_error = f"OSError: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < config.RETRY_ATTEMPTS:
            wait = config.RETRY_BACKOFF_BASE**attempt
            events.on_download_retry(artist, song, attempt, config.RETRY_ATTEMPTS, last_error, wait)
            time.sleep(wait)

    if downloaded_file is None or not downloaded_file.exists():
        return None, last_error

    return downloaded_file, ""


def download_partial(
    url: str,
    output_dir: Path,
    events: DownloaderEvents,
    cookies_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Optional[Path]:
    """
    Download a short partial clip for fingerprint verification.

    Returns the path to the partial MP3, or ``None`` on failure.
    """
    token = uuid4().hex[:8]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_base = output_dir / f"_partial_{token}"
    expected_mp3 = partial_base.with_suffix(".mp3")

    ydl_opts: Any = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "outtmpl": str(partial_base) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "ignoreerrors": False,
        "socket_timeout": 30,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 5,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "extractor_args": {"youtube": {"player_client": ["web"]}},
        "remote_components": ["ejs:github"],
    }

    node_path = shutil.which("node")
    if node_path:
        ydl_opts["js_runtimes"] = {"node": {"path": node_path}}

    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)
    if proxy:
        ydl_opts["proxy"] = proxy

    hook = make_progress_hook(events, "Partial", url)
    ydl_opts["progress_hooks"] = [hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                return None

            source_file = Path(ydl.prepare_filename(info))
            final_mp3 = source_file.with_suffix(".mp3")

            if final_mp3.exists():
                return final_mp3

            if expected_mp3.exists():
                return expected_mp3

            for candidate in sorted(
                output_dir.glob(f"_partial_{token}.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                if candidate.exists():
                    return candidate

    except Exception:
        return None

    return None
