"""
yt-dlp option building and downloaded-file resolution.

Standalone functions extracted from MusicDownloader so they can be
tested and reused independently of the main class.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional


def normalize_browser_cookies(value: Any) -> tuple[Any, ...]:
    """Convert a browser cookie value to a tuple for yt-dlp's cookiesfrombrowser."""
    if isinstance(value, tuple):
        return value
    return (value,)


def build_ytdlp_base_opts(
    output_dir: Path,
    fmt: str,
    quality: str,
    quiet: bool,
    no_warnings: bool,
    progress_hook: Any = None,
    skip_existing: bool = False,
    max_downloads: Optional[int] = None,
    cookies_browser: Optional[Any] = None,
    cookies_file: Optional[Path] = None,
    proxy: Optional[str] = None,
    download_archive: Optional[Path] = None,
    enable_remote_components: bool = True,
    youtube_player_clients: Optional[list[str]] = None,
    noplaylist: bool = False,
    for_scan: bool = False,
) -> dict[str, Any]:
    """
    Build the base yt-dlp options dictionary used by both scan and download passes.

    Parameters
    ----------
    output_dir:
        Target directory for downloaded files.
    fmt:
        Output format — ``"mp4"`` for video, anything else for audio.
    quality:
        Maximum height (video) or bitrate label (audio).
    quiet / no_warnings:
        yt-dlp verbosity flags.
    progress_hook:
        Optional single progress hook callable.
    skip_existing:
        If True, set ``nooverwrites``.
    max_downloads:
        Limit the number of items from a playlist.
    cookies_browser / cookies_file / proxy:
        Authentication and network options.
    download_archive:
        Optional archive file to skip previously-downloaded items.
    enable_remote_components:
        Whether to enable ``remote_components`` (EJS challenge solving).
    youtube_player_clients:
        Optional list of player client names for the extractor args.
    noplaylist:
        If True, download only the first item.
    for_scan:
        If True, enable ``ignoreerrors`` and skip thumbnail/postprocessor setup.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    is_video = fmt == "mp4"
    ydl_opts: dict[str, Any] = {
        "format": (
            f"bestvideo[height<={quality}]+bestaudio/bestvideo[height<={quality}]/best"
            if is_video
            else "bestaudio/best"
        ),
        "outtmpl": str(
            output_dir
            / "%(uploader)s"
            / ("%(title)s [%(id)s].mp4" if is_video else "%(title)s [%(id)s].%(ext)s")
        ),
        "quiet": quiet,
        "no_warnings": no_warnings,
        "noprogress": True,
        "progress_hooks": [progress_hook] if progress_hook else [],
        "extract_flat": False,
        "ignoreerrors": for_scan,
        "skip_unavailable_fragments": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 10,
        "file_access_retries": 5,
        "windowsfilenames": True,
        "noplaylist": noplaylist,
    }

    if not for_scan:
        ydl_opts["writethumbnail"] = True
        if is_video:
            ydl_opts["merge_output_format"] = "mp4"
            ydl_opts["postprocessors"] = [
                {"key": "FFmpegMetadata"},
                {"key": "EmbedThumbnail", "already_have_thumbnail": False},
            ]
        else:
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": fmt,
                    "preferredquality": quality,
                },
                {"key": "FFmpegMetadata"},
                {"key": "EmbedThumbnail", "already_have_thumbnail": False},
            ]

    if skip_existing:
        ydl_opts["nooverwrites"] = True

    if max_downloads is not None:
        ydl_opts["playlist_items"] = f"1-{max_downloads}"

    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = normalize_browser_cookies(cookies_browser)

    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)

    if proxy:
        ydl_opts["proxy"] = proxy

    if download_archive:
        ydl_opts["download_archive"] = str(download_archive)

    if youtube_player_clients:
        ydl_opts["extractor_args"] = {
            "youtube": {
                "player_client": list(youtube_player_clients),
            }
        }

    node_path = shutil.which("node")
    if node_path:
        ydl_opts["js_runtimes"] = {"node": {"path": node_path}}

    if enable_remote_components:
        ydl_opts["remote_components"] = ["ejs:github"]

    return ydl_opts


def resolve_downloaded_file(base_file: Path, fmt: str) -> Optional[Path]:
    """
    Try to find the final converted file produced by yt-dlp / ffmpeg.

    yt-dlp sometimes writes with a different extension than requested,
    so we check common audio/video extensions and fall back to a glob.
    """
    exact = base_file.with_suffix(f".{fmt}")
    if exact.exists():
        return exact

    parent = base_file.parent
    stem = base_file.stem

    candidates: list[Path] = []
    for ext in (fmt, "mp3", "m4a", "opus", "mp4", "ogg", "webm", "flac", "aac", "wav"):
        candidate = parent / f"{stem}.{ext}"
        if candidate.exists():
            candidates.append(candidate)

    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    globbed = sorted(
        parent.glob(f"{stem}.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return globbed[0] if globbed else None
