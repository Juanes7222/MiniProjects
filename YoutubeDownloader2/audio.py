"""
Audio download using yt-dlp with Rich progress integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yt_dlp
from rich.progress import Progress, TaskID


def download_audio(
    url: str,
    output_path: Path,
    fmt: str,
    quality: str,
    progress: Progress,
    task_id: TaskID,
    opts: dict,
) -> Path:
    """
    Download and convert audio from *url* via yt-dlp + ffmpeg.

    The file is saved as:
        {output_path.parent}/{output_path.stem}.{fmt}

    Args:
        url:         Direct or webpage URL.
        output_path: Desired destination (stem used; extension assigned by yt-dlp).
        fmt:         Target codec: "mp3", "m4a", or "opus".
        quality:     Bitrate string, e.g. "192".
        progress:    Rich Progress instance (updated via hook).
        task_id:     Task ID inside *progress*.
        opts:        Dict with optional keys: cookies_browser, proxy.

    Returns:
        Path to the converted audio file.

    Raises:
        FileNotFoundError: If no output file is discovered after download.
        yt_dlp.utils.DownloadError / ExtractorError: Propagated on yt-dlp failures.
        OSError: Propagated on I/O errors (including ENOSPC).
    """
    template = str(output_path.parent / output_path.stem) + ".%(ext)s"

    def _hook(d: dict) -> None:
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            if total:
                progress.update(task_id, completed=done, total=total)
        elif d["status"] == "finished":
            total = d.get("total_bytes", 100)
            progress.update(task_id, completed=total, total=total)

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": fmt,
                "preferredquality": quality,
            }
        ],
        "progress_hooks": [_hook],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    if opts.get("cookies_browser"):
        ydl_opts["cookiesfrombrowser"] = (opts["cookies_browser"],)
    if opts.get("proxy"):
        ydl_opts["proxy"] = opts["proxy"]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        ydl.download([url])

    for ext in (fmt, "mp3", "m4a", "opus", "ogg", "webm", "flac"):
        candidate = output_path.parent / f"{output_path.stem}.{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"yt-dlp completed but no output file found for stem: {output_path.stem}"
    )