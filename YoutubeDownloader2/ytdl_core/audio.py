"""
Robust audio download helper using yt-dlp + FFmpeg + Rich progress.

Features:
- Auto-detects Node / Deno / QuickJS / Bun runtimes when available.
- Enables yt-dlp remote components (default: ejs:github).
- Uses a temporary staging directory to avoid collisions and partial-file confusion.
- Supports browser cookies, cookie files, proxies, archive files, headers, and extractor args.
- Adds sane retries for YouTube stability.
- Optionally uses aria2c if explicitly enabled and installed.
- Keeps Rich progress updated via yt-dlp progress hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4
import shutil
import tempfile

import yt_dlp
from rich.progress import Progress, TaskID


ALLOWED_AUDIO_FORMATS = {
    "mp3",
    "m4a",
    "opus",
    "ogg",
    "flac",
    "wav",
    "aac",
}


def _normalize_browser_cookie_value(value: Any) -> tuple[Any, ...]:
    """
    Accepts:
      - "chrome"
      - ("chrome", "default")
      - ("firefox", "default", None, "Meta")
    and returns a tuple suitable for yt-dlp's cookiesfrombrowser option.
    """
    if isinstance(value, tuple):
        return value
    return (value,)


def _build_final_target(output_path: Path, fmt: str) -> Path:
    # The function keeps the stem and swaps the extension to the requested format.
    return output_path.parent / f"{output_path.stem}.{fmt}"


def download_audio(
    url: str,
    output_path: Path,
    fmt: str = "mp3",
    quality: str = "192",
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    opts: dict[str, Any] | None = None,
) -> Path:
    """
    Download and convert audio from *url* via yt-dlp + ffmpeg.

    Parameters
    ----------
    url:
        Direct URL or webpage URL.
    output_path:
        Desired destination base path. The final file will be saved as:
        {output_path.parent}/{output_path.stem}.{fmt}
    fmt:
        Target audio format. Recommended: mp3, m4a, opus, ogg, flac, wav, aac.
    quality:
        Bitrate/quality value used by FFmpegExtractAudio (for example "192").
    progress:
        Optional Rich Progress instance.
    task_id:
        Optional task id inside the Rich Progress instance.
    opts:
        Optional configuration dictionary. Supported keys:
            - cookies_browser: str | tuple
            - cookies_file: str | Path
            - proxy: str
            - headers: dict[str, str]
            - extractor_args: dict[str, dict[str, list[str]]]
            - archive_path: str | Path
            - ffmpeg_location: str | Path
            - js_runtime: "node" | "deno" | "quickjs" | "bun"
            - remote_components: list[str] | set[str] | tuple[str, ...]
            - enable_remote_components: bool
            - use_aria2c: bool
            - retries: int
            - fragment_retries: int
            - extractor_retries: int
            - file_access_retries: int
            - socket_timeout: int
            - concurrent_fragment_downloads: int
            - noplaylist: bool

    Returns
    -------
    Path
        Final path to the converted audio file.

    Raises
    ------
    ValueError
        If the requested format is not supported.
    FileNotFoundError
        If ffmpeg is missing, or if the final converted file cannot be found.
    yt_dlp.utils.DownloadError / yt_dlp.utils.ExtractorError
        Propagated from yt-dlp.
    OSError
        Propagated from filesystem or process errors.
    """
    opts = dict(opts or {})

    fmt = fmt.lower().strip()
    if fmt not in ALLOWED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{fmt}'. "
            f"Supported formats: {', '.join(sorted(ALLOWED_AUDIO_FORMATS))}"
        )

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_target = _build_final_target(output_path, fmt)

    # ffmpeg is required for FFmpegExtractAudio
    ffmpeg_bin = opts.get("ffmpeg_location") or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError(
            "ffmpeg was not found in PATH. Install ffmpeg and make sure it is available."
        )

    # yt-dlp accepts either the binary path or the containing directory.
    ffmpeg_location = Path(ffmpeg_bin)
    if ffmpeg_location.is_file():
        ffmpeg_location = ffmpeg_location.parent

    # Build JS runtimes config.
    # yt-dlp supports Node 22+ as a runtime, and also supports Deno / QuickJS / Bun.
    # We auto-detect the requested runtime when possible.
    js_runtime = str(opts.get("js_runtime", "node")).lower().strip()
    js_runtimes: dict[str, dict[str, str]] = {}

    def _add_runtime(name: str) -> None:
        runtime_path = shutil.which(name)
        if runtime_path:
            js_runtimes[name] = {"path": runtime_path}

    if js_runtime == "node":
        _add_runtime("node")
    elif js_runtime == "deno":
        _add_runtime("deno")
    elif js_runtime == "quickjs":
        _add_runtime("quickjs")
    elif js_runtime == "bun":
        _add_runtime("bun")

    # A temporary staging folder keeps this isolated from existing files
    # and makes the final file discovery deterministic.
    with tempfile.TemporaryDirectory(
        prefix="yt-dlp-audio-",
        dir=str(output_path.parent),
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        job_id = uuid4().hex[:10]
        stage_stem = f"{output_path.stem}-{job_id}"
        stage_outtmpl = str(tmpdir_path / f"{stage_stem}.%(ext)s")

        def _hook(d: dict[str, Any]) -> None:
            if progress is None or task_id is None:
                return

            status = d.get("status")
            if status == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                done = d.get("downloaded_bytes", 0)
                if total > 0:
                    progress.update(task_id, completed=done, total=total)
                else:
                    # Keep the task alive even when the total size is unknown.
                    progress.update(task_id, completed=done)
            elif status == "finished":
                total = d.get("total_bytes") or d.get("downloaded_bytes") or 0
                if total > 0:
                    progress.update(task_id, completed=total, total=total)

        ydl_opts: dict[str, Any] = {
            "format": opts.get("format", "bestaudio/best"),
            "outtmpl": stage_outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": fmt,
                    "preferredquality": quality,
                }
            ],
            "progress_hooks": [_hook] if progress is not None and task_id is not None else [],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": bool(opts.get("noplaylist", True)),
            "retries": int(opts.get("retries", 10)),
            "fragment_retries": int(opts.get("fragment_retries", 10)),
            "extractor_retries": int(opts.get("extractor_retries", 10)),
            "file_access_retries": int(opts.get("file_access_retries", 5)),
            "socket_timeout": int(opts.get("socket_timeout", 30)),
            "concurrent_fragment_downloads": int(
                opts.get("concurrent_fragment_downloads", 4)
            ),
            "ffmpeg_location": str(ffmpeg_location),
            "final_ext": fmt,
        }

        # Optional: browser cookies
        cookies_browser = opts.get("cookies_browser")
        if cookies_browser:
            ydl_opts["cookiesfrombrowser"] = _normalize_browser_cookie_value(cookies_browser)

        # Optional: cookies file
        cookies_file = opts.get("cookies_file")
        if cookies_file:
            ydl_opts["cookiefile"] = str(cookies_file)

        # Optional: proxy
        proxy = opts.get("proxy")
        if proxy:
            ydl_opts["proxy"] = proxy

        # Optional: custom headers
        headers = opts.get("headers")
        if headers:
            ydl_opts["http_headers"] = dict(headers)

        # Optional: extractor args. Values must be lists of strings.
        extractor_args = opts.get("extractor_args")
        if extractor_args:
            ydl_opts["extractor_args"] = extractor_args

        # Optional: archive file to prevent re-downloading the same item.
        archive_path = opts.get("archive_path")
        if archive_path:
            ydl_opts["download_archive"] = str(archive_path)

        # Optional: use a specific JS runtime path if the caller wants to force it.
        # If no path is available, yt-dlp will still work with its default runtime setup.
        if js_runtimes:
            ydl_opts["js_runtimes"] = js_runtimes

        # Remote components for EJS challenge solving.
        # Recommended default: ejs:github.
        enable_remote_components = bool(opts.get("enable_remote_components", True))
        if enable_remote_components:
            remote_components = opts.get("remote_components", ["ejs:github"])
            if isinstance(remote_components, (list, tuple, set)):
                ydl_opts["remote_components"] = list(remote_components)
            else:
                ydl_opts["remote_components"] = [str(remote_components)]

        # Optional aria2c acceleration (opt-in only).
        use_aria2c = bool(opts.get("use_aria2c", False))
        if use_aria2c and shutil.which("aria2c"):
            ydl_opts["external_downloader"] = "aria2c"
            ydl_opts["external_downloader_args"] = ["-x", "8", "-s", "8", "-k", "1M"]

        # Let yt-dlp do the actual work.
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # The converted file should exist with the requested extension.
        expected_file = tmpdir_path / f"{stage_stem}.{fmt}"
        if not expected_file.exists():
            # Defensive fallback: search for the newest matching file.
            candidates = sorted(
                tmpdir_path.glob(f"{stage_stem}.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"yt-dlp completed, but no output file was found for stem '{stage_stem}'."
                )

            # Prefer the exact requested format if it exists among candidates.
            exact = next((p for p in candidates if p.suffix.lower() == f".{fmt}"), None)
            if exact is None:
                raise FileNotFoundError(
                    f"yt-dlp completed, but the converted '{fmt}' file was not produced."
                )
            expected_file = exact

        if final_target.exists():
            final_target.unlink()

        shutil.move(str(expected_file), str(final_target))
        return final_target