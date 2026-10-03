"""
yt-dlp option building and downloaded-file resolution.

Standalone functions extracted from MusicDownloader so they can be
tested and reused independently of the main class.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config
from .events import DownloaderEvents

DEFAULT_CONFIG = Config()


def normalize_browser_cookies(value: Any) -> tuple[Any, ...]:
    """Convert a browser cookie value to a tuple for yt-dlp's cookiesfrombrowser."""
    if isinstance(value, tuple):
        return value
    return (value,)


def _retry_counts(for_scan: bool, config: Optional[Config] = None) -> dict[str, int]:
    """How many internal retries yt-dlp should make on its own.

    Two retry layers used to stack: yt-dlp's internal ten, wrapped by the
    application's three, for as many as thirty attempts against a single URL --
    with yt-dlp sleeping between each. The application loop can see the error
    text and knows whether a candidate is even worth retrying (a dead URL is
    usually a dead *candidate*), so it owns retries now and yt-dlp is left with
    enough attempts to ride out a transient blip.

    A scan pass downloads nothing, so it gets the shorter budget: there is no
    partial file to protect and no reason to keep hammering.
    """
    cfg = config or DEFAULT_CONFIG
    attempts = cfg.YTDLP_SCAN_RETRIES if for_scan else cfg.YTDLP_DOWNLOAD_RETRIES
    return {
        "retries": attempts,
        "fragment_retries": attempts,
        "extractor_retries": attempts,
        "file_access_retries": min(attempts, 3),
    }


def apply_request_shaping(
    ydl_opts: dict[str, Any],
    config: Optional[Config] = None,
    *,
    for_scan: bool = False,
) -> dict[str, Any]:
    """Apply shared retry and politeness settings to a yt-dlp options dict.

    ``sleep_interval_requests`` lets yt-dlp pace itself between extraction
    requests, with jitter, and only when it needs to. That is strictly better
    than the unconditional per-song sleep it replaces: that sleep fired even
    when the budget was not exhausted, and it paced songs rather than requests,
    so one song firing five extractions multiplied the load fivefold.
    """
    cfg = config or DEFAULT_CONFIG
    ydl_opts.update(_retry_counts(for_scan, cfg))
    ydl_opts["socket_timeout"] = cfg.SOCKET_TIMEOUT
    if cfg.YTDLP_RETRY_SLEEP:
        # Bounded and exponential: an unbounded internal sleep is how a single
        # bad URL ends up holding a worker for minutes.
        ydl_opts["retry_sleep"] = cfg.YTDLP_RETRY_SLEEP
    if cfg.SLEEP_INTERVAL_REQUESTS > 0:
        ydl_opts["sleep_interval_requests"] = cfg.SLEEP_INTERVAL_REQUESTS
    if cfg.MAX_SLEEP_INTERVAL > 0:
        ydl_opts["max_sleep_interval"] = cfg.MAX_SLEEP_INTERVAL
    return ydl_opts


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
    cookies_file: str | Path | None = None,
    proxy: Optional[str] = None,
    download_archive: Optional[Path] = None,
    enable_remote_components: bool = True,
    youtube_player_clients: Optional[list[str]] = None,
    noplaylist: bool = False,
    for_scan: bool = False,
    output_template: str | Path | None = None,
    embed_thumbnail: bool = True,
    config: Config | None = None,
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
        "outtmpl": str(output_template)
        if output_template
        else str(
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
        "windowsfilenames": True,
        "noplaylist": noplaylist,
    }
    apply_request_shaping(ydl_opts, config, for_scan=for_scan)

    if not for_scan:
        if embed_thumbnail:
            ydl_opts["writethumbnail"] = True
        if is_video:
            ydl_opts["merge_output_format"] = "mp4"
            ydl_opts["postprocessors"] = [
                {"key": "FFmpegMetadata"},
            ]
        else:
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": fmt,
                    "preferredquality": quality,
                },
                {"key": "FFmpegMetadata"},
            ]
        if embed_thumbnail:
            ydl_opts["postprocessors"].append(
                {"key": "EmbedThumbnail", "already_have_thumbnail": False}
            )

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


def make_progress_hook(
    events: DownloaderEvents,
    artist: str,
    song: str,
) -> Callable[[dict], None]:
    """
    Create a yt-dlp progress hook that forwards events to *events*.

    This eliminates the four nearly-identical ``_progress_hook`` / ``_hook``
    closures that were scattered across ``_process_song``, ``download_url``,
    and ``_download_partial``.
    """

    def _hook(d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded_b = d.get("downloaded_bytes") or 0
            pct = (downloaded_b / total_b * 100.0) if total_b else 0.0
            events.on_download_progress(
                artist, song[:30], pct, d.get("speed") or 0.0, downloaded_b, total_b
            )
        elif status == "finished":
            total_b = d.get("total_bytes") or d.get("downloaded_bytes") or 0
            events.on_download_progress(artist, song[:30], 100.0, 0.0, total_b, total_b)
        elif status == "processing":
            events.on_info(f"[cyan]Processing: {song}[/cyan]")

    return _hook


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
