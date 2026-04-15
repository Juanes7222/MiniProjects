"""
YT Music Downloader v2.0
========================
A production-quality CLI tool that searches YouTube (and SoundCloud / Bandcamp
as fallbacks) and downloads audio tracks defined in a structured JSON file.

Usage
-----
python downloader.py --file songs.json [OPTIONS]
python downloader.py --data '{"Artist": ["Song"]}' [OPTIONS]

Dependencies
------------
pip install yt-dlp rich mutagen requests musicbrainzngs rapidfuzz pyacoustid pydub

See --help for a full option reference.
"""

from __future__ import annotations

import sys

_REQUIRED_PKGS = {
    "yt_dlp": "yt-dlp",
    "rich": "rich",
    "mutagen": "mutagen",
    "requests": "requests",
    "musicbrainzngs": "musicbrainzngs",
    "rapidfuzz": "rapidfuzz",
    "acoustid": "pyacoustid",
    "pydub": "pydub",
}
_MISSING = []
for _mod, _pkg in _REQUIRED_PKGS.items():
    try:
        __import__(_mod)
    except ImportError:
        _MISSING.append(_pkg)

if _MISSING:
    try:
        from rich.console import Console as _C
        from rich.panel import Panel as _P

        _C().print(
            _P(
                "[red]The following packages are not installed:[/red]\n"
                + "\n".join(f"  {p}" for p in _MISSING)
                + "\n\n[yellow]Fix with:[/yellow]\n"
                "  [bold]pip install yt-dlp rich mutagen requests musicbrainzngs "
                "rapidfuzz pyacoustid pydub[/bold]",
                title="[bold red] Missing Dependencies[/bold red]",
                border_style="red",
            )
        )
    except ImportError:
        print("Missing packages:", ", ".join(_MISSING))
        print(
            "pip install yt-dlp rich mutagen requests musicbrainzngs "
            "rapidfuzz pyacoustid pydub"
        )
    sys.exit(1)

import argparse
import concurrent.futures
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import acoustid
import mutagen  # type: ignore
import musicbrainzngs
import yt_dlp
from pydub import AudioSegment
from pydub.silence import detect_silence
from rapidfuzz import fuzz
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.rule import Rule
from rich.table import Table
from yt_dlp.utils import DownloadError, ExtractorError, download_range_func  # type: ignore

from audio import download_audio
from config import Config
from metadata import embed_metadata, fetch_musicbrainz
from reports import export_report, update_json_file
from search import build_search_query, search_source, select_best_result
from state import load_state, save_state
from utils import (
    apply_delay,
    check_ffmpeg,
    compute_md5,
    format_duration,
    format_size,
    sanitize_filename,
)

_CONFIG = Config()

fpcalc_available: bool = shutil.which("fpcalc") is not None

_fingerprint_semaphore = threading.Semaphore(2)


def parse_args() -> argparse.Namespace:
    """Build and parse the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="downloader.py",
        description=" YT Music Downloader v2.0 — batch audio download with metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python downloader.py --file songs.json\n"
            "  python downloader.py --file songs.json --format mp3 --quality 320 "
            "--workers 3 --musicbrainz --report json --report csv --report m3u\n"
            "  python downloader.py --file songs.json --dry-run\n"
            "  python downloader.py --file songs.json "
            "--acoustid-key YOUR_KEY --quality 320 --musicbrainz\n"
            "  python downloader.py --data '{\"Tame Impala\": [\"Breathe Deeper\"]}' "
            "--fuzzy-threshold 80"
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", metavar="PATH", type=Path, help="Path to JSON input file")
    src.add_argument("--data", metavar="JSON_STR", help="Inline JSON string")

    p.add_argument(
        "--output", metavar="DIR", type=Path,
        default=Path(_CONFIG.DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {_CONFIG.DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--format", metavar="FORMAT",
        choices=_CONFIG.SUPPORTED_FORMATS,
        default=_CONFIG.DEFAULT_FORMAT,
        help=f"Audio format: {', '.join(_CONFIG.SUPPORTED_FORMATS)} (default: {_CONFIG.DEFAULT_FORMAT})",
    )
    p.add_argument(
        "--quality", metavar="QUALITY",
        choices=["128", "192", "320"],
        default=_CONFIG.DEFAULT_QUALITY,
        help=f"Bitrate kbps: 128|192|320 (default: {_CONFIG.DEFAULT_QUALITY})",
    )

    p.add_argument(
        "--max-results", metavar="INT", type=int,
        default=_CONFIG.DEFAULT_MAX_RESULTS,
        help=f"Max yt-dlp search results per song (default: {_CONFIG.DEFAULT_MAX_RESULTS})",
    )
    p.add_argument(
        "--max-duration", metavar="INT", type=int,
        default=_CONFIG.MAX_DURATION_SECONDS,
        help=f"Max duration in seconds (default: {_CONFIG.MAX_DURATION_SECONDS})",
    )
    p.add_argument(
        "--min-duration", metavar="INT", type=int,
        default=_CONFIG.MIN_DURATION_SECONDS,
        help=f"Min duration in seconds (default: {_CONFIG.MIN_DURATION_SECONDS})",
    )
    p.add_argument(
        "--fuzzy-threshold", metavar="INT", type=int,
        default=_CONFIG.DEFAULT_FUZZY_THRESHOLD,
        help=f"Min rapidfuzz score 0-100 (default: {_CONFIG.DEFAULT_FUZZY_THRESHOLD})",
    )

    p.add_argument(
        "--workers", metavar="INT", type=int,
        default=_CONFIG.DEFAULT_WORKERS,
        help=f"Parallel download threads (default: {_CONFIG.DEFAULT_WORKERS}, max: {_CONFIG.MAX_WORKERS})",
    )
    p.add_argument(
        "--delay", metavar="FLOAT", type=float, nargs=2,
        default=[_CONFIG.DEFAULT_DELAY_MIN, _CONFIG.DEFAULT_DELAY_MAX],
        help=(
            f"Min/max per-worker delay in seconds "
            f"(default: {_CONFIG.DEFAULT_DELAY_MIN} {_CONFIG.DEFAULT_DELAY_MAX})"
        ),
    )

    p.add_argument(
        "--sources", metavar="LIST",
        default=",".join(_CONFIG.DEFAULT_SOURCES),
        help=f"Comma-separated search sources (default: {','.join(_CONFIG.DEFAULT_SOURCES)})",
    )
    p.add_argument(
        "--cookies-browser", metavar="BROWSER",
        choices=["chrome", "firefox", "edge", "safari"],
        help="Extract cookies from browser for age-restricted content",
    )
    p.add_argument("--proxy", metavar="URL", help="SOCKS5/HTTP proxy URL")

    p.add_argument(
        "--musicbrainz", action="store_true",
        help="Enable MusicBrainz metadata enrichment",
    )

    p.add_argument(
        "--acoustid-key", metavar="KEY", dest="acoustid_key",
        help=(
            "AcoustID API key for audio fingerprint verification. "
            "Obtain a free key at https://acoustid.org/login. "
            "When provided and fpcalc is available, enables partial-download + "
            "fingerprint verification before full download."
        ),
    )
    p.add_argument(
        "--skip-fingerprint", action="store_true",
        help=(
            "Disable AcoustID fingerprint verification even if --acoustid-key is provided. "
            "Useful when speed is more important than accuracy."
        ),
    )
    p.add_argument(
        "--score-threshold", metavar="INT", type=int,
        default=_CONFIG.SCORE_THRESHOLD_REJECT,
        help=(
            f"Minimum composite pre-download score required to attempt a download "
            f"(default: {_CONFIG.SCORE_THRESHOLD_REJECT})"
        ),
    )
    p.add_argument(
        "--no-silence-check", action="store_true",
        help="Disable the post-download silence detection check.",
    )

    p.add_argument("--skip-existing", action="store_true", help="Skip already-downloaded songs")
    p.add_argument(
        "--update-json", action="store_true",
        help="Rewrite the input JSON with per-song status when done",
    )
    p.add_argument(
        "--report", metavar="FORMAT", action="append",
        choices=["json", "csv", "m3u"], default=[], dest="report",
        help="Export report: json | csv | m3u (repeatable)",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview matches without downloading")
    p.add_argument("--interactive", action="store_true", help="Confirm each download interactively")
    p.add_argument("--log-file", metavar="PATH", type=Path, help="Write log to file")

    args = p.parse_args()

    args.workers = max(1, min(args.workers, _CONFIG.MAX_WORKERS))
    args.sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    return args


def load_songs(file: Optional[Path], data: Optional[str]) -> dict[str, list[str]]:
    """Return the songs dict from a JSON file or inline string."""
    if file is not None:
        with file.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    if data is not None:
        return json.loads(data)
    raise ValueError("Either --file or --data must be specified.")


def validate_songs(songs: dict) -> bool:
    """Return True iff *songs* is a {str: [str, ...]} dict."""
    if not isinstance(songs, dict):
        return False
    for artist, lst in songs.items():
        if not isinstance(artist, str):
            return False
        if not isinstance(lst, list):
            return False
        if any(not isinstance(s, str) for s in lst):
            return False
    return True


def download_partial_audio(
    url: str,
    output_dir: Path,
    seconds: int,
    extra_opts: dict,
) -> Optional[Path]:
    """
    Download only the first *seconds* seconds of audio from *url*.

    Saves to a UUID-prefixed temp file to avoid worker collisions.
    Returns the Path to the downloaded .mp3, or None on failure.
    The caller is responsible for deleting the file.
    """
    token = uuid4().hex[:8]
    ydl_opts: dict[str, Any] = {
        **{k: v for k, v in extra_opts.items() if v is not None},
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"_partial_{token}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "download_ranges": download_range_func([], [(0, seconds)]),
        "force_keyframes_at_cuts": True,
    }
    if extra_opts.get("cookies_browser"):
        ydl_opts["cookiesfrombrowser"] = (extra_opts["cookies_browser"],)
    if extra_opts.get("proxy"):
        ydl_opts["proxy"] = extra_opts["proxy"]

    expected = output_dir / f"_partial_{token}.mp3"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            ydl.download([url])
        if expected.exists():
            return expected
        for candidate in output_dir.glob(f"_partial_{token}.*"):
            return candidate
    except Exception:
        pass
    return None


def verify_fingerprint(
    partial_path: Path,
    expected_artist: str,
    expected_song: str,
    acoustid_api_key: str,
    config: Config,
) -> tuple[bool, float, str]:
    """
    Fingerprint *partial_path* via Chromaprint and query the AcoustID API.

    Returns (is_match, confidence, matched_title).
    Handles AcoustID errors gracefully — never crashes the caller.
    """
    try:
        results = list(
            acoustid.match(acoustid_api_key, str(partial_path), meta="recordings")
        )

        best_confidence = 0.0
        best_title = ""

        for score, _recording_id, title, artist in results:
            if score < config.FINGERPRINT_MIN_CONFIDENCE:
                continue
            artist_sim = fuzz.token_sort_ratio(
                expected_artist.lower(), (artist or "").lower()
            )
            title_sim = fuzz.token_sort_ratio(
                expected_song.lower(), (title or "").lower()
            )
            if artist_sim > 75 and title_sim > 75:
                return True, score, title or ""
            if score > best_confidence:
                best_confidence = score
                best_title = f"{artist} -- {title}"

        return False, best_confidence, best_title

    except (acoustid.AcoustidError, acoustid.WebServiceError) as exc:
        _ = exc
        return False, 0.0, "fingerprint_error"
    except Exception:
        return False, 0.0, "fingerprint_error"


def has_excessive_silence(
    file_path: Path,
    config: Config,
) -> tuple[bool, float]:
    """
    Analyse *file_path* for excessive silence.

    Returns (is_excessive, silence_ratio) where silence_ratio is 0.0-1.0.
    Handles pydub decode errors gracefully — never crashes the caller.
    """
    try:
        audio = AudioSegment.from_file(str(file_path))
        silent_ranges = detect_silence(
            audio,
            min_silence_len=config.SILENCE_MIN_DURATION_MS,
            silence_thresh=config.SILENCE_THRESHOLD_DB,
        )
        total_silent_ms = sum(end - start for start, end in silent_ranges)
        ratio = total_silent_ms / len(audio) if len(audio) > 0 else 0.0
        return ratio > config.EXCESSIVE_SILENCE_RATIO, ratio
    except Exception:
        return False, 0.0


def verify_duration_match(
    downloaded_path: Path,
    expected_duration_seconds: int | None,
    tolerance: float = 0.20,
) -> tuple[bool, int]:
    """
    Compare the actual audio duration (via mutagen) against *expected_duration_seconds*.

    Returns (is_match, actual_duration_seconds).
    If expected is None, always returns (True, actual_duration).
    Returns (False, 0) if mutagen cannot read the file.
    """
    try:
        info = mutagen.File(str(downloaded_path))  # type: ignore
        if info is None or info.info is None:
            return False, 0
        actual = int(info.info.length)
        if expected_duration_seconds is None:
            return True, actual
        ratio = abs(actual - expected_duration_seconds) / max(expected_duration_seconds, 1)
        return ratio <= tolerance, actual
    except Exception:
        return False, 0

def process_song(
    artist: str,
    song: str,
    args: argparse.Namespace,
    config: Config,
    state: dict,
    state_lock: threading.Lock,
    console: Console,
    console_lock: threading.Lock,
    progress: Progress,
    stop_event: threading.Event,
    printed_artists: set,
    printed_artists_lock: threading.Lock,
    pairs: list[tuple[str, str]],
    interactive_lock: Optional[threading.Lock] = None,
) -> dict:
    """
    Execute the full download pipeline for one (artist, song) pair.

    Pipeline:
        1.  Check persistent state / --skip-existing + MD5 verification
        2.  Apply per-worker random delay
        3.  Search each source in --sources order
        4.  Select best result via composite scoring (select_best_result)
        5.  Optional AcoustID fingerprint verification
        6.  Dry-run / interactive gate
        7.  Download with retry + exponential backoff
        8.  Post-download: duration match + silence check
        9.  Embed metadata (+ optional MusicBrainz enrichment)
        10. MD5 hash and state save

    Returns:
        A result record dict suitable for report export.
    """
    with printed_artists_lock:
        if artist not in printed_artists:
            printed_artists.add(artist)
            count = sum(1 for a, _ in pairs if a == artist)
            with console_lock:
                console.print(Rule(
                    f"[bold cyan]{artist}[/bold cyan] [dim]({count} songs)[/dim]"
                ))

    key = f"{artist}::{song}"
    result: dict[str, Any] = {
        "artist": artist,
        "song": song,
        "status": "failed",
        "source": None,
        "url": None,
        "matched_title": None,
        "fuzzy_score": 0,
        "duration_seconds": 0,
        "file_path": None,
        "file_size_bytes": 0,
        "md5": None,
        "musicbrainz_enriched": False,
        "album": None,
        "year": None,
        "genre": None,
        "reason": None,

        "composite_score": 0,
        "score_breakdown": {},
        "fingerprint_verified": False,
        "fingerprint_confidence": 0.0,
        "fingerprint_matched_title": None,
        "silence_ratio": 0.0,
        "duration_verified": True,
    }

    if stop_event.is_set():
        result.update(status="skipped", reason="Interrupted")
        return result

    output_dir: Path = args.output
    safe_artist = sanitize_filename(artist)
    safe_song = sanitize_filename(song)
    expected_file = output_dir / safe_artist / f"{safe_song}.{args.format}"

    with state_lock:
        existing = state.get("downloads", {}).get(key)

    if args.skip_existing and existing and existing.get("status") == "downloaded":
        stored_md5 = existing.get("md5")
        if expected_file.exists():
            if stored_md5:
                current_md5 = compute_md5(expected_file)
                if current_md5 == stored_md5:
                    with console_lock:
                        console.print(
                            f"[yellow] Skipped (exists): {artist} -- {song}[/yellow]"
                        )
                    result.update(
                        status="skipped",
                        file_path=str(expected_file),
                        md5=stored_md5,
                    )
                    return result
                with console_lock:
                    console.print(
                        f"[blue] MD5 mismatch for '{song}' -- re-downloading...[/blue]"
                    )
            else:
                with console_lock:
                    console.print(
                        f"[yellow] Skipped (exists, no MD5): {artist} -- {song}[/yellow]"
                    )
                result.update(status="skipped", file_path=str(expected_file))
                return result

    apply_delay(args.delay[0], args.delay[1])

    search_opts: dict[str, Any] = {
        "max_results": args.max_results,
        "cookies_browser": args.cookies_browser,
        "proxy": args.proxy,
    }

    best_result: Optional[dict] = None
    ranked_candidates: list[tuple[dict, int, dict]] = []
    chosen_source: Optional[str] = None
    score_threshold: int = getattr(args, "score_threshold", config.SCORE_THRESHOLD_REJECT)

    for source in args.sources:
        if stop_event.is_set():
            break

        query = build_search_query(artist, song, source)
        with console_lock:
            console.print(f"[dim]  [{source}] {song} -- {artist}[/dim]")

        raw = search_source(query, source, search_opts)
        if not raw:
            with console_lock:
                console.print(f"[dim]  no results from {source}[/dim]")
            continue

        found, ranked = select_best_result(
            raw, artist, song,
            mb_duration_seconds=None,
            config=config,
            console=console,
            console_lock=console_lock,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            score_threshold=score_threshold,
        )

        if found:
            best_result = found
            ranked_candidates = ranked
            chosen_source = source
            break

    if best_result is None:
        with console_lock:
            console.print(
                Panel(
                    f"[red]No valid result for: [bold]{artist} -- {song}[/bold][/red]\n"
                    f"Sources tried: {', '.join(args.sources)}\n"
                    f"Duration: [{format_duration(args.min_duration)}--"
                    f"{format_duration(args.max_duration)}] | "
                    f"Score threshold: {score_threshold}",
                    title="[bold red] Search Failed[/bold red]",
                    border_style="red",
                )
            )
        result["reason"] = "No valid result found after all sources"
        _persist(state, state_lock, key, "failed", None, None, None, output_dir)
        return result

    url: str = best_result.get("webpage_url") or best_result.get("url", "")
    matched_title: str = best_result.get("title") or ""
    fuzzy_score: int = best_result.get("_composite_score", 0)
    duration_s: int = int(best_result.get("duration") or 0)
    thumbnail_url: Optional[str] = best_result.get("thumbnail")
    composite_score: int = best_result.get("_composite_score", 0)
    score_breakdown: dict = best_result.get("_score_breakdown", {})

    result.update(
        source=chosen_source,
        url=url,
        matched_title=matched_title,
        fuzzy_score=fuzzy_score,
        duration_seconds=duration_s,
        composite_score=composite_score,
        score_breakdown=score_breakdown,
    )

    fp_verified = False
    fp_confidence = 0.0
    fp_matched_title: Optional[str] = None
    fp_status_label = "-- disabled"

    needs_fingerprint = (
        bool(getattr(args, "acoustid_key", None))
        and not getattr(args, "skip_fingerprint", False)
        and fpcalc_available
        and composite_score < config.SCORE_THRESHOLD_SKIP_FINGERPRINT
    )

    if getattr(args, "acoustid_key", None) and composite_score >= config.SCORE_THRESHOLD_SKIP_FINGERPRINT:
        fp_status_label = f"skipped -- score threshold met ({composite_score})"
    elif getattr(args, "acoustid_key", None) and not fpcalc_available:
        fp_status_label = "disabled -- fpcalc not found"

    if needs_fingerprint:
        t_fp_start = time.monotonic()
        with console_lock:
            console.print(
                f"[dim]  Verifying audio fingerprint "
                f"(downloading {config.PARTIAL_DOWNLOAD_SECONDS}s)...[/dim]"
            )
        partial_path: Optional[Path] = None
        try:
            with _fingerprint_semaphore:
                partial_path = download_partial_audio(
                    url,
                    output_dir,
                    config.PARTIAL_DOWNLOAD_SECONDS,
                    {"cookies_browser": args.cookies_browser, "proxy": args.proxy},
                )

                if partial_path is None:
                    with console_lock:
                        console.print(
                            "[yellow]  Fingerprint: partial download failed, "
                            "proceeding anyway[/yellow]"
                        )
                    fp_status_label = "partial download failed"
                else:
                    is_match, confidence, fp_title = verify_fingerprint(
                        partial_path, artist, song, args.acoustid_key, config
                    )
                    time.sleep(0.35)  # AcoustID rate limit
                    fp_time = time.monotonic() - t_fp_start
                    fp_confidence = confidence
                    fp_matched_title = fp_title

                    if is_match:
                        fp_verified = True
                        fp_status_label = (
                            f"verified {confidence:.0%} conf. ({fp_time:.1f}s)"
                        )
                        with console_lock:
                            console.print(
                                f"[green]  Fingerprint: verified "
                                f"({confidence:.0%} confidence, {fp_time:.1f}s)[/green]"
                            )
                    elif confidence > 0.4:
                        fp_status_label = f"low confidence ({fp_title})"
                        with console_lock:
                            console.print(
                                f"[yellow]  Fingerprint: low confidence match "
                                f"({fp_title}), trying next result[/yellow]"
                            )

                        _tried_next = False
                        for cand_result, cand_score, _bd in ranked_candidates[1:]:
                            if cand_score < score_threshold:
                                break
                            next_url = (
                                cand_result.get("webpage_url")
                                or cand_result.get("url", "")
                            )
                            next_partial: Optional[Path] = None
                            try:
                                next_partial = download_partial_audio(
                                    next_url,
                                    output_dir,
                                    config.PARTIAL_DOWNLOAD_SECONDS,
                                    {
                                        "cookies_browser": args.cookies_browser,
                                        "proxy": args.proxy,
                                    },
                                )
                                if next_partial:
                                    n_match, n_conf, n_title = verify_fingerprint(
                                        next_partial, artist, song,
                                        args.acoustid_key, config,
                                    )
                                    time.sleep(0.35)
                                    if n_match:
                                        best_result = cand_result
                                        url = next_url
                                        matched_title = cand_result.get("title") or ""
                                        duration_s = int(cand_result.get("duration") or 0)
                                        fp_verified = True
                                        fp_confidence = n_conf
                                        fp_matched_title = n_title
                                        fp_status_label = (
                                            f"verified next candidate {n_conf:.0%}"
                                        )
                                        composite_score = cand_result.get(
                                            "_composite_score", 0
                                        )
                                        score_breakdown = cand_result.get(
                                            "_score_breakdown", {}
                                        )
                                        result.update(
                                            composite_score=composite_score,
                                            score_breakdown=score_breakdown,
                                            url=url,
                                            matched_title=matched_title,
                                            duration_seconds=duration_s,
                                        )
                                        _tried_next = True
                                        break
                            finally:
                                if next_partial and next_partial.exists():
                                    try:
                                        next_partial.unlink(missing_ok=True)
                                    except OSError:
                                        pass
                            break  # only try one next candidate

                        if not _tried_next:
                            with console_lock:
                                console.print(
                                    "[yellow]  Fingerprint: next candidate also "
                                    "failed or unavailable, proceeding with "
                                    "original[/yellow]"
                                )
                            fp_status_label = "no match (both candidates tried)"
                    else:
                        fp_status_label = "no AcoustID match found"
                        with console_lock:
                            console.print(
                                "[yellow]  Fingerprint: no AcoustID match found, "
                                "proceeding with score-based selection[/yellow]"
                            )
        finally:
            if partial_path and partial_path.exists():
                try:
                    partial_path.unlink(missing_ok=True)
                except OSError:
                    pass

    if composite_score >= config.SCORE_THRESHOLD_SKIP_FINGERPRINT:
        score_color = "green"
        score_label = "high confidence"
    elif composite_score >= config.SCORE_THRESHOLD_REJECT:
        score_color = "yellow"
        score_label = "moderate"
    else:
        score_color = "red"
        score_label = "low"

    with console_lock:
        console.print(
            f"  Verification: "
            f"[{score_color}][SCORE: {composite_score} {score_label}][/{score_color}] "
            f"[FINGERPRINT: {fp_status_label}]"
        )

    result.update(
        fingerprint_verified=fp_verified,
        fingerprint_confidence=fp_confidence,
        fingerprint_matched_title=fp_matched_title,
    )

    if args.dry_run:
        result["status"] = "skipped"
        result["reason"] = "dry-run"
        return result

    if args.interactive and interactive_lock is not None:
        with interactive_lock:
            with console_lock:
                console.print(
                    Panel(
                        f"[bold]Title:[/bold] {matched_title}\n"
                        f"[bold]Channel:[/bold] "
                        f"{best_result.get('channel') or best_result.get('uploader', '')}\n"
                        f"[bold]Duration:[/bold] {format_duration(duration_s)}\n"
                        f"[bold]Score:[/bold] {composite_score}\n"
                        f"[bold]URL:[/bold] {url}\n"
                        f"[bold]Thumb:[/bold] {thumbnail_url or 'N/A'}",
                        title=f"[bold cyan]  {artist} -- {song}[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print(
                    "[bold yellow]\\[Y][/bold yellow] Download "
                    "[bold yellow]\\[n][/bold yellow] Skip "
                    "[bold yellow]\\[s][/bold yellow] Try next result "
                    "[bold yellow]\\[q][/bold yellow] Quit"
                )
            try:
                choice = input("Choice [Y/n/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

        if choice == "q":
            stop_event.set()
            result.update(status="skipped", reason="User quit")
            return result
        if choice == "n":
            result.update(status="skipped", reason="User skipped")
            return result

    (output_dir / safe_artist).mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_artist / safe_song

    task_id: TaskID = progress.add_task(
        f"[cyan]{song[:45]}[/cyan]",
        total=100,
        visible=True,
    )

    dl_opts: dict[str, Any] = {
        "cookies_browser": args.cookies_browser,
        "proxy": args.proxy,
    }

    downloaded_file: Optional[Path] = None
    last_error = ""

    for attempt in range(1, config.RETRY_ATTEMPTS + 1):
        if stop_event.is_set():
            break
        try:
            downloaded_file = download_audio(
                url, output_path, args.format, args.quality,
                progress, task_id, dl_opts,
            )
            break

        except DownloadError as exc:
            last_error = f"DownloadError: {exc}"
        except ExtractorError as exc:
            last_error = f"ExtractorError: {exc}"
        except OSError as exc:
            if exc.errno == 28:
                stop_event.set()
                with console_lock:
                    console.print(
                        Panel(
                            "[bold red]Disk full![/bold red]\n"
                            "Free space and restart with [bold]--skip-existing[/bold].",
                            title="[bold red] Disk Full[/bold red]",
                            border_style="red",
                        )
                    )
                save_state(state, output_dir)
                sys.exit(1)
            last_error = f"OSError: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < config.RETRY_ATTEMPTS:
            wait = config.RETRY_BACKOFF_BASE ** attempt
            with console_lock:
                console.print(
                    f"[blue]  Retry {attempt}/{config.RETRY_ATTEMPTS}: "
                    f"{song} -- {last_error} (waiting {wait:.1f}s)[/blue]"
                )
            time.sleep(wait)

    progress.remove_task(task_id)

    if downloaded_file is None or not downloaded_file.exists():
        with console_lock:
            console.print(
                Panel(
                    f"[red][bold]{artist} -- {song}[/bold][/red]\n{last_error}",
                    title="[bold red] Download Failed[/bold red]",
                    border_style="red",
                )
            )
        result["reason"] = last_error
        _persist(state, state_lock, key, "failed", url, None, None, output_dir)
        return result

    actual_dur = duration_s

    dur_ok, actual_dur = verify_duration_match(downloaded_file, duration_s)
    result["duration_verified"] = dur_ok
    if not dur_ok:
        discrepancy = (
            abs(actual_dur - duration_s) / max(duration_s, 1)
            if duration_s > 0
            else 1.0
        )
        with console_lock:
            console.print(
                f"[yellow]  Duration mismatch: expected {format_duration(duration_s)}, "
                f"got {format_duration(actual_dur)}[/yellow]"
            )
        if discrepancy > 0.40:
            with console_lock:
                console.print(
                    f"[red]  Duration discrepancy {discrepancy:.0%} > 40%, "
                    "marking as failed[/red]"
                )
            try:
                downloaded_file.unlink(missing_ok=True)
            except OSError:
                pass
            result["reason"] = f"Duration discrepancy {discrepancy:.0%}"
            _persist(state, state_lock, key, "failed", url, None, None, output_dir)
            return result

    silence_ratio = 0.0
    if not getattr(args, "no_silence_check", False):
        is_excessive, silence_ratio = has_excessive_silence(downloaded_file, config)
        result["silence_ratio"] = silence_ratio
        if is_excessive:
            with console_lock:
                console.print(
                    Panel(
                        f"[red]Rejected: {silence_ratio:.1%} silence detected "
                        "(likely copyright block or muted audio)[/red]",
                        title=f"[bold red] Silence Check Failed: {artist} -- {song}[/bold red]",
                        border_style="red",
                    )
                )
            try:
                downloaded_file.unlink(missing_ok=True)
            except OSError:
                pass
            result["reason"] = f"Excessive silence detected ({silence_ratio:.1%})"
            _persist(state, state_lock, key, "failed", url, None, None, output_dir)
            return result
        elif silence_ratio > 0.15:
            with console_lock:
                console.print(
                    f"[yellow]  Warning: {silence_ratio:.1%} silence in audio[/yellow]"
                )

    dur_label = (
        f"DURATION: {format_duration(actual_dur)} matches"
        if dur_ok
        else f"DURATION: mismatch {format_duration(actual_dur)}"
    )
    if silence_ratio <= 0.0:
        sil_label = "SILENCE: check disabled"
    elif silence_ratio <= 0.15:
        sil_label = f"SILENCE: {silence_ratio:.1%} -- normal"
    elif silence_ratio <= 0.30:
        sil_label = f"SILENCE: {silence_ratio:.1%} -- elevated"
    else:
        sil_label = f"SILENCE: {silence_ratio:.1%} -- excessive"

    with console_lock:
        console.print(f"  Post-check: [{dur_label}] [{sil_label}]")

    mb_data: Optional[dict] = None
    mb_enriched = False

    if args.musicbrainz:
        mb_data = fetch_musicbrainz(artist, song)
        if mb_data:
            mb_enriched = True
            with console_lock:
                console.print(
                    f"[green]  MusicBrainz: ENRICHED -- "
                    f"album={mb_data.get('album') or 'N/A'}, "
                    f"year={mb_data.get('year') or 'N/A'}, "
                    f"genre={mb_data.get('genre') or 'N/A'}[/green]"
                )
        else:
            with console_lock:
                console.print(
                    "[yellow]  MusicBrainz: FALLBACK -- no match found[/yellow]"
                )

    extra: dict[str, Any] = {
        "source_url": url,
        "album": mb_data.get("album") if mb_data else None,
        "year": mb_data.get("year") if mb_data else None,
        "genre": mb_data.get("genre") if mb_data else None,
        "track_num": mb_data.get("track_num") if mb_data else None,
        "mb_id": mb_data.get("mb_id") if mb_data else None,
        "cover_url": mb_data.get("cover_url") if mb_data else None,
    }

    def _warn(msg: str) -> None:
        with console_lock:
            console.print(msg)

    embed_ok = embed_metadata(
        downloaded_file, song, artist, extra, thumbnail_url, args.format, _warn
    )

    if not embed_ok:
        with console_lock:
            console.print(
                Panel(
                    f"[red]Integrity check failed for [bold]{downloaded_file.name}[/bold].\n"
                    "File deleted -- marked as failed.",
                    title="[bold red] Metadata Error[/bold red]",
                    border_style="red",
                )
            )
        try:
            downloaded_file.unlink(missing_ok=True)
        except OSError:
            pass
        result["reason"] = "Metadata integrity check failed"
        _persist(state, state_lock, key, "failed", url, None, None, output_dir)
        return result

    md5 = compute_md5(downloaded_file)
    file_size = downloaded_file.stat().st_size

    result.update(
        status="downloaded",
        file_path=str(downloaded_file),
        file_size_bytes=file_size,
        md5=md5,
        musicbrainz_enriched=mb_enriched,
        album=extra.get("album"),
        year=extra.get("year"),
        genre=extra.get("genre"),
        silence_ratio=silence_ratio,
        duration_verified=dur_ok,
    )

    _persist(
        state, state_lock, key, "downloaded",
        url, str(downloaded_file), md5, output_dir,
    )

    with console_lock:
        console.print(
            f"[green]  Downloaded: {artist} -- {song} "
            f"({format_duration(duration_s)}, {format_size(file_size)})[/green]"
        )
    return result


def _persist(
    state: dict,
    lock: threading.Lock,
    key: str,
    status: str,
    url: Optional[str],
    file_path: Optional[str],
    md5: Optional[str],
    output_dir: Path,
) -> None:
    """Update the shared state dict and flush to disk (thread-safe)."""
    with lock:
        state.setdefault("downloads", {})[key] = {
            "status": status,
            "url": url,
            "file_path": file_path,
            "md5": md5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state, output_dir)


def _print_startup_banner(
    console: Console,
    args: argparse.Namespace,
    total: int,
) -> None:
    acoustid_status = (
        "enabled (fpcalc found)"
        if getattr(args, "acoustid_key", None) and fpcalc_available
        else "disabled"
        if not getattr(args, "acoustid_key", None)
        else "key set but fpcalc not found"
    )
    console.print(
        Panel(
            f"[bold]Output dir:[/bold] {args.output.resolve()}\n"
            f"[bold]Format:[/bold] {args.format} @ {args.quality} kbps\n"
            f"[bold]Workers:[/bold] {args.workers}\n"
            f"[bold]Total songs:[/bold] {total}\n"
            f"[bold]Sources:[/bold] {' -> '.join(args.sources)}\n"
            f"[bold]MusicBrainz:[/bold] {'enabled' if args.musicbrainz else 'disabled'}\n"
            f"[bold]Dry run:[/bold] {'yes' if args.dry_run else 'no'}\n"
            f"[bold]Interactive:[/bold] {'yes' if args.interactive else 'no'}\n"
            f"[bold]Skip existing:[/bold] {'yes' if args.skip_existing else 'no'}\n"
            f"[bold]AcoustID:[/bold] {acoustid_status}\n"
            f"[bold]Silence check:[/bold] "
            f"{'disabled' if getattr(args, 'no_silence_check', False) else 'enabled'}\n"
            f"[bold]Score threshold:[/bold] "
            f"{getattr(args, 'score_threshold', _CONFIG.SCORE_THRESHOLD_REJECT)}",
            title="[bold cyan] YT Music Downloader v2.0[/bold cyan]",
            border_style="cyan",
        )
    )


def _print_summary(
    console: Console,
    results: list[dict],
    elapsed: float,
) -> None:
    """Render the final Rich summary table."""
    downloaded = [r for r in results if r.get("status") == "downloaded"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") == "failed"]
    total_bytes = sum(r.get("file_size_bytes", 0) for r in downloaded)

    tbl = Table(
        title="[bold] Download Summary[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        caption=(
            f" {len(downloaded)} downloaded "
            f" {len(skipped)} skipped "
            f" {len(failed)} failed | "
            f" {elapsed:.1f}s | {format_size(total_bytes)}"
        ),
    )
    tbl.add_column("#", style="dim", width=4)
    tbl.add_column("Artist", style="cyan", min_width=14)
    tbl.add_column("Song", style="white", min_width=18)
    tbl.add_column("Status", min_width=14)
    tbl.add_column("Duration", style="yellow", width=10)
    tbl.add_column("Fuzzy", style="magenta", width=6)
    tbl.add_column("Score", width=7)
    tbl.add_column("Fingerprint", width=16)
    tbl.add_column("Silence", width=10)
    tbl.add_column("MusicBrainz", style="blue", width=12)
    tbl.add_column("Size", style="green", width=9)
    tbl.add_column("File / Reason", style="dim", min_width=28)

    for i, r in enumerate(results, 1):
        status = r.get("status", "unknown")
        if status == "downloaded":
            status_cell = "[green] downloaded[/green]"
        elif status == "skipped":
            status_cell = "[yellow] skipped[/yellow]"
        else:
            status_cell = "[red] failed[/red]"

        dur = r.get("duration_seconds") or 0
        mb = "yes" if r.get("musicbrainz_enriched") else "--"
        sz = format_size(r["file_size_bytes"]) if r.get("file_size_bytes") else "--"
        detail = (r.get("file_path") or r.get("reason") or "--")[:55]

        sc = r.get("composite_score", 0)
        if sc >= 70:
            score_cell = f"[green]{sc}[/green]"
        elif sc >= 30:
            score_cell = f"[yellow]{sc}[/yellow]"
        else:
            score_cell = f"[red]{sc}[/red]" if sc > 0 else "--"

        if r.get("fingerprint_verified"):
            conf = r.get("fingerprint_confidence", 0.0)
            fp_cell = f"[green]verified {conf:.0%}[/green]"
        elif r.get("fingerprint_confidence", 0.0) > 0:
            fp_cell = "[yellow]no match[/yellow]"
        else:
            fp_cell = "[dim]-- disabled[/dim]"

        sil = r.get("silence_ratio", 0.0)
        if sil <= 0.0:
            sil_cell = "[dim]--[/dim]"
        elif sil < 0.15:
            sil_cell = f"[green]{sil:.1%}[/green]"
        elif sil < 0.30:
            sil_cell = f"[yellow]{sil:.1%}[/yellow]"
        else:
            sil_cell = f"[red]{sil:.1%}[/red]"

        tbl.add_row(
            str(i),
            r.get("artist", ""),
            r.get("song", ""),
            status_cell,
            format_duration(int(dur)) if dur else "--",
            str(r.get("fuzzy_score", 0)),
            score_cell,
            fp_cell,
            sil_cell,
            mb, sz, detail,
        )

    console.print(tbl)


def _dry_run_table(
    console: Console,
    pairs: list[tuple[str, str]],
    args: argparse.Namespace,
    config: Config,
) -> None:
    """Search all sources in dry-run mode and render a scored preview table."""
    tbl = Table(
        title="[bold cyan] Dry Run Preview[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
    )
    tbl.add_column("Artist", style="cyan", min_width=14)
    tbl.add_column("Song", style="white", min_width=18)
    tbl.add_column("Source", style="dim", width=12)
    tbl.add_column("Matched Title", style="green", min_width=30)
    tbl.add_column("Channel", style="dim", min_width=18)
    tbl.add_column("Duration", style="yellow", width=10)
    tbl.add_column("Score", width=7)
    tbl.add_column("Top 3 signals", min_width=35, style="dim")
    tbl.add_column("Fingerprint avail.", width=18)

    fp_avail_label = (
        "yes" if (getattr(args, "acoustid_key", None) and fpcalc_available)
        else "no key"
        if not getattr(args, "acoustid_key", None)
        else "fpcalc missing"
    )

    opts: dict[str, Any] = {
        "max_results": args.max_results,
        "cookies_browser": args.cookies_browser,
        "proxy": args.proxy,
    }
    score_threshold = getattr(args, "score_threshold", config.SCORE_THRESHOLD_REJECT)

    for artist, song in pairs:
        found: Optional[dict] = None
        src_used: Optional[str] = None
        ranked: list = []

        for source in args.sources:
            raw = search_source(build_search_query(artist, song, source), source, opts)
            if raw:
                found, ranked = select_best_result(
                    raw, artist, song,
                    mb_duration_seconds=None,
                    config=config,
                    console=console,
                    console_lock=None,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration,
                    score_threshold=score_threshold,
                )
                if found:
                    src_used = source
                    break

        if found:
            dur = int(found.get("duration") or 0)
            sc = found.get("_composite_score", 0)
            bd = found.get("_score_breakdown", {})
            top_signals = sorted(bd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            signals_str = ", ".join(
                f"{'+' if v >= 0 else ''}{v} {k}" for k, v in top_signals
            )
            if sc >= 70:
                score_cell = f"[green]{sc}[/green]"
            elif sc >= 30:
                score_cell = f"[yellow]{sc}[/yellow]"
            else:
                score_cell = f"[red]{sc}[/red]"

            tbl.add_row(
                artist, song,
                src_used or "--",
                (found.get("title") or "")[:48],
                (found.get("channel") or found.get("uploader") or "")[:26],
                format_duration(dur),
                score_cell,
                signals_str,
                fp_avail_label,
            )
        else:
            tbl.add_row(
                artist, song, "--",
                "[red]No match found[/red]",
                "--", "--", "--", "--", fp_avail_label,
            )

    console.print(tbl)


def _check_fpcalc(console: Console) -> None:
    """Print a warning panel if fpcalc is not available (non-fatal)."""
    if not fpcalc_available:
        console.print(
            Panel(
                "[yellow]fpcalc (Chromaprint) not found. "
                "Audio fingerprint verification is disabled.[/yellow]\n\n"
                "Install it from: https://acoustid.org/chromaprint\n"
                "  Ubuntu/Debian : sudo apt install libchromaprint-tools\n"
                "  macOS         : brew install chromaprint\n"
                "  Windows       : download from https://acoustid.org/chromaprint\n\n"
                "The --acoustid-key flag will be ignored until fpcalc is available.",
                title="[bold yellow] fpcalc Not Found[/bold yellow]",
                border_style="yellow",
            )
        )


def main() -> None:
    """Orchestrate the full download session."""
    args = parse_args()
    config = Config()

    log_file_handle = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = args.log_file.open("w", encoding="utf-8")
        console = Console(record=True, file=log_file_handle)
    else:
        console = Console()

    check_ffmpeg(console)
    _check_fpcalc(console)

    if args.musicbrainz:
        musicbrainzngs.set_useragent(
            "YTMusicDownloader", "2.0",
            "https://github.com/example/yt-music-downloader",
        )

    try:
        songs = load_songs(args.file, args.data)
    except (json.JSONDecodeError, OSError) as exc:
        console.print(
            Panel(
                f"[red]Could not load input: {exc}[/red]",
                title="[bold red] Input Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)

    if not validate_songs(songs):
        console.print(
            Panel(
                '[red]Invalid JSON structure.[/red]\n'
                'Expected: [bold]{ "Artist": ["song1", "song2"] }[/bold]',
                title="[bold red] Validation Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)

    for artist, lst in songs.items():
        if not lst:
            console.print(f"[yellow]  Skipping '{artist}': empty song list[/yellow]")

    pairs: list[tuple[str, str]] = [
        (artist, song)
        for artist, lst in songs.items()
        for song in lst
        if lst
    ]
    total = len(pairs)

    args.output.mkdir(parents=True, exist_ok=True)
    state = load_state(args.output)
    state_lock = threading.Lock()
    console_lock = threading.Lock()
    stop_event = threading.Event()
    interactive_lock: Optional[threading.Lock] = (
        threading.Lock() if args.interactive else None
    )
    printed_artists: set[str] = set()
    printed_artists_lock = threading.Lock()

    _print_startup_banner(console, args, total)

    if args.dry_run:
        _dry_run_table(console, pairs, args, config)
        if log_file_handle:
            log_file_handle.close()
        return

    all_results: list[dict] = []
    start = time.monotonic()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
        transient=False,
    )

    try:
        with progress:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                futures: dict[concurrent.futures.Future, tuple[str, str]] = {}

                for artist, song in pairs:
                    if stop_event.is_set():
                        break
                    fut = executor.submit(
                        process_song,
                        artist, song, args, config,
                        state, state_lock,
                        console, console_lock,
                        progress, stop_event,
                        printed_artists,
                        printed_artists_lock,
                        pairs,
                        interactive_lock,
                    )
                    futures[fut] = (artist, song)

                for fut in concurrent.futures.as_completed(futures):
                    artist, song = futures[fut]
                    try:
                        all_results.append(fut.result())
                    except Exception as exc:
                        with console_lock:
                            console.print(
                                Panel(
                                    f"[red]{type(exc).__name__}: {exc}[/red]",
                                    title=f"[bold red] Unhandled: {artist} -- {song}[/bold red]",
                                    border_style="red",
                                )
                            )
                        all_results.append(
                            {
                                "artist": artist, "song": song,
                                "status": "failed", "reason": str(exc),
                            }
                        )

    except KeyboardInterrupt:
        stop_event.set()
        elapsed = time.monotonic() - start
        console.print(
            Panel(
                f"[yellow] Interrupted by user after {elapsed:.1f}s\n"
                f"Completed: {len(all_results)}/{total} songs[/yellow]",
                title="[bold yellow] Interrupted[/bold yellow]",
                border_style="yellow",
            )
        )
        save_state(state, args.output)
        _print_summary(console, all_results, elapsed)
        if log_file_handle:
            log_file_handle.close()
        sys.exit(130)

    elapsed = time.monotonic() - start
    _print_summary(console, all_results, elapsed)

    if args.report:
        export_report(all_results, args.output, args.report)
        console.print(
            f"[green]  Reports saved to: {args.output.resolve()}[/green]"
        )

    if args.update_json and args.file:
        console.print(
            f"[bold yellow]  About to overwrite input file: {args.file}[/bold yellow]"
        )
        update_json_file(args.file, all_results)
        console.print(f"[green]  Input JSON updated: {args.file}[/green]")

    if log_file_handle:
        log_file_handle.close()


if __name__ == "__main__":
    main()