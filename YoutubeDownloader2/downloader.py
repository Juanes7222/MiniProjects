"""
YT Music Downloader v2.0
════════════════════════
A production-quality CLI tool that searches YouTube (and SoundCloud / Bandcamp
as fallbacks) and downloads audio tracks defined in a structured JSON file.

Usage
-----
    python downloader.py --file songs.json [OPTIONS]
    python downloader.py --data '{"Artist": ["Song"]}' [OPTIONS]

Dependencies
------------
    pip install yt-dlp rich mutagen requests musicbrainzngs rapidfuzz

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
                + "\n".join(f"  • {p}" for p in _MISSING)
                + "\n\n[yellow]Fix with:[/yellow]\n"
                "  [bold]pip install yt-dlp rich mutagen requests musicbrainzngs rapidfuzz[/bold]",
                title="[bold red] Missing Dependencies[/bold red]",
                border_style="red",
            )
        )
    except ImportError:
        print("Missing packages:", ", ".join(_MISSING))
        print("pip install yt-dlp rich mutagen requests musicbrainzngs rapidfuzz")
    sys.exit(1)

import argparse
import concurrent.futures
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import musicbrainzngs
from yt_dlp.utils import DownloadError, ExtractorError # type: ignore
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

from audio import download_audio
from config import Config
from metadata import embed_metadata, fetch_musicbrainz
from reports import export_report, update_json_file
from search import build_search_query, filter_results, search_source
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

    # Search parameters
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
        help=f"Min rapidfuzz score 0–100 (default: {_CONFIG.DEFAULT_FUZZY_THRESHOLD})",
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
        1. Check persistent state / --skip-existing + MD5 verification
        2. Apply per-worker random delay
        3. Search each source in --sources order
        4. Log all candidates with fuzzy scores
        5. Dry-run / interactive gate
        6. Download with retry + exponential backoff
        7. Embed metadata (+ optional MusicBrainz enrichment)
        8. MD5 hash and state save

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
                            f"[yellow]  Skipped (exists): {artist} — {song}[/yellow]"
                        )
                    result.update(
                        status="skipped",
                        file_path=str(expected_file),
                        md5=stored_md5,
                    )
                    return result
                # MD5 mismatch → re-download silently
                with console_lock:
                    console.print(
                        f"[blue] MD5 mismatch for '{song}' — re-downloading...[/blue]"
                    )
            else:
                with console_lock:
                    console.print(
                        f"[yellow]  Skipped (exists, no MD5): {artist} — {song}[/yellow]"
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
    all_candidates: list[dict] = []
    chosen_source: Optional[str] = None

    for source in args.sources:
        if stop_event.is_set():
            break

        query = build_search_query(artist, song, source)
        with console_lock:
            console.print(
                f"[dim]   [{source}] {song} — {artist}[/dim]"
            )

        raw = search_source(query, source, search_opts)
        if not raw:
            with console_lock:
                console.print(f"[dim]     └─ no results from {source}[/dim]")
            continue

        found, candidates = filter_results(
            raw, artist, song, args.max_duration, args.min_duration, args.fuzzy_threshold
        )

        with console_lock:
            for c in candidates:
                idx = c.get("_index", "?")
                title = (c.get("title") or "")[:60]
                channel = (c.get("channel") or c.get("uploader") or "")[:30]
                dur = c.get("duration")
                dur_str = format_duration(int(dur)) if dur else "??:??"
                score = c.get("_fuzzy_score", 0)
                verdict = "ACCEPT" if c.get("_accepted") else "REJECT"
                reason = c.get("_reason", "")
                console.print(
                    f"  [dim][{idx}] {title} | {channel} | {dur_str} | "
                    f"score={score} | {verdict}: {reason}[/dim]"
                )

        if found:
            best_result = found
            all_candidates = candidates
            chosen_source = source
            break

    if best_result is None:
        with console_lock:
            console.print(
                Panel(
                    f"[red]No valid result for: [bold]{artist} — {song}[/bold][/red]\n"
                    f"Sources tried: {', '.join(args.sources)}\n"
                    f"Duration: [{format_duration(args.min_duration)}–"
                    f"{format_duration(args.max_duration)}] | "
                    f"Fuzzy threshold: {args.fuzzy_threshold}",
                    title="[bold red] Search Failed[/bold red]",
                    border_style="red",
                )
            )
        result["reason"] = "No valid result found after all sources"
        _persist(state, state_lock, key, "failed", None, None, None, output_dir)
        return result

    url: str = best_result.get("webpage_url") or best_result.get("url", "")
    matched_title: str = best_result.get("title") or ""
    fuzzy_score: int = best_result.get("_fuzzy_score", 0)
    duration_s: int = int(best_result.get("duration") or 0)
    thumbnail_url: Optional[str] = best_result.get("thumbnail")

    result.update(
        source=chosen_source,
        url=url,
        matched_title=matched_title,
        fuzzy_score=fuzzy_score,
        duration_seconds=duration_s,
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
                        f"[bold]Title:[/bold]    {matched_title}\n"
                        f"[bold]Channel:[/bold]  "
                        f"{best_result.get('channel') or best_result.get('uploader','')}\n"
                        f"[bold]Duration:[/bold] {format_duration(duration_s)}\n"
                        f"[bold]Score:[/bold]    {fuzzy_score}\n"
                        f"[bold]URL:[/bold]      {url}\n"
                        f"[bold]Thumb:[/bold]    {thumbnail_url or 'N/A'}",
                        title=f"[bold cyan] {artist} — {song}[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print(
                    "[bold yellow]\\[Y][/bold yellow] Download  "
                    "[bold yellow]\\[n][/bold yellow] Skip  "
                    "[bold yellow]\\[s][/bold yellow] Try next result  "
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
                    f"[blue] Retry {attempt}/{config.RETRY_ATTEMPTS}: "
                    f"{song} — {last_error} (waiting {wait:.1f}s)[/blue]"
                )
            time.sleep(wait)

    progress.remove_task(task_id)

    if downloaded_file is None or not downloaded_file.exists():
        with console_lock:
            console.print(
                Panel(
                    f"[red][bold]{artist} — {song}[/bold][/red]\n{last_error}",
                    title="[bold red] Download Failed[/bold red]",
                    border_style="red",
                )
            )
        result["reason"] = last_error
        _persist(state, state_lock, key, "failed", url, None, None, output_dir)
        return result

    mb_data: Optional[dict] = None
    mb_enriched = False

    if args.musicbrainz:
        mb_data = fetch_musicbrainz(artist, song)
        if mb_data:
            mb_enriched = True
            with console_lock:
                console.print(
                    f"[green] MusicBrainz: ENRICHED — "
                    f"album={mb_data.get('album') or 'N/A'}, "
                    f"year={mb_data.get('year') or 'N/A'}, "
                    f"genre={mb_data.get('genre') or 'N/A'}[/green]"
                )
        else:
            with console_lock:
                console.print(
                    "[yellow] MusicBrainz: FALLBACK — no match found[/yellow]"
                )

    extra: dict[str, Any] = {
        "source_url": url,
        "album":     mb_data.get("album")     if mb_data else None,
        "year":      mb_data.get("year")      if mb_data else None,
        "genre":     mb_data.get("genre")     if mb_data else None,
        "track_num": mb_data.get("track_num") if mb_data else None,
        "mb_id":     mb_data.get("mb_id")     if mb_data else None,
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
                    "File deleted — marked as failed.",
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
    )

    _persist(
        state, state_lock, key, "downloaded",
        url, str(downloaded_file), md5, output_dir,
    )

    with console_lock:
        console.print(
            f"[green] Downloaded: {artist} — {song} "
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
    console.print(
        Panel(
            f"[bold]Output dir:[/bold]    {args.output.resolve()}\n"
            f"[bold]Format:[/bold]        {args.format} @ {args.quality} kbps\n"
            f"[bold]Workers:[/bold]       {args.workers}\n"
            f"[bold]Total songs:[/bold]   {total}\n"
            f"[bold]Sources:[/bold]       {' → '.join(args.sources)}\n"
            f"[bold]MusicBrainz:[/bold]   {' enabled' if args.musicbrainz else ' disabled'}\n"
            f"[bold]Dry run:[/bold]       {' yes' if args.dry_run else ' no'}\n"
            f"[bold]Interactive:[/bold]   {' yes' if args.interactive else ' no'}\n"
            f"[bold]Skip existing:[/bold] {' yes' if args.skip_existing else ' no'}",
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
    skipped    = [r for r in results if r.get("status") == "skipped"]
    failed     = [r for r in results if r.get("status") == "failed"]
    total_bytes = sum(r.get("file_size_bytes", 0) for r in downloaded)

    tbl = Table(
        title="[bold] Download Summary[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        caption=(
            f" {len(downloaded)} downloaded  "
            f"  {len(skipped)} skipped  "
            f" {len(failed)} failed  |  "
            f" {elapsed:.1f}s  |   {format_size(total_bytes)}"
        ),
    )
    tbl.add_column("#",            style="dim",     width=4)
    tbl.add_column("Artist",       style="cyan",    min_width=14)
    tbl.add_column("Song",         style="white",   min_width=18)
    tbl.add_column("Status",                        min_width=14)
    tbl.add_column("Duration",     style="yellow",  width=10)
    tbl.add_column("Fuzzy",        style="magenta", width=6)
    tbl.add_column("MusicBrainz",  style="blue",    width=12)
    tbl.add_column("Size",         style="green",   width=9)
    tbl.add_column("File / Reason",style="dim",     min_width=28)

    for i, r in enumerate(results, 1):
        status = r.get("status", "unknown")
        if status == "downloaded":
            status_cell = "[green] downloaded[/green]"
        elif status == "skipped":
            status_cell = "[yellow]  skipped[/yellow]"
        else:
            status_cell = "[red] failed[/red]"

        dur = r.get("duration_seconds") or 0
        mb  = "" if r.get("musicbrainz_enriched") else "—"
        sz  = format_size(r["file_size_bytes"]) if r.get("file_size_bytes") else "—"
        detail = (r.get("file_path") or r.get("reason") or "—")[:55]

        tbl.add_row(
            str(i),
            r.get("artist", ""),
            r.get("song", ""),
            status_cell,
            format_duration(int(dur)) if dur else "—",
            str(r.get("fuzzy_score", 0)),
            mb, sz, detail,
        )

    console.print(tbl)


def _dry_run_table(
    console: Console,
    pairs: list[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    """Search all sources in dry-run mode and render a preview table."""
    tbl = Table(
        title="[bold cyan] Dry Run Preview[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
    )
    tbl.add_column("Artist",        style="cyan",    min_width=14)
    tbl.add_column("Song",          style="white",   min_width=18)
    tbl.add_column("Source",        style="dim",     width=12)
    tbl.add_column("Matched Title", style="green",   min_width=30)
    tbl.add_column("Channel",       style="dim",     min_width=18)
    tbl.add_column("Duration",      style="yellow",  width=10)
    tbl.add_column("Fuzzy",         style="magenta", width=6)

    opts: dict[str, Any] = {
        "max_results": args.max_results,
        "cookies_browser": args.cookies_browser,
        "proxy": args.proxy,
    }

    for artist, song in pairs:
        found: Optional[dict] = None
        src_used: Optional[str] = None
        for source in args.sources:
            raw = search_source(build_search_query(artist, song, source), source, opts)
            if raw:
                found, _ = filter_results(
                    raw, artist, song,
                    args.max_duration, args.min_duration, args.fuzzy_threshold,
                )
                if found:
                    src_used = source
                    break

        if found:
            dur = int(found.get("duration") or 0)
            tbl.add_row(
                artist, song,
                src_used or "—",
                (found.get("title") or "")[:48],
                (found.get("channel") or found.get("uploader") or "")[:26],
                format_duration(dur),
                str(found.get("_fuzzy_score", 0)),
            )
        else:
            tbl.add_row(
                artist, song, "—", "[red]No match found[/red]", "—", "—", "—"
            )

    console.print(tbl)


def main() -> None:
    """Orchestrate the full download session."""
    args   = parse_args()
    config = Config()

    log_file_handle = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = args.log_file.open("w", encoding="utf-8")
        console = Console(record=True, file=log_file_handle)
    else:
        console = Console()

    check_ffmpeg(console)

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
            console.print(f"[yellow] Skipping '{artist}': empty song list[/yellow]")

    pairs: list[tuple[str, str]] = [
        (artist, song)
        for artist, lst in songs.items()
        for song in lst
        if lst
    ]
    total = len(pairs)

    args.output.mkdir(parents=True, exist_ok=True)
    state         = load_state(args.output)
    state_lock    = threading.Lock()
    console_lock  = threading.Lock()
    stop_event    = threading.Event()
    interactive_lock: Optional[threading.Lock] = (
        threading.Lock() if args.interactive else None
    )
    printed_artists: set[str] = set()
    printed_artists_lock = threading.Lock()

    _print_startup_banner(console, args, total)

    if args.dry_run:
        _dry_run_table(console, pairs, args)
        if log_file_handle:
            log_file_handle.close()
        return

    artists_seen: set[str] = set()

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

                    with console_lock:
                        if artist not in artists_seen:
                            artists_seen.add(artist)
                            song_count = sum(1 for a, _ in pairs if a == artist)
                            console.print(
                                Rule(
                                    f"[bold cyan]{artist}[/bold cyan] "
                                    f"[dim]({song_count} songs)[/dim]"
                                )
                            )

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
                                    title=f"[bold red] Unhandled: {artist} — {song}[/bold red]",
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
                f"[yellow]  Interrupted by user after {elapsed:.1f}s\n"
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
            f"[green] Reports saved to: {args.output.resolve()}[/green]"
        )

    if args.update_json and args.file:
        console.print(
            f"[bold yellow] About to overwrite input file: {args.file}[/bold yellow]"
        )
        update_json_file(args.file, all_results)
        console.print(f"[green] Input JSON updated: {args.file}[/green]")

    if log_file_handle:
        log_file_handle.close()


if __name__ == "__main__":
    main()