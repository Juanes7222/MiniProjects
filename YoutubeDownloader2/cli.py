"""
ytdl_core CLI — Rich terminal interface for MusicDownloader.

This module is the ONLY place where Rich, argparse, and sys.exit() appear.
It implements RichEvents(DownloaderEvents) to translate every library callback
into formatted terminal output, then wires everything together in main().

Entry point (after `pip install ytdl-core[cli]`):
    ytdl --file songs.json --quality 320 --musicbrainz
    python -m ytdl_core.cli --file songs.json --acoustid-key KEY
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

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

from config import Config
from core import MusicDownloader
from events import DownloaderEvents
from result import DownloadResult
from search import build_search_query, search_source, select_best_result
from utils import check_ffmpeg, format_duration, format_size, sanitize_filename

_CONFIG = Config()


class RichEvents(DownloaderEvents):
    """
    Full Rich implementation of DownloaderEvents.

    One instance is shared across all worker threads; every console.print()
    is protected by self._lock.
    """

    def __init__(self, console: Console, score_threshold: int, config: Config) -> None:
        self.console = console
        self.score_threshold = score_threshold
        self.config = config
        self._lock = threading.Lock()

        self._progress: Optional[Progress] = None
        self._tasks: dict[str, TaskID] = {} 
        self._tasks_lock = threading.Lock()


    def _k(self, artist: str, song: str) -> str:
        return f"{artist}::{song}"

    def _print(self, *args, **kwargs) -> None:
        with self._lock:
            self.console.print(*args, **kwargs)


    def on_session_start(self, total: int) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=self.console,
            transient=False,
        )
        self._progress.start()

    def on_session_complete(
        self, results: list[DownloadResult], elapsed: float
    ) -> None:
        if self._progress:
            self._progress.stop()
        self._print_summary(results, elapsed)

    def on_interrupted(
        self, completed: int, total: int, elapsed: float
    ) -> None:
        if self._progress:
            self._progress.stop()
        self._print(
            Panel(
                f"[yellow]Interrupted by user after {elapsed:.1f}s\n"
                f"Completed: {completed}/{total} songs[/yellow]",
                title="[bold yellow] Interrupted[/bold yellow]",
                border_style="yellow",
            )
        )


    def on_artist_start(self, artist: str, song_count: int) -> None:
        self._print(
            Rule(
                f"[bold cyan]{artist}[/bold cyan] [dim]({song_count} songs)[/dim]"
            )
        )


    def on_search_start(self, artist: str, song: str, source: str) -> None:
        self._print(f"[dim]  [{source}] {song} -- {artist}[/dim]")

    def on_no_results(self, artist: str, song: str, source: str) -> None:
        self._print(f"[dim]  no results from {source}[/dim]")

    def on_candidates_scored(
        self,
        artist: str,
        song: str,
        ranked: list[tuple[dict, int, dict]],
    ) -> None:
        if not ranked:
            return

        tbl = Table(
            title=f"Candidates for: {artist} -- {song}",
            box=box.SIMPLE,
            show_lines=False,
            expand=False,
        )
        tbl.add_column("#", width=3, style="dim")
        tbl.add_column("Title", max_width=55)
        tbl.add_column("Channel", max_width=30)
        tbl.add_column("Duration", width=10, style="yellow")
        tbl.add_column("Score", width=7)
        tbl.add_column("Top signals", min_width=30, style="dim")

        best_idx = (
            0 if ranked and ranked[0][1] >= self.score_threshold else None
        )

        for i, (entry, sc, bd) in enumerate(ranked):
            dur = int(entry.get("duration") or 0)
            title_str = (entry.get("title") or "")[:55]
            channel_str = (
                entry.get("channel") or entry.get("uploader") or ""
            )[:30]
            top = sorted(bd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            signals = ", ".join(
                f"{'+' if v >= 0 else ''}{v} {k}" for k, v in top
            )
            if sc >= 70:
                score_cell = f"[green]{sc}[/green]"
            elif sc >= 30:
                score_cell = f"[yellow]{sc}[/yellow]"
            else:
                score_cell = f"[red]{sc}[/red]"

            prefix = ">" if i == best_idx else " "
            tbl.add_row(
                f"{prefix}{i + 1}",
                title_str,
                channel_str,
                format_duration(dur),
                score_cell,
                signals,
            )

        self._print(tbl)

    def on_search_failed(
        self, artist: str, song: str, sources_tried: list[str]
    ) -> None:
        self._print(
            Panel(
                f"[red]No valid result for: [bold]{artist} -- {song}[/bold][/red]\n"
                f"Sources tried: {', '.join(sources_tried)}\n"
                f"Score threshold: {self.score_threshold}",
                title="[bold red] Search Failed[/bold red]",
                border_style="red",
            )
        )


    def on_verification_status(
        self,
        artist: str,
        song: str,
        score: int,
        score_label: str,
        fp_label: str,
    ) -> None:
        if score >= self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT:
            color = "green"
        elif score >= self.config.SCORE_THRESHOLD_REJECT:
            color = "yellow"
        else:
            color = "red"
        self._print(
            f"  Verification: [{color}][SCORE: {score} {score_label}][/{color}] "
            f"[FINGERPRINT: {fp_label}]"
        )


    def on_fingerprint_start(
        self, artist: str, song: str, seconds: int
    ) -> None:
        self._print(
            f"[dim]  Verifying audio fingerprint (downloading {seconds}s)...[/dim]"
        )

    def on_fingerprint_partial_failed(
        self, artist: str, song: str
    ) -> None:
        self._print(
            "[yellow]  Fingerprint: partial download failed, "
            "proceeding anyway[/yellow]"
        )

    def on_fingerprint_result(
        self,
        artist: str,
        song: str,
        verified: bool,
        confidence: float,
        matched_title: str,
    ) -> None:
        if verified:
            self._print(
                f"[green]  Fingerprint: verified "
                f"({confidence:.0%} confidence)[/green]"
            )

    def on_fingerprint_low_confidence(
        self, artist: str, song: str, matched_title: str
    ) -> None:
        self._print(
            f"[yellow]  Fingerprint: low confidence match "
            f"({matched_title}), trying next result[/yellow]"
        )

    def on_fingerprint_no_match(self, artist: str, song: str) -> None:
        self._print(
            "[yellow]  Fingerprint: no AcoustID match found, "
            "proceeding with score-based selection[/yellow]"
        )

    def on_fingerprint_error(
        self, artist: str, song: str, error: str
    ) -> None:
        self._print(
            f"[yellow]  Fingerprint error ({error}), "
            "proceeding anyway[/yellow]"
        )


    def on_skip_existing(
        self, artist: str, song: str, file_path: Path, md5_ok: bool
    ) -> None:
        label = "exists" if md5_ok else "exists, no MD5"
        self._print(
            f"[yellow]  Skipped ({label}): {artist} -- {song}[/yellow]"
        )

    def on_md5_mismatch(self, artist: str, song: str) -> None:
        self._print(
            f"[blue]  MD5 mismatch for '{song}' -- re-downloading...[/blue]"
        )

    def on_download_start(self, artist: str, song: str, url: str) -> None:
        if self._progress:
            task_id = self._progress.add_task(
                f"[cyan]{song[:45]}[/cyan]", total=100, visible=True
            )
            with self._tasks_lock:
                self._tasks[self._k(artist, song)] = task_id

    def on_download_progress(
        self,
        artist: str,
        song: str,
        percent: float,
        speed_bps: float,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        if self._progress:
            with self._tasks_lock:
                task_id = self._tasks.get(self._k(artist, song))
            if task_id is not None:
                self._progress.update(task_id, completed=percent)

    def on_download_retry(
        self,
        artist: str,
        song: str,
        attempt: int,
        max_attempts: int,
        error: str,
        wait_seconds: float,
    ) -> None:
        self._print(
            f"[blue]  Retry {attempt}/{max_attempts}: "
            f"{song} -- {error} (waiting {wait_seconds:.1f}s)[/blue]"
        )

    def on_download_failed(self, artist: str, song: str, error: str) -> None:
        self._remove_task(artist, song)
        self._print(
            Panel(
                f"[red][bold]{artist} -- {song}[/bold][/red]\n{error}",
                title="[bold red] Download Failed[/bold red]",
                border_style="red",
            )
        )

    def on_disk_full(self) -> None:
        self._print(
            Panel(
                "[bold red]Disk full![/bold red]\n"
                "Free space and restart with [bold]--skip-existing[/bold].",
                title="[bold red] Disk Full[/bold red]",
                border_style="red",
            )
        )


    def on_duration_check(
        self,
        artist: str,
        song: str,
        expected_seconds: int,
        actual_seconds: int,
        ok: bool,
    ) -> None:
        if not ok:
            self._print(
                f"[yellow]  Duration mismatch: expected "
                f"{format_duration(expected_seconds)}, "
                f"got {format_duration(actual_seconds)}[/yellow]"
            )

    def on_silence_check(
        self,
        artist: str,
        song: str,
        silence_ratio: float,
        excessive: bool,
    ) -> None:
        if not excessive and silence_ratio > 0.15:
            self._print(
                f"[yellow]  Warning: {silence_ratio:.1%} silence in audio[/yellow]"
            )

    def on_silence_rejected(
        self, artist: str, song: str, silence_ratio: float
    ) -> None:
        self._print(
            Panel(
                f"[red]Rejected: {silence_ratio:.1%} silence detected "
                "(likely copyright block or muted audio)[/red]",
                title=f"[bold red] Silence Check Failed: {artist} -- {song}[/bold red]",
                border_style="red",
            )
        )

    def on_post_check_summary(
        self,
        artist: str,
        song: str,
        dur_ok: bool,
        actual_dur: int,
        silence_ratio: float,
    ) -> None:
        dur_label = (
            f"DURATION: {format_duration(actual_dur)} matches"
            if dur_ok
            else f"DURATION: mismatch {format_duration(actual_dur)}"
        )
        if silence_ratio <= 0.0:
            sil_label = "SILENCE: check disabled"
        elif silence_ratio < 0.15:
            sil_label = f"SILENCE: {silence_ratio:.1%} -- normal"
        elif silence_ratio <= 0.30:
            sil_label = f"SILENCE: {silence_ratio:.1%} -- elevated"
        else:
            sil_label = f"SILENCE: {silence_ratio:.1%} -- excessive"

        self._print(f"  Post-check: [{dur_label}] [{sil_label}]")


    def on_musicbrainz_result(
        self, artist: str, song: str, enriched: bool, data: dict
    ) -> None:
        if enriched:
            self._print(
                f"[green]  MusicBrainz: ENRICHED -- "
                f"album={data.get('album') or 'N/A'}, "
                f"year={data.get('year') or 'N/A'}, "
                f"genre={data.get('genre') or 'N/A'}[/green]"
            )
        else:
            self._print(
                "[yellow]  MusicBrainz: FALLBACK -- no match found[/yellow]"
            )

    def on_metadata_error(
        self, artist: str, song: str, file_name: str
    ) -> None:
        self._print(
            Panel(
                f"[red]Integrity check failed for [bold]{file_name}[/bold].\n"
                "File deleted -- marked as failed.",
                title="[bold red] Metadata Error[/bold red]",
                border_style="red",
            )
        )

    def on_warn(self, message: str) -> None:
        self._print(message)


    def on_result(self, result: DownloadResult) -> None:
        self._remove_task(result.artist, result.song)
        if result.status == "downloaded":
            self._print(
                f"[green]  Downloaded: {result.artist} -- {result.song} "
                f"({format_duration(result.duration_seconds)}, "
                f"{format_size(result.file_size_bytes)})[/green]"
            )


    def _remove_task(self, artist: str, song: str) -> None:
        if not self._progress:
            return
        with self._tasks_lock:
            task_id = self._tasks.pop(self._k(artist, song), None)
        if task_id is not None:
            try:
                self._progress.remove_task(task_id)
            except Exception:
                pass

    def _print_summary(
        self, results: list[DownloadResult], elapsed: float
    ) -> None:
        downloaded = [r for r in results if r.status == "downloaded"]
        skipped    = [r for r in results if r.status == "skipped"]
        failed     = [r for r in results if r.status == "failed"]
        total_bytes = sum(r.file_size_bytes for r in downloaded)

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
        tbl.add_column("#",            style="dim",     width=4)
        tbl.add_column("Artist",       style="cyan",    min_width=14)
        tbl.add_column("Song",         style="white",   min_width=18)
        tbl.add_column("Status",                        min_width=14)
        tbl.add_column("Duration",     style="yellow",  width=10)
        tbl.add_column("Fuzzy",        style="magenta", width=6)
        tbl.add_column("Score",                         width=7)
        tbl.add_column("Fingerprint",                   width=16)
        tbl.add_column("Silence",                       width=10)
        tbl.add_column("MusicBrainz",  style="blue",    width=12)
        tbl.add_column("Size",         style="green",   width=9)
        tbl.add_column("File / Reason",style="dim",     min_width=28)

        for i, r in enumerate(results, 1):
            if r.status == "downloaded":
                status_cell = "[green] downloaded[/green]"
            elif r.status == "skipped":
                status_cell = "[yellow] skipped[/yellow]"
            else:
                status_cell = "[red] failed[/red]"

            sc = r.composite_score
            if sc >= 70:
                score_cell = f"[green]{sc}[/green]"
            elif sc >= 30:
                score_cell = f"[yellow]{sc}[/yellow]"
            else:
                score_cell = f"[red]{sc}[/red]" if sc > 0 else "--"

            if r.fingerprint_verified:
                fp_cell = f"[green]verified {r.fingerprint_confidence:.0%}[/green]"
            elif r.fingerprint_confidence > 0:
                fp_cell = "[yellow]no match[/yellow]"
            else:
                fp_cell = "[dim]-- disabled[/dim]"

            sil = r.silence_ratio
            if sil <= 0.0:
                sil_cell = "[dim]--[/dim]"
            elif sil < 0.15:
                sil_cell = f"[green]{sil:.1%}[/green]"
            elif sil < 0.30:
                sil_cell = f"[yellow]{sil:.1%}[/yellow]"
            else:
                sil_cell = f"[red]{sil:.1%}[/red]"

            mb     = "yes" if r.musicbrainz_enriched else "--"
            sz     = format_size(r.file_size_bytes) if r.file_size_bytes else "--"
            detail = (str(r.file_path) if r.file_path else (r.reason or "--"))[:55]
            dur    = r.duration_seconds or 0

            tbl.add_row(
                str(i), r.artist, r.song, status_cell,
                format_duration(int(dur)) if dur else "--",
                str(r.fuzzy_score), score_cell, fp_cell,
                sil_cell, mb, sz, detail,
            )

        self.console.print(tbl)



def _dry_run_table(
    console: Console,
    pairs: list[tuple[str, str]],
    args: argparse.Namespace,
    config: Config,
) -> None:
    fp_label = (
        "yes"
        if (getattr(args, "acoustid_key", None) and shutil.which("fpcalc"))
        else "no key"
        if not getattr(args, "acoustid_key", None)
        else "fpcalc missing"
    )

    tbl = Table(
        title="[bold cyan] Dry Run Preview[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
    )
    tbl.add_column("Artist",        style="cyan",  min_width=14)
    tbl.add_column("Song",          style="white", min_width=18)
    tbl.add_column("Source",        style="dim",   width=12)
    tbl.add_column("Matched Title", style="green", min_width=30)
    tbl.add_column("Channel",       style="dim",   min_width=18)
    tbl.add_column("Duration",      style="yellow",width=10)
    tbl.add_column("Score",                        width=7)
    tbl.add_column("Top 3 signals", min_width=35,  style="dim")
    tbl.add_column("Fingerprint",                  width=14)

    opts: dict[str, Any] = {
        "max_results":     args.max_results,
        "cookies_browser": args.cookies_browser,
        "proxy":           args.proxy,
    }
    threshold = getattr(args, "score_threshold", config.SCORE_THRESHOLD_REJECT)

    for artist, song in pairs:
        found: Optional[dict] = None
        src_used: Optional[str] = None

        for source in args.sources:
            raw = search_source(
                build_search_query(artist, song, source), source, opts
            )
            if raw:
                found, _ = select_best_result(
                    results=raw, artist=artist, song=song,
                    mb_duration_seconds=None, config=config,
                    console=None, console_lock=None,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration,
                    score_threshold=threshold,
                )
                if found:
                    src_used = source
                    break

        if found:
            dur = int(found.get("duration") or 0)
            sc  = found.get("_composite_score", 0)
            bd  = found.get("_score_breakdown", {})
            top = sorted(bd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            signals = ", ".join(f"{'+' if v >= 0 else ''}{v} {k}" for k, v in top)
            score_cell = (
                f"[green]{sc}[/green]" if sc >= 70
                else f"[yellow]{sc}[/yellow]" if sc >= 30
                else f"[red]{sc}[/red]"
            )
            tbl.add_row(
                artist, song, src_used or "--",
                (found.get("title") or "")[:48],
                (found.get("channel") or found.get("uploader") or "")[:26],
                format_duration(dur), score_cell, signals, fp_label,
            )
        else:
            tbl.add_row(
                artist, song, "--",
                "[red]No match found[/red]",
                "--", "--", "--", "--", fp_label,
            )

    console.print(tbl)



def _make_interactive_confirm(
    console: Console,
    console_lock: threading.Lock,
    interactive_lock: threading.Lock,
    stop_event: threading.Event,
) -> Any:
    def confirm_fn(artist: str, song: str, best_result: dict) -> bool:
        from utils import format_duration as _fd
        dur = int(best_result.get("duration") or 0)
        with interactive_lock:
            with console_lock:
                console.print(
                    Panel(
                        f"[bold]Title:[/bold] {best_result.get('title', '')}\n"
                        f"[bold]Channel:[/bold] "
                        f"{best_result.get('channel') or best_result.get('uploader', '')}\n"
                        f"[bold]Duration:[/bold] {_fd(dur)}\n"
                        f"[bold]Score:[/bold] {best_result.get('_composite_score', 0)}\n"
                        f"[bold]URL:[/bold] "
                        f"{best_result.get('webpage_url') or best_result.get('url', '')}",
                        title=f"[bold cyan]  {artist} -- {song}[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print(
                    "[bold yellow]\\[Y][/bold yellow] Download  "
                    "[bold yellow]\\[n][/bold yellow] Skip  "
                    "[bold yellow]\\[q][/bold yellow] Quit"
                )
            try:
                choice = input("Choice [Y/n/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

        if choice == "q":
            stop_event.set()
            return False
        return choice != "n"

    return confirm_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ytdl",
        description="YT Music Downloader v2.0 -- batch audio download with metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ytdl --file songs.json\n"
            "  ytdl --file songs.json --format mp3 --quality 320 "
            "--workers 3 --musicbrainz --report json\n"
            "  ytdl --file songs.json --acoustid-key KEY --quality 320\n"
            "  ytdl --file songs.json --skip-fingerprint --no-silence-check\n"
            "  ytdl --data '{\"Radiohead\": [\"Creep\"]}' --dry-run"
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", metavar="PATH", type=Path)
    src.add_argument("--data", metavar="JSON_STR")

    p.add_argument("--output",           metavar="DIR",     type=Path,
                   default=Path(_CONFIG.DEFAULT_OUTPUT_DIR))
    p.add_argument("--format",           metavar="FORMAT",
                   choices=_CONFIG.SUPPORTED_FORMATS,
                   default=_CONFIG.DEFAULT_FORMAT)
    p.add_argument("--quality",          metavar="QUALITY",
                   choices=["128", "192", "320"],
                   default=_CONFIG.DEFAULT_QUALITY)
    p.add_argument("--max-results",      metavar="INT",     type=int,
                   default=_CONFIG.DEFAULT_MAX_RESULTS)
    p.add_argument("--max-duration",     metavar="INT",     type=int,
                   default=_CONFIG.MAX_DURATION_SECONDS)
    p.add_argument("--min-duration",     metavar="INT",     type=int,
                   default=_CONFIG.MIN_DURATION_SECONDS)
    p.add_argument("--fuzzy-threshold",  metavar="INT",     type=int,
                   default=_CONFIG.DEFAULT_FUZZY_THRESHOLD)
    p.add_argument("--workers",          metavar="INT",     type=int,
                   default=_CONFIG.DEFAULT_WORKERS)
    p.add_argument("--delay",            metavar="FLOAT",   type=float, nargs=2,
                   default=[_CONFIG.DEFAULT_DELAY_MIN, _CONFIG.DEFAULT_DELAY_MAX])
    p.add_argument("--sources",          metavar="LIST",
                   default=",".join(_CONFIG.DEFAULT_SOURCES))
    p.add_argument("--cookies-browser",  metavar="BROWSER",
                   choices=["chrome", "firefox", "edge", "safari"])
    p.add_argument("--proxy",            metavar="URL")
    p.add_argument("--musicbrainz",      action="store_true")

    p.add_argument("--acoustid-key",     metavar="KEY",     dest="acoustid_key")
    p.add_argument("--skip-fingerprint", action="store_true")
    p.add_argument("--score-threshold",  metavar="INT",     type=int,
                   default=_CONFIG.SCORE_THRESHOLD_REJECT)
    p.add_argument("--no-silence-check", action="store_true")

    p.add_argument("--skip-existing",    action="store_true")
    p.add_argument("--update-json",      action="store_true")
    p.add_argument("--report",           metavar="FORMAT",  action="append",
                   choices=["json", "csv", "m3u"], default=[], dest="report")
    p.add_argument("--dry-run",          action="store_true")
    p.add_argument("--interactive",      action="store_true")
    p.add_argument("--log-file",         metavar="PATH",    type=Path)

    args = p.parse_args()
    args.workers = max(1, min(args.workers, _CONFIG.MAX_WORKERS))
    args.sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    return args


def main() -> None:
    import json

    args = parse_args()
    config = Config()

    log_fh = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = args.log_file.open("w", encoding="utf-8")
        console = Console(record=True, file=log_fh)
    else:
        console = Console()

    check_ffmpeg(console)

    if not shutil.which("fpcalc"):
        console.print(
            Panel(
                "[yellow]fpcalc (Chromaprint) not found. "
                "Audio fingerprint verification is disabled.[/yellow]\n\n"
                "  Ubuntu/Debian : sudo apt install libchromaprint-tools\n"
                "  macOS         : brew install chromaprint\n"
                "  Windows       : https://acoustid.org/chromaprint\n\n"
                "The --acoustid-key flag will be ignored until fpcalc is installed.",
                title="[bold yellow] fpcalc Not Found[/bold yellow]",
                border_style="yellow",
            )
        )

    try:
        if args.file:
            with args.file.open("r", encoding="utf-8") as fh:
                songs: dict = json.load(fh)
        else:
            songs = json.loads(args.data)
    except (json.JSONDecodeError, OSError) as exc:
        console.print(Panel(f"[red]{exc}[/red]",
                            title="[bold red] Input Error[/bold red]",
                            border_style="red"))
        sys.exit(1)

    pairs: list[tuple[str, str]] = [
        (artist, song)
        for artist, lst in songs.items()
        for song in (lst or [])
    ]
    total = len(pairs)

    acoustid_status = (
        "enabled (fpcalc found)"
        if (args.acoustid_key and shutil.which("fpcalc"))
        else "disabled"
        if not args.acoustid_key
        else "key provided but fpcalc not found"
    )
    console.print(
        Panel(
            f"[bold]Output dir:[/bold] {args.output.resolve()}\n"
            f"[bold]Format:[/bold] {args.format} @ {args.quality} kbps\n"
            f"[bold]Workers:[/bold] {args.workers} | "
            f"[bold]Total songs:[/bold] {total}\n"
            f"[bold]Sources:[/bold] {' -> '.join(args.sources)}\n"
            f"[bold]MusicBrainz:[/bold] {'enabled' if args.musicbrainz else 'disabled'} | "
            f"[bold]Dry run:[/bold] {'yes' if args.dry_run else 'no'}\n"
            f"[bold]AcoustID:[/bold] {acoustid_status}\n"
            f"[bold]Silence check:[/bold] "
            f"{'disabled' if args.no_silence_check else 'enabled'} | "
            f"[bold]Score threshold:[/bold] {args.score_threshold}",
            title="[bold cyan] YT Music Downloader v2.0[/bold cyan]",
            border_style="cyan",
        )
    )

    if args.dry_run:
        _dry_run_table(console, pairs, args, config)
        if log_fh:
            log_fh.close()
        return

    events = RichEvents(console, args.score_threshold, config)

    confirm_fn = None
    if args.interactive:
        stop_event = threading.Event()
        confirm_fn = _make_interactive_confirm(
            console,
            threading.Lock(),
            threading.Lock(),
            stop_event,
        )

    import musicbrainzngs as _mbz
    if args.musicbrainz:
        _mbz.set_useragent(
            "YTMusicDownloader", "2.0",
            "https://github.com/example/yt-music-downloader",
        )

    dl = MusicDownloader(
        config=config,
        events=events,
        acoustid_key=args.acoustid_key,
        skip_fingerprint=args.skip_fingerprint,
        no_silence_check=args.no_silence_check,
        score_threshold=args.score_threshold,
        sources=args.sources,
        workers=args.workers,
        delay=tuple(args.delay),
        max_results=args.max_results,
        fuzzy_threshold=args.fuzzy_threshold,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        musicbrainz=args.musicbrainz,
        cookies_browser=args.cookies_browser,
        proxy=args.proxy,
    )

    results = dl.download_batch(
        songs=songs,
        output_dir=args.output,
        fmt=args.format,
        quality=args.quality,
        skip_existing=args.skip_existing,
        report_formats=args.report or None,
        update_json_path=args.file if args.update_json else None,
    )

    if args.report:
        console.print(
            f"[green]  Reports saved to: {args.output.resolve()}[/green]"
        )

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()