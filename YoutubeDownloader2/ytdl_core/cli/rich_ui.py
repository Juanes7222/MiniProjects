"""
Rich terminal UI — the ``RichEvents`` implementation of ``DownloaderEvents``.

One instance is shared across all worker threads; every ``console.print()``
call is protected by ``self._lock``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

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

from ..config import Config
from ..events import DownloaderEvents
from ..result import DownloadResult
from ..utils import format_duration, format_size


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

        self._buffer: list[str] = []
        self._buffering = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _k(self, artist: str, song: str) -> str:
        return f"{artist}::{song}"

    def suspend_progress(self) -> None:
        if self._progress:
            with self._lock:
                self._progress.stop()

    def resume_progress(self) -> None:
        if self._progress:
            with self._lock:
                self._progress.start()

    def _print(self, *args, **kwargs) -> None:
        with self._lock:
            if self._buffering:
                self._buffer.append((args, kwargs))
            else:
                self.console.print(*args, **kwargs)

    def start_buffering(self) -> None:
        with self._lock:
            self._buffering = True

    def stop_buffering(self) -> None:
        with self._lock:
            self._buffering = False

    def flush_buffer(self) -> None:
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            was_buffering = self._buffering
            self._buffering = False
        for args, kwargs in items:
            try:
                self.console.print(*args, **kwargs)
            except Exception:
                pass
        if was_buffering:
            with self._lock:
                self._buffering = True

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def on_session_start(self, total: int, is_verify: bool = False) -> None:
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ]
        if not is_verify:
            columns.extend([DownloadColumn(), TransferSpeedColumn()])

        self._progress = Progress(*columns, console=self.console, transient=False)
        self._overall_task = self._progress.add_task(
            "[cyan]Verifying Library...[/cyan]" if is_verify else "[cyan]Processing Batch...[/cyan]",
            total=total,
            visible=True,
        )
        self._progress.start()

    def on_session_complete(self, results: list[DownloadResult], elapsed: float) -> None:
        if self._progress:
            self._progress.stop()
        self._print_summary(results, elapsed)

    def on_interrupted(self, completed: int, total: int, elapsed: float) -> None:
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

    # ------------------------------------------------------------------
    # Artist / search events
    # ------------------------------------------------------------------

    def on_artist_start(self, artist: str, song_count: int) -> None:
        from rich.markup import escape

        self._print(
            Rule(
                f"[bold cyan]{escape(str(artist))}[/bold cyan] "
                f"[dim]({song_count} songs)[/dim]"
            )
        )

    def on_search_start(self, artist: str, song: str, source: str) -> None:
        # Suppress in select mode — the candidates table is sufficient
        if self._buffering:
            return
        from rich.markup import escape

        self._print(
            f"[dim]  [{escape(str(source))}] {escape(str(song))} -- "
            f"{escape(str(artist))}[/dim]"
        )

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

        best_idx = 0 if ranked and ranked[0][1] >= self.score_threshold else None

        for i, (entry, sc, bd) in enumerate(ranked):
            dur = int(entry.get("duration") or 0)
            title_str = (entry.get("title") or "")[:55]
            channel_str = (entry.get("channel") or entry.get("uploader") or "")[:30]
            top = sorted(bd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            signals = ", ".join(f"{'+' if v >= 0 else ''}{v} {k}" for k, v in top)
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

    def on_search_failed(self, artist: str, song: str, sources_tried: list[str]) -> None:
        self._print(
            Panel(
                f"[red]No valid result for: [bold]{artist} -- {song}[/bold][/red]\n"
                f"Sources tried: {', '.join(sources_tried)}\n"
                f"Score threshold: {self.score_threshold}",
                title="[bold red] Search Failed[/bold red]",
                border_style="red",
            )
        )

    # ------------------------------------------------------------------
    # Verification / fingerprint events
    # ------------------------------------------------------------------

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

    def on_fingerprint_start(self, artist: str, song: str, seconds: int) -> None:
        self._print(f"[dim]  Verifying audio fingerprint (downloading {seconds}s)...[/dim]")

    def on_fingerprint_partial_failed(self, artist: str, song: str) -> None:
        self._print("[yellow]  Fingerprint: partial download failed, proceeding anyway[/yellow]")

    def on_fingerprint_result(
        self,
        artist: str,
        song: str,
        verified: bool,
        confidence: float,
        matched_title: str,
    ) -> None:
        if verified:
            self._print(f"[green]  Fingerprint: verified ({confidence:.0%} confidence)[/green]")

    def on_fingerprint_low_confidence(self, artist: str, song: str, matched_title: str) -> None:
        self._print(
            f"[yellow]  Fingerprint: low confidence match "
            f"({matched_title}), trying next result[/yellow]"
        )

    def on_fingerprint_no_match(self, artist: str, song: str) -> None:
        self._print(
            "[yellow]  Fingerprint: no AcoustID match found, "
            "proceeding with score-based selection[/yellow]"
        )

    def on_fingerprint_error(self, artist: str, song: str, error: str) -> None:
        self._print(f"[yellow]  Fingerprint error ({error}), proceeding anyway[/yellow]")

    # ------------------------------------------------------------------
    # Skip / MD5 events
    # ------------------------------------------------------------------

    def on_skip_existing(self, artist: str, song: str, file_path: Path, md5_ok: bool) -> None:
        label = "exists" if md5_ok else "exists, no MD5"
        self._print(f"[yellow]  Skipped ({label}): {artist} -- {song}[/yellow]")

    def on_md5_mismatch(self, artist: str, song: str) -> None:
        self._print(f"[blue]  MD5 mismatch for '{song}' -- re-downloading...[/blue]")

    # ------------------------------------------------------------------
    # Download events
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Post-download check events
    # ------------------------------------------------------------------

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
            self._print(f"[yellow]  Warning: {silence_ratio:.1%} silence in audio[/yellow]")

    def on_silence_rejected(self, artist: str, song: str, silence_ratio: float) -> None:
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
        silence_checked: bool = True,
    ) -> None:
        dur_label = (
            f"DURATION: {format_duration(actual_dur)} matches"
            if dur_ok
            else f"DURATION: mismatch {format_duration(actual_dur)}"
        )
        if not silence_checked:
            sil_label = "SILENCE: check disabled"
        elif silence_ratio <= 0.0:
            sil_label = "SILENCE: 0.0% -- normal"
        elif silence_ratio < 0.15:
            sil_label = f"SILENCE: {silence_ratio:.1%} -- normal"
        elif silence_ratio <= 0.30:
            sil_label = f"SILENCE: {silence_ratio:.1%} -- elevated"
        else:
            sil_label = f"SILENCE: {silence_ratio:.1%} -- excessive"

        self._print(f"  Post-check: [{dur_label}] [{sil_label}]")

    # ------------------------------------------------------------------
    # MusicBrainz / metadata events
    # ------------------------------------------------------------------

    def on_musicbrainz_result(self, artist: str, song: str, enriched: bool, data: dict) -> None:
        if enriched:
            self._print(
                f"[green]  MusicBrainz: ENRICHED -- "
                f"album={data.get('album') or 'N/A'}, "
                f"year={data.get('year') or 'N/A'}, "
                f"genre={data.get('genre') or 'N/A'}[/green]"
            )
        else:
            self._print("[yellow]  MusicBrainz: FALLBACK -- no match found[/yellow]")

    def on_metadata_error(self, artist: str, song: str, file_name: str) -> None:
        self._print(
            Panel(
                f"[red]Integrity check failed for [bold]{file_name}[/bold].\n"
                "File deleted -- marked as failed.",
                title="[bold red] Metadata Error[/bold red]",
                border_style="red",
            )
        )

    # ------------------------------------------------------------------
    # Generic channels
    # ------------------------------------------------------------------

    def on_warn(self, message: str) -> None:
        self._print(message)

    def on_info(self, message: str) -> None:
        self._print(message)

    # ------------------------------------------------------------------
    # Result hook
    # ------------------------------------------------------------------

    def on_result(self, result: DownloadResult) -> None:
        self._remove_task(result.artist, result.song)
        if self._progress and getattr(self, "_overall_task", None) is not None:
            self._progress.advance(self._overall_task)

        from rich.markup import escape

        safe_artist = escape(str(result.artist))
        safe_song = escape(str(result.song))
        safe_reason = escape(str(result.reason))

        if result.status == "downloaded":
            self._print(
                f"[green]  Downloaded: {safe_artist} -- {safe_song} "
                f"({format_duration(result.duration_seconds or 0)}, "
                f"{format_size(result.file_size_bytes or 0)})[/green]"
            )
        elif result.status == "verified":
            self._print(
                f"[green]  Verified: {safe_artist} -- {safe_song} "
                f"({format_duration(result.duration_seconds or 0)}, "
                f"{format_size(result.file_size_bytes or 0)})[/green]"
            )
        elif result.status == "failed":
            self._print(
                f"[red]  Failed: {safe_artist} -- {safe_song} | {safe_reason}[/red]"
            )
        elif result.status == "skipped":
            if "exists" not in str(result.reason).lower():
                self._print(
                    f"[yellow]  Skipped: {safe_artist} -- {safe_song} | {safe_reason}[/yellow]"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _print_summary(self, results: list[DownloadResult], elapsed: float) -> None:
        downloaded = [r for r in results if r.status in ("downloaded", "verified")]
        skipped = [r for r in results if r.status == "skipped"]
        failed = [r for r in results if r.status == "failed"]
        total_bytes = sum(r.file_size_bytes for r in downloaded)

        tbl = Table(
            title="[bold] Summary[/bold]",
            box=box.ROUNDED,
            caption=(
                f" {len(downloaded)} success "
                f" {len(skipped)} skipped "
                f" {len(failed)} failed | "
                f" {elapsed:.1f}s | {format_size(total_bytes)}"
            ),
        )
        tbl.add_column("#", style="dim", width=4)
        tbl.add_column("Artist", style="cyan", overflow="ellipsis", ratio=2)
        tbl.add_column("Song", style="white", overflow="ellipsis", ratio=3)
        tbl.add_column("Status", overflow="ellipsis")
        tbl.add_column("Duration", style="yellow", width=9)
        tbl.add_column("Fuzzy", style="magenta", width=5)
        tbl.add_column("Score", width=6)
        tbl.add_column("Fingerprint", overflow="ellipsis", ratio=2)
        tbl.add_column("Silence", width=8)
        tbl.add_column("MusicBrainz", style="blue", width=8)
        tbl.add_column("Size", style="green", width=8)
        tbl.add_column("File / Reason", style="dim", overflow="ellipsis", ratio=2)

        for i, r in enumerate(results, 1):
            if r.status == "downloaded":
                status_cell = "[green] downloaded[/green]"
            elif r.status == "verified":
                status_cell = "[green] verified[/green]"
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

            fl = r.fingerprint_label
            if r.fingerprint_verified:
                fp_cell = f"[green]verified {r.fingerprint_confidence:.0%}[/green]"
            elif fl and "disabled" not in fl:
                fp_cell = f"[yellow]{fl[:14]}[/yellow]"
            elif fl:
                fp_cell = "[dim]disabled[/dim]"
            else:
                fp_cell = "[dim]--[/dim]"

            sil = r.silence_ratio
            if sil <= 0.0:
                sil_cell = "[dim]--[/dim]"
            elif sil < 0.15:
                sil_cell = f"[green]{sil:.1%}[/green]"
            elif sil < 0.30:
                sil_cell = f"[yellow]{sil:.1%}[/yellow]"
            else:
                sil_cell = f"[red]{sil:.1%}[/red]"

            mb = "yes" if r.musicbrainz_enriched else "--"
            sz = format_size(r.file_size_bytes) if r.file_size_bytes else "--"
            detail = (str(r.file_path) if r.file_path else (r.reason or "--"))[:55]
            dur = r.duration_seconds or 0

            from rich.markup import escape

            row = [
                str(i),
                escape(str(r.artist)),
                escape(str(r.song)),
                status_cell,
                format_duration(int(dur)) if dur else "--",
                str(r.fuzzy_score),
                score_cell,
                fp_cell,
                sil_cell,
                mb,
                sz,
                escape(str(detail)),
            ]
            tbl.add_row(*row)

        self.console.print(tbl)
