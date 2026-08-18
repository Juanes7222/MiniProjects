"""
Summary table rendering for the ytdl CLI.

The ``print_summary`` function builds a Rich table with per-song results
(status, duration, score, fingerprint, silence, MusicBrainz, file path).
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table
from rich.markup import escape

from ..result import DownloadResult
from ..utils import format_duration, format_size


def print_summary(
    console: Console,
    results: list[DownloadResult],
    elapsed: float,
) -> None:
    """Render the final session summary table."""
    downloaded = [r for r in results if r.status in ("downloaded", "verified")]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    total_bytes = sum(r.file_size_bytes for r in downloaded)

    tbl = Table(
        title="[bold] Summary[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        caption=(
            f" {len(downloaded)} success "
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

        mb = "yes" if r.musicbrainz_enriched else "--"
        sz = format_size(r.file_size_bytes) if r.file_size_bytes else "--"
        detail = (str(r.file_path) if r.file_path else (r.reason or "--"))[:55]
        dur = r.duration_seconds or 0

        tbl.add_row(
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
        )

    console.print(tbl)
