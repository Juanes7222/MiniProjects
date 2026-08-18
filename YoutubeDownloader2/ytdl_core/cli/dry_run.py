"""
Dry-run preview table for the ytdl CLI.

Renders a Rich table showing what *would* be downloaded without actually
downloading anything.
"""

from __future__ import annotations

import shutil
from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.table import Table

from ..config import Config
from ..search import build_search_query, search_source, select_best_result
from ..utils import format_duration


def dry_run_table(
    console: Console,
    pairs: list[tuple[str, str]],
    args: Any,
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
    tbl.add_column("Artist", style="cyan", min_width=14)
    tbl.add_column("Song", style="white", min_width=18)
    tbl.add_column("Source", style="dim", width=12)
    tbl.add_column("Matched Title", style="green", min_width=30)
    tbl.add_column("Channel", style="dim", min_width=18)
    tbl.add_column("Duration", style="yellow", width=10)
    tbl.add_column("Score", width=7)
    tbl.add_column("Top 3 signals", min_width=35, style="dim")
    tbl.add_column("Fingerprint", width=14)

    opts: dict[str, Any] = {
        "max_results": args.max_results,
        "cookies_browser": getattr(args, "cookies_browser", None),
        "cookies_file": getattr(args, "cookies", None),
        "proxy": args.proxy,
    }
    threshold = getattr(args, "score_threshold", config.SCORE_THRESHOLD_REJECT)

    for artist, song in pairs:
        found: Optional[dict] = None
        src_used: Optional[str] = None

        for source in args.sources:
            raw = search_source(build_search_query(artist, song, source), source, opts)
            if raw:
                found, _ = select_best_result(
                    results=raw,
                    artist=artist,
                    song=song,
                    mb_duration_seconds=None,
                    config=config,
                    console=None,
                    console_lock=None,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration,
                    score_threshold=threshold,
                )
                if found:
                    src_used = source
                    break

        if found:
            dur = int(found.get("duration") or 0)
            sc = found.get("_composite_score", 0)
            bd = found.get("_score_breakdown", {})
            top = sorted(bd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            signals = ", ".join(f"{'+' if v >= 0 else ''}{v} {k}" for k, v in top)
            score_cell = (
                f"[green]{sc}[/green]"
                if sc >= 70
                else f"[yellow]{sc}[/yellow]"
                if sc >= 30
                else f"[red]{sc}[/red]"
            )
            tbl.add_row(
                artist,
                song,
                src_used or "--",
                (found.get("title") or "")[:48],
                (found.get("channel") or found.get("uploader") or "")[:26],
                format_duration(dur),
                score_cell,
                signals,
                fp_label,
            )
        else:
            tbl.add_row(
                artist,
                song,
                "--",
                "[red]No match found[/red]",
                "--",
                "--",
                "--",
                "--",
                fp_label,
            )

    console.print(tbl)
