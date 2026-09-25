from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

from ..utils import format_duration


def print_candidate_table(
    scored: list[tuple[dict, int, dict]],
    artist: str,
    song: str,
    console: Console,
    reject_threshold: int,
) -> None:
    table = Table(title=f"Candidates for: {artist} -- {song}", box=box.SIMPLE)
    table.add_column("#", width=3, style="dim")
    table.add_column("Title", max_width=55)
    table.add_column("Channel", max_width=30)
    table.add_column("Duration", width=10, style="yellow")
    table.add_column("Score", width=7)
    table.add_column("Top signals", min_width=30, style="dim")

    best_index = 0 if scored and scored[0][1] >= reject_threshold else None

    for index, (entry, score, breakdown) in enumerate(scored):
        duration = int(entry.get("duration") or 0)
        top_signals = sorted(breakdown.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        signals = ", ".join(
            f"{'+' if value >= 0 else ''}{value} {name}" for name, value in top_signals
        )
        score_markup = (
            f"[green]{score}[/green]"
            if score >= 70
            else f"[yellow]{score}[/yellow]"
            if score >= 30
            else f"[red]{score}[/red]"
        )
        table.add_row(
            f"{'>' if index == best_index else ' '}{index + 1}",
            (entry.get("title") or "")[:55],
            (entry.get("channel") or entry.get("uploader") or "")[:30],
            format_duration(duration),
            score_markup,
            signals,
        )

    console.print(table)
