from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from ..result import DownloadResult


def print_action_summary(
    console: Console,
    results: list[DownloadResult],
    args: Any,
    queue_stats: Mapping[str, int | str],
    retry_items: list[dict[str, Any]] | None = None,
) -> None:
    successful = sum(result.status in ("downloaded", "verified") for result in results)
    failed = sum(result.status == "failed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    unverified = sum(
        result.status in ("downloaded", "verified") and not result.fingerprint_verified
        for result in results
    )
    lines = [
        f"[green]{successful} completed[/green] | "
        f"[yellow]{failed} failed[/yellow] | "
        f"[dim]{skipped} skipped[/dim] | "
        f"[yellow]{unverified} unverified[/yellow] | "
        f"[cyan]{queue_stats.get('pending', 0)} queued[/cyan]"
    ]

    warning = queue_stats.get("warning")
    if warning:
        lines.append(f"[yellow]{escape(str(warning))}[/yellow]")

    for artist, song, reason in [
        (result.artist, result.song, result.reason or "Unknown error")
        for result in results
        if result.status == "failed"
    ][:3]:
        lines.append(f"[red]x[/red] {escape(artist)} - {escape(song)}: {escape(reason[:120])}")

    stuck = sorted(
        retry_items or [],
        key=lambda item: int(item.get("attempts", 0)),
        reverse=True,
    )[:3]
    for item in stuck:
        reason = escape(str(item.get("reason", "Unknown error"))[:100])
        lines.append(
            f"[yellow]retry {item.get('attempts', 0)}x[/yellow] "
            f"{escape(str(item.get('artist', '')))} - {escape(str(item.get('song', '')))}: {reason}"
        )

    if results and not failed and not unverified:
        lines.extend(("", "[green]Library complete. No pending actions.[/green]"))
    elif not results:
        lines.extend(("", "[dim]No songs processed.[/dim]"))

    border = "green" if results and not failed and not unverified else "yellow"
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Next Actions[/bold]",
            border_style=border,
        )
    )

    commands = _suggested_commands(
        args,
        output_dir=Path(args.output),
        failed=failed,
        unverified=unverified,
    )
    if commands:
        console.print("[bold]Next steps:[/bold]")
        for command in commands:
            console.print(Text(f"  {command}", style="cyan"), soft_wrap=True)


def _suggested_commands(
    args: Any,
    output_dir: Path,
    failed: int,
    unverified: int,
) -> list[str]:
    prefix = f"ytdl --output {_quote(output_dir)}"
    profile = getattr(args, "profile", None)
    profiles_path = getattr(args, "profiles_path", None)
    if profile:
        prefix = f"{prefix} --profile {_quote(profile)}"
        if profiles_path:
            prefix = f"{prefix} --profiles-file {_quote(profiles_path)}"
    if getattr(args, "format", None):
        prefix = f"{prefix} --format {_quote(args.format)}"
    if getattr(args, "quality", None):
        prefix = f"{prefix} --quality {_quote(args.quality)}"
    source = _source_arguments(args)
    commands: list[str] = []

    if getattr(args, "verify", False):
        if source:
            commands.append(f"{prefix}{source} --repair")
    elif failed:
        commands.append(f"{prefix} --retry")
    if unverified and source:
        commands.append(f"{prefix}{source} --review")
    return commands


def _source_arguments(args: Any) -> str:
    file_path = getattr(args, "file", None)
    if file_path:
        return f" --file {_quote(file_path)}"
    data = getattr(args, "data", None)
    if data:
        return f" --data {_quote(data)}"
    return ""


def _quote(value: Any) -> str:
    text = str(value)
    if os.name == "nt":
        return '"' + text.replace('"', '""') + '"'
    return shlex.quote(text)
