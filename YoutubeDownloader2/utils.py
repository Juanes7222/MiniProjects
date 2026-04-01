"""
Shared utility helpers: filename sanitisation, formatting, MD5, delay.
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


def sanitize_filename(name: str) -> str:
    """Strip illegal filesystem characters, collapse whitespace, cap at 200 chars."""
    name = re.sub(r'[/\\:*?"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:200]


def format_duration(seconds: int) -> str:
    """Return MM:SS string for a given number of seconds."""
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def format_size(nbytes: int) -> str:
    """Return a human-readable file-size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def compute_md5(path: Path) -> str:
    """Compute the MD5 hex-digest of a file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_delay(min_s: float, max_s: float) -> None:
    """Sleep for a uniformly-random duration in [min_s, max_s]."""
    time.sleep(random.uniform(min_s, max_s))


def check_ffmpeg(console: "Console") -> None:
    """Exit with an informative Rich panel if ffmpeg is not on PATH."""
    from rich.panel import Panel  # local import avoids circular deps at module level

    if shutil.which("ffmpeg") is None:
        console.print(
            Panel(
                "[red]ffmpeg executable not found on PATH.[/red]\n\n"
                "Install instructions:\n"
                "  [bold]Ubuntu/Debian:[/bold]  sudo apt install ffmpeg\n"
                "  [bold]macOS (brew):[/bold]   brew install ffmpeg\n"
                "  [bold]Windows:[/bold]        https://ffmpeg.org/download.html\n"
                "  [bold]Arch Linux:[/bold]     sudo pacman -S ffmpeg",
                title="[bold red]⚠ Missing Dependency: ffmpeg[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)