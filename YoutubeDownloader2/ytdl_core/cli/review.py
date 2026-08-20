"""
Interactive manual review of fingerprint-unverified downloads.

Walks every downloaded file that AcoustID could not confirm and lets the
user decide by ear, with single-key actions that are persisted to the
download state immediately:

    a      accept (mark as manually verified)
    d      delete the file (requires a second y)
    r      re-download the song (strict fingerprint mode)
    f      re-run the AcoustID fingerprint on the full file
    l      listen to a clip of the file (ffplay)
    s      skip
    q      quit (progress is saved after every action)

The review session is resumable: accepted/deleted songs are persisted to
``.download_state.json``, so re-running the command picks up where you left.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from ..state import load_state, save_state
from ..utils import format_duration

_ACTIONS = ("accept", "delete", "redownload", "fingerprint", "listen", "skip")


def _format_bytes(n: int) -> str:
    if not n:
        return "--"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _duration_of(path: Path) -> int:
    try:
        info = mutagen_file(str(path))
        if info and info.info and info.info.length:
            return int(info.info.length)
    except Exception:
        pass
    return 0


def mutagen_file(path: str) -> Any:
    import mutagen

    return mutagen.File(path)


def _build_candidates(
    state: dict,
    output_dir: Path,
    fmt: str,
    songs: dict[str, list[str]],
    only_suspects: bool,
) -> list[dict]:
    """Return unverified files present on disk, suspects first."""
    scope = {f"{a}::{s}" for a, lst in songs.items() for s in (lst or [])}
    candidates: list[dict] = []
    for key, entry in state.get("downloads", {}).items():
        if scope and key not in scope:
            continue
        if entry.get("fingerprint_verified"):
            continue
        fp = entry.get("file_path")
        if not fp:
            continue
        path = Path(fp)
        if not path.exists():
            continue
        suspect = bool(entry.get("fingerprint_confidence", 0) > 0)
        if only_suspects and not suspect:
            continue
        artist, song = key.split("::", 1)
        candidates.append(
            {
                "key": key,
                "artist": artist,
                "song": song,
                "entry": entry,
                "path": path,
                "suspect": suspect,
            }
        )
    candidates.sort(key=lambda c: (not c["suspect"], c["key"]))
    return candidates


def _play_clip(path: Path, seconds: int) -> Any:
    """Start ffplay on a ~12s clip in the background (non-blocking).

    Returns the running ``subprocess.Popen``, or None on failure. ffplay
    supports ``-ss``/``-t`` natively, so no temporary file is needed.
    """
    if not shutil.which("ffplay"):
        return None
    dur = _duration_of(path)
    if dur <= 0:
        return None
    offset = max(0, int(dur * 0.3))
    cmd = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
        "-ss",
        str(offset),
        "-t",
        str(seconds),
        str(path),
    ]
    try:
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs
        )
    except Exception:
        return None


def _stop_player(player: Any) -> None:
    """Kill a still-running ffplay process, if any."""
    if player is None:
        return
    try:
        if player.poll() is None:
            player.kill()
            player.wait(timeout=5)
    except Exception:
        pass


def _update_state(output_dir: Path, key: str, **fields: Any) -> None:
    """Reload the state from disk, apply *fields* to the entry, and save."""
    state = load_state(output_dir)
    entry = state.setdefault("downloads", {}).setdefault(key, {})
    entry.update(fields)
    save_state(state, output_dir)


def _run_fingerprint(dl: Any, path: Path, artist: str, song: str) -> tuple[bool, float, str]:
    if not getattr(dl, "acoustid_key", None):
        return False, 0.0, "no key"
    from ..fingerprint import verify_fingerprint

    with dl._fp_semaphore:
        return verify_fingerprint(
            path, artist, song, dl.acoustid_key, dl.config, dl._circuit_breaker,
            on_warn=dl.events.on_warn, on_info=dl.events.on_info,
            on_fingerprint_error=dl.events.on_fingerprint_error,
        )


def _redownload(dl: Any, artist: str, song: str, output_dir: Path, fmt: str, quality: str) -> None:
    """Re-run the strict download pipeline for a single song."""
    was_strict = dl.require_fingerprint
    dl.require_fingerprint = True
    try:
        dl.download_batch(
            songs={artist: [song]},
            output_dir=output_dir,
            fmt=fmt,
            quality=quality,
            skip_existing=False,
        )
    finally:
        dl.require_fingerprint = was_strict


def _fingerprint_text(entry: dict) -> str:
    conf = entry.get("fingerprint_confidence", 0) or 0
    label = entry.get("fingerprint_label") or "not attempted"
    if conf > 0 and "no match" not in str(label):
        return f"{label} ({conf:.0%})"
    return str(label)


def _render(
    console: Console,
    cand: dict,
    idx: int,
    total: int,
    stats: dict[str, int],
    last_action: str,
    first: bool,
) -> None:
    width = console.width - 2
    if not first:
        sys.stdout.write("\033[7A\033[J")
    entry, path = cand["entry"], cand["path"]
    dur = _duration_of(path)
    size = _format_bytes(path.stat().st_size)
    flag = " [red]SUSPECT[/red]" if cand["suspect"] else ""
    title = f"{cand['artist']} -- {cand['song']}"
    if len(title) > width:
        title = title[: width - 1] + "\u2026"
    pstr = str(path)
    if len(pstr) > width:
        pstr = "\u2026" + pstr[-(width - 1):]
    console.print(
        f"[bold cyan]{idx + 1}/{total}[/bold cyan]  [bold]{title}[/bold]{flag}  "
        f"[dim]{format_duration(dur)} | {size}[/dim]"
    )
    console.print(f"  [dim]{pstr}[/dim]")
    console.print(f"  [yellow]fingerprint:[/yellow] {_fingerprint_text(entry)}")
    console.print(f"  [dim]{last_action}[/dim]")
    console.print()
    console.print(
        "[bold]\\[a][/bold]ceptar  [bold]\\[d][/bold]eliminar  [bold]\\[r][/bold]e-descargar  "
        "[bold]\\[f][/bold]ingerprint  [bold]\\[l][/bold]istening  [bold]\\[s][/bold]altar  "
        "[bold]\\[q][/bold]uit"
    )
    console.print(
        f"[dim]aceptadas {stats['accept']} | eliminadas {stats['delete']} | "
        f"re-descargadas {stats['redownload']} | saltadas {stats['skip']}[/dim]"
    )


def run_interactive_review(
    console: Console,
    dl: Any,
    songs: dict[str, list[str]],
    output_dir: Path,
    fmt: str,
    quality: str,
    only_suspects: bool = False,
    clip_seconds: int = 12,
) -> None:
    output_dir = Path(output_dir)
    state = load_state(output_dir)
    candidates = _build_candidates(state, output_dir, fmt, songs, only_suspects)

    if not candidates:
        console.print(
            Panel(
                "[green]No unverified files to review.[/green]\n"
                "(All downloaded files were confirmed by fingerprint, or nothing to review "
                "in this output dir.)",
                title="[bold green] Review Complete[/bold green]",
                border_style="green",
            )
        )
        return

    try:
        import msvcrt
    except ImportError:
        msvcrt = None  # type: ignore[assignment]

    use_keyboard = msvcrt is not None and sys.stdin.isatty()

    stats = {"accept": 0, "delete": 0, "redownload": 0, "fingerprint": 0, "skip": 0}
    i = 0
    first = True
    last_action = ""
    player = None

    while i < len(candidates):
        cand = candidates[i]
        _render(console, cand, i, len(candidates), stats, last_action, first)
        first = False

        if use_keyboard:
            key = msvcrt.getch().lower()
            if key in (b"\xe0", b"\x00"):
                msvcrt.getch()
                continue
            if key == b"\r":
                key = b"a"
            elif key == b"\x1b":
                key = b"s"
        else:
            try:
                key = input("  [a]ccept [d]elete [r]edownload [f]ingerprint [l]isten [s]kip [q]uit: ").strip().lower()[:1].encode()
            except (EOFError, KeyboardInterrupt):
                key = b"q"

        if key != b"l":
            _stop_player(player)
            player = None

        if key == b"a":
            _update_state(
                output_dir, cand["key"],
                status="verified",
                fingerprint_verified=True,
                fingerprint_label="manual verify (listened)",
            )
            stats["accept"] += 1
            last_action = f"[green]Accepted: {cand['song']}[/green]"
            i += 1
        elif key == b"d":
            _render(console, cand, i, len(candidates), stats, "[yellow]press \\[y] to confirm delete, any other key to cancel[/yellow]", first)
            if use_keyboard:
                confirm = msvcrt.getch().lower() == b"y"
            else:
                try:
                    confirm = input("  Confirm delete? [y/N]: ").strip().lower() == "y"
                except (EOFError, KeyboardInterrupt):
                    confirm = False
            if confirm:
                try:
                    cand["path"].unlink(missing_ok=True)
                except OSError:
                    pass
                _update_state(
                    output_dir, cand["key"],
                    status="deleted",
                    file_path=None,
                    md5=None,
                    fingerprint_verified=False,
                    fingerprint_label="deleted manually",
                )
                stats["delete"] += 1
                last_action = f"[red]Deleted: {cand['song']}[/red]"
                i += 1
            else:
                last_action = "[dim]delete cancelled[/dim]"
        elif key == b"r":
            last_action = f"[cyan]Re-downloading {cand['song']} (strict)...[/cyan]"
            _render(console, cand, i, len(candidates), stats, last_action, first)
            _redownload(dl, cand["artist"], cand["song"], output_dir, fmt, quality)
            state = load_state(output_dir)
            refreshed = _build_candidates(state, output_dir, fmt, songs, only_suspects)
            try:
                ni = next(k for k, c in enumerate(refreshed) if c["key"] == cand["key"])
                cand = refreshed[ni]
                candidates = refreshed
                i = ni
            except StopIteration:
                candidates = refreshed
                i = min(i, len(candidates))
            stats["redownload"] += 1
            last_action = f"[cyan]Re-downloaded: {cand['song']}[/cyan]"
            first = True
        elif key == b"f":
            ok, conf, title = _run_fingerprint(dl, cand["path"], cand["artist"], cand["song"])
            if ok:
                _update_state(
                    output_dir, cand["key"],
                    status="verified",
                    fingerprint_verified=True,
                    fingerprint_confidence=conf,
                    fingerprint_label=f"verified {conf:.0%} (re-run)",
                )
                stats["accept"] += 1
                last_action = f"[green]Fingerprint confirmed {cand['song']} ({conf:.0%})[/green]"
                i += 1
            else:
                label = title if conf <= 0 else f"no match ({title or 'unknown'})"
                _update_state(
                    output_dir, cand["key"],
                    fingerprint_confidence=conf,
                    fingerprint_label=label,
                )
                last_action = f"[yellow]Fingerprint: {label}[/yellow]"
            stats["fingerprint"] += 1
        elif key == b"l":
            _stop_player(player)
            player = _play_clip(cand["path"], clip_seconds)
            if player is None:
                last_action = "[red]clip failed (ffplay missing or unreadable file)[/red]"
            else:
                last_action = "[cyan]playing in background -- continue reviewing with any key[/cyan]"
        elif key == b"s":
            stats["skip"] += 1
            last_action = f"[dim]skipped: {cand['song']}[/dim]"
            i += 1
        elif key == b"q":
            break
        else:
            last_action = "[dim]unknown key -- a/d/r/f/l/s/q[/dim]"

    _stop_player(player)
    _regenerate_not_verified(output_dir, songs)
    console.print(
        Panel(
            f"[bold]Accepted:[/bold] {stats['accept']}\n"
            f"[bold]Deleted:[/bold] {stats['delete']}\n"
            f"[bold]Re-downloaded:[/bold] {stats['redownload']}\n"
            f"[bold]Fingerprint re-run:[/bold] {stats['fingerprint']}\n"
            f"[bold]Skipped:[/bold] {stats['skip']}\n\n"
            f"[dim]{i}/{len(candidates)} reviewed -- remaining files still in "
            f"not_verified.json[/dim]",
            title="[bold yellow] Review Session Complete[/bold yellow]",
            border_style="yellow",
        )
    )


def _regenerate_not_verified(output_dir: Path, songs: dict[str, list[str]]) -> None:
    """Rebuild not_verified.json from state (scoped to *songs*)."""
    scope = {f"{a}::{s}" for a, lst in songs.items() for s in (lst or [])}
    state = load_state(output_dir)
    rows = []
    for key, entry in state.get("downloads", {}).items():
        if scope and key not in scope:
            continue
        if entry.get("fingerprint_verified"):
            continue
        status = entry.get("status")
        fp = entry.get("file_path")
        if status not in ("downloaded", "verified") or not fp:
            continue
        artist, song = key.split("::", 1)
        rows.append(
            {
                "artist": artist,
                "song": song,
                "file_path": fp,
                "fingerprint": entry.get("fingerprint_label") or "not attempted",
                "composite_score": 0,
                "suspect": bool(entry.get("fingerprint_confidence", 0) > 0),
            }
        )
    rows.sort(key=lambda r: (not r["suspect"], r["artist"], r["song"]))
    out_file = output_dir / "not_verified.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)