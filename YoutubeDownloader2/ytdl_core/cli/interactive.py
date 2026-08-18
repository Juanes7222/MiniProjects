"""
Interactive selection and preview helpers for the ytdl CLI.

Provides keyboard-driven candidate selection (Windows via ``msvcrt``,
fallback via ``input()``) and audio/video preview via ``ffplay``.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any, Optional

from rich.console import Console


def preview_candidate(
    candidate: dict,
    preview_type: str,
    seconds: int,
    opts: dict,
) -> bool:
    """Stream a short preview of a candidate to ffplay.

    Args:
        candidate: The candidate metadata dict (must contain webpage_url or url).
        preview_type: 'audio' or 'video'.
        seconds: How many seconds to preview.
        opts: Search options dict (cookies, proxy, etc.)

    Returns:
        True if the preview played successfully, False otherwise.
    """
    url = candidate.get("webpage_url") or candidate.get("url", "")
    if not url:
        return False

    if not shutil.which("ffplay"):
        return False

    try:
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if opts.get("cookies_browser"):
            ydl_opts["cookiesfrombrowser"] = (opts["cookies_browser"],)
        if opts.get("cookies_file"):
            ydl_opts["cookiefile"] = str(opts["cookies_file"])
        if opts.get("proxy"):
            ydl_opts["proxy"] = opts["proxy"]

        import yt_dlp

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return False

            if preview_type == "audio":
                candidates = [
                    f
                    for f in info.get("formats", [])
                    if f.get("vcodec") == "none"
                    and f.get("acodec") != "none"
                    and f.get("url")
                ]
            else:
                candidates = [
                    f
                    for f in info.get("formats", [])
                    if f.get("vcodec") != "none"
                    and f.get("acodec") != "none"
                    and f.get("url")
                ]

            if candidates:
                stream_url = candidates[0]["url"]
            else:
                stream_url = info.get("url", "")
            if not stream_url:
                return False

        cmd = [
            "ffplay",
            "-t",
            str(seconds),
            "-autoexit",
            "-loglevel",
            "quiet",
        ]
        if preview_type == "audio":
            cmd.extend(["-nodisp", "-vn"])
        cmd.append(stream_url)

        subprocess.run(
            cmd,
            timeout=seconds + 30,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True

    except Exception:
        return False


def _render_selection(
    console: Console,
    ranked: list[tuple[dict, int, dict]],
    current: int,
    preview: bool,
    video_preview: bool,
) -> None:
    from ..utils import format_duration as _fd
    from rich.markup import escape as _esc

    entry, sc, _bd = ranked[current]
    title = _esc(str(entry.get("title", ""))[:60])
    channel = _esc(str(entry.get("channel") or entry.get("uploader", ""))[:25])
    dur = _fd(entry.get("duration") or 0)
    n = len(ranked)

    sep = "=" * min(60, console.width - 4)
    console.print(f"  [dim]{sep}[/dim]")
    console.print(
        f"  [dim]|[/dim] [bold cyan]#{current + 1}[/bold cyan] of [bold]{n}[/bold]     "
        f"[white]{title}[/white] [dim]|[/dim] "
        f"[yellow]{channel}[/yellow] [dim]|[/dim] "
        f"[green]{dur}[/green] [dim]|[/dim] Score: [bold]{sc}[/bold]"
    )
    help_parts = ["[bold]Up/Down[/bold]=navigate", "[bold]Enter[/bold]=download"]
    if preview or video_preview:
        help_parts.append("[bold]Space[/bold]=preview")
    help_parts.append("[bold]Esc[/bold]=skip")
    help_parts.append("[bold]Q[/bold]=quit")
    console.print(f"  [dim]{sep}[/dim]  [dim]{' | '.join(help_parts)}[/dim]")


def _keyboard_select(
    console: Console,
    interactive_lock: threading.Lock,
    console_lock: threading.Lock,
    ranked: list[tuple[dict, int, dict]],
    stop_event: threading.Event,
    preview: bool,
    video_preview: bool,
    preview_seconds: int,
    search_opts: dict,
    events: Any = None,
) -> Optional[dict]:
    import sys

    try:
        import msvcrt
    except ImportError:
        return None

    n = len(ranked)
    current = 0
    first = True

    # Flush any buffered output before starting selection
    if events:
        events.flush_buffer()
        events.start_buffering()

    try:
        # Suspend the progress bar so ANSI escape codes work cleanly
        if events:
            events.suspend_progress()

        while not stop_event.is_set():
            # Flush buffered output from other threads before redrawing prompt
            if events and not first:
                events.flush_buffer()

            with interactive_lock:
                with console_lock:
                    if not first:
                        sys.stdout.write("\033[4A\033[J")
                    first = False
                    _render_selection(console, ranked, current, preview, video_preview)

            key = msvcrt.getch()

            if key == b"\xe0":
                key = msvcrt.getch()
                if key == b"H":
                    current = (current - 1) % n
                elif key == b"P":
                    current = (current + 1) % n

            elif key == b"\r":
                if events:
                    events.flush_buffer()
                entry, sc, _bd = ranked[current]
                with interactive_lock:
                    with console_lock:
                        sys.stdout.write("\033[4A\033[J")
                        console.print(
                            f"  [green]Downloading #{current + 1}: {entry.get('title', '')} "
                            f"(score: {sc})[/green]"
                        )
                return entry

            elif key == b" " and (preview or video_preview):
                if events:
                    events.flush_buffer()
                entry = ranked[current][0]
                pt = "video" if video_preview else "audio"
                with interactive_lock:
                    with console_lock:
                        sys.stdout.write("\033[4A\033[J")
                        console.print(
                            f"  [cyan]Previewing ({pt}) candidate #{current + 1}: "
                            f"{entry.get('title', '')[:60]}...[/cyan]"
                        )
                preview_candidate(entry, pt, preview_seconds, search_opts)

            elif key == b"\x1b":
                if events:
                    events.flush_buffer()
                return None

            elif key in (b"q", b"Q"):
                if events:
                    events.flush_buffer()
                stop_event.set()
                return None
    finally:
        if events:
            events.resume_progress()
            events.stop_buffering()
            events.flush_buffer()

    return None


def make_interactive_selector(
    console: Console,
    console_lock: threading.Lock,
    interactive_lock: threading.Lock,
    stop_event: threading.Event,
    preview: bool,
    video_preview: bool,
    preview_seconds: int,
    search_opts: dict,
    pause_progress: Optional[callable] = None,
    resume_progress: Optional[callable] = None,
    events: Any = None,
) -> Any:
    use_keyboard = False
    try:
        import msvcrt  # noqa: F401

        use_keyboard = True
    except ImportError:
        pass

    def selector_fn(
        artist: str,
        song: str,
        ranked: list[tuple[dict, int, dict]],
    ) -> Optional[dict]:
        if not ranked:
            with interactive_lock:
                with console_lock:
                    console.print("[yellow]  No candidates to select from.[/yellow]")
            return None

        if use_keyboard:
            return _keyboard_select(
                console,
                interactive_lock,
                console_lock,
                ranked,
                stop_event,
                preview,
                video_preview,
                preview_seconds,
                search_opts,
                events,
            )

        # Fallback: input()-based selector for non-Windows platforms
        while not stop_event.is_set():
            with interactive_lock:
                with console_lock:
                    console.print()
                    prompt_parts = [
                        f"[bold yellow]Select candidate[/bold yellow] [1-{len(ranked)}]"
                    ]
                    if preview:
                        prompt_parts.append("[bold]p#[/bold]=preview audio")
                    if video_preview:
                        prompt_parts.append("[bold]v#[/bold]=preview video")
                    prompt_parts.append("[bold]s[/bold]=skip")
                    prompt_parts.append("[bold]q[/bold]=quit")
                    console.print("  " + " | ".join(prompt_parts))

            try:
                if pause_progress:
                    pause_progress()
                choice = input("  Choice: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"
            finally:
                if resume_progress:
                    resume_progress()

            if choice == "q":
                stop_event.set()
                return None

            if choice == "s":
                return None

            if choice.startswith(("p", "v")):
                try:
                    idx = int(choice[1:]) - 1
                except (ValueError, IndexError):
                    with interactive_lock:
                        with console_lock:
                            console.print(
                                "  [red]Invalid format. Use p#, v#, or just a number.[/red]"
                            )
                    continue

                if idx < 0 or idx >= len(ranked):
                    with interactive_lock:
                        with console_lock:
                            console.print(
                                f"  [red]Invalid index. Must be 1-{len(ranked)}.[/red]"
                            )
                    continue

                preview_type = "video" if choice.startswith("v") else "audio"
                entry = ranked[idx][0]
                with interactive_lock:
                    with console_lock:
                        console.print(
                            f"  [cyan]Previewing ({preview_type}) candidate #{idx + 1}: "
                            f"{entry.get('title', '')[:60]}...[/cyan]"
                        )

                ok = preview_candidate(entry, preview_type, preview_seconds, search_opts)
                if not ok:
                    with interactive_lock:
                        with console_lock:
                            console.print(
                                "  [red]Preview failed (ffplay not found or stream error).[/red]"
                            )
                continue

            try:
                idx = int(choice) - 1
            except ValueError:
                with interactive_lock:
                    with console_lock:
                        console.print(
                            "  [red]Invalid input. Enter a number, p#, v#, s, or q.[/red]"
                        )
                continue

            if idx < 0 or idx >= len(ranked):
                with interactive_lock:
                    with console_lock:
                        console.print(
                            f"  [red]Invalid index. Must be 1-{len(ranked)}.[/red]"
                        )
                continue

            entry, sc, _bd = ranked[idx]
            with interactive_lock:
                with console_lock:
                    console.print(
                        f"  [green]Selected #{idx + 1}: {entry.get('title', '')} "
                        f"(score: {sc})[/green]"
                    )
            return entry

        return None

    return selector_fn


def make_interactive_confirm(
    console: Console,
    console_lock: threading.Lock,
    interactive_lock: threading.Lock,
    stop_event: threading.Event,
) -> Any:
    def confirm_fn(artist: str, song: str, best_result: dict) -> bool:
        from ..utils import format_duration as _fd
        from rich.panel import Panel

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
