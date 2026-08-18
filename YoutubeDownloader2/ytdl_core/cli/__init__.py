"""
ytdl_core CLI — Rich terminal interface for MusicDownloader.

This package is the ONLY place where Rich, argparse, and sys.exit() appear.
It implements RichEvents(DownloaderEvents) to translate every library callback
into formatted terminal output, then wires everything together in main().

Entry point (after ``pip install ytdl-core[cli]``):

    ytdl --file songs.json --quality 320 --musicbrainz
    python -m ytdl_core.cli --file songs.json --acoustid-key KEY

Submodules
----------
arg_parser   – argparse definitions (``parse_args``)
rich_ui      – ``RichEvents(DownloaderEvents)`` implementation
dry_run      – dry-run preview table
interactive  – keyboard/input candidate selection + ffplay preview
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ..config import Config
from ..core import MusicDownloader
from ..utils import check_ffmpeg

# Re-export the submodules so ``from .cli import parse_args`` still works.
from .arg_parser import parse_args  # noqa: F401
from .dry_run import dry_run_table
from .interactive import make_interactive_confirm, make_interactive_selector
from .rich_ui import RichEvents  # noqa: F401


def main() -> None:
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
        if args.url:
            songs = {}
            pairs = []
        elif args.file:
            with args.file.open("r", encoding="utf-8") as fh:
                songs: dict = json.load(fh)
            pairs = [
                (artist, song) for artist, lst in songs.items() for song in (lst or [])
            ]
        else:
            songs = json.loads(args.data)
            pairs = [
                (artist, song) for artist, lst in songs.items() for song in (lst or [])
            ]
    except (json.JSONDecodeError, OSError) as exc:
        console.print(
            Panel(
                f"[red]{exc}[/red]",
                title="[bold red] Input Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)

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
        dry_run_table(console, pairs, args, config)
        if log_fh:
            log_fh.close()
        return

    events = RichEvents(console, args.score_threshold, config)

    if args.interactive:
        stop_event = threading.Event()
        events.confirm_fn = make_interactive_confirm(
            console,
            threading.Lock(),
            threading.Lock(),
            stop_event,
        )

    if args.select:
        stop_event = threading.Event()
        search_opts = {
            "max_results": args.max_results,
            "cookies_browser": args.cookies_browser,
            "cookies_file": str(args.cookies) if args.cookies else None,
            "proxy": args.proxy,
        }
        events.selector_fn = make_interactive_selector(
            console,
            threading.Lock(),
            threading.Lock(),
            stop_event,
            preview=args.preview or args.video_preview,
            video_preview=args.video_preview,
            preview_seconds=args.preview_seconds,
            search_opts=search_opts,
            pause_progress=events.suspend_progress,
            resume_progress=events.resume_progress,
            events=events,
        )

    import musicbrainzngs as _mbz

    if args.musicbrainz:
        _mbz.set_useragent(
            "YTMusicDownloader",
            "2.0",
            "https://github.com/example/yt-music-downloader",
        )

    dl = MusicDownloader(
        config=config,
        events=events,
        acoustid_key=args.acoustid_key,
        force_fingerprint=args.force_fingerprint,
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
        cookies_file=str(args.cookies) if args.cookies else None,
        proxy=args.proxy,
    )

    if getattr(args, "verify", False) or getattr(args, "repair", False):
        if getattr(args, "repair", False):
            console.print(
                "[dim italic]Entering Repair Mode: Auditing library before re-download...[/dim italic]"
            )

        all_results = dl.verify_library(
            songs=songs,
            output_dir=args.output,
            fmt=args.format,
        )
        if args.report:
            from ..reports import export_report

            export_report([r.to_dict() for r in all_results], args.output, args.report)
            console.print(
                f"[green]  Verification reports saved to: {args.output.resolve()}[/green]"
            )

        missing_songs = {}
        for r in all_results:
            if getattr(r, "status", "") not in ("downloaded", "verified"):
                missing_songs.setdefault(r.artist, []).append(r.song)
                # Ensure the bad file is deleted if attempting to repair
                if getattr(args, "repair", False) and getattr(r, "file_path", None):
                    try:
                        p = Path(r.file_path)
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass

        if missing_songs:
            import json as _json

            out_missing = args.output / "missing_songs.json"
            with out_missing.open("w", encoding="utf-8") as f:
                _json.dump(missing_songs, f, indent=2, ensure_ascii=False)
            console.print(
                f"[yellow]  Generated missing/failed songs to: {out_missing}[/yellow]"
            )

            if getattr(args, "repair", False):
                total_missing = sum(len(lst) for lst in missing_songs.values())
                console.print(
                    f"\n[cyan]Starting repair cycle for {total_missing} "
                    f"missing/corrupted files...[/cyan]"
                )

                # Force re-download by bypassing 'skip_existing'
                dl.download_batch(
                    songs=missing_songs,
                    output_dir=args.output,
                    fmt=args.format,
                    quality=args.quality,
                    skip_existing=False,
                    report_formats=args.report or None,
                    update_json_path=args.file if args.update_json else None,
                )
        else:
            console.print("[green]  All songs verified successfully![/green]")

    elif getattr(args, "url", None):
        console.print(f"[cyan]Downloading from URL: {args.url}[/cyan]")

        limit_val = getattr(args, "limit", None)
        dl.download_url(
            url=args.url,
            output_dir=args.output,
            fmt=args.format,
            quality=args.quality,
            max_downloads=limit_val,
            skip_existing=args.skip_existing,
            match_title=args.match_title,
            reject_title=args.reject_title,
        )
    else:
        dl.download_batch(
            songs=songs,
            output_dir=args.output,
            fmt=args.format,
            quality=args.quality,
            skip_existing=args.skip_existing,
            report_formats=args.report or None,
            update_json_path=args.file if args.update_json else None,
        )

    if not (getattr(args, "verify", False) or getattr(args, "repair", False)) and args.report:
        console.print(f"[green]  Reports saved to: {args.output.resolve()}[/green]")

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
