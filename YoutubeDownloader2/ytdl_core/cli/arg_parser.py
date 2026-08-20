"""
Argument parser for ytdl_core CLI.

All argparse definitions live here. The ``parse_args()`` function returns
a fully-validated ``argparse.Namespace``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config

_CONFIG = Config()


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
            '  ytdl --data \'{"Radiohead": ["Creep"]}\' --dry-run'
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", metavar="PATH", type=Path)
    src.add_argument("--data", metavar="JSON_STR")
    src.add_argument(
        "--url",
        metavar="URL",
        type=str,
        help="Download a playlist, channel, or video directly by URL",
    )

    p.add_argument(
        "--output", metavar="DIR", type=Path, default=Path(_CONFIG.DEFAULT_OUTPUT_DIR)
    )
    p.add_argument(
        "--format",
        metavar="FORMAT",
        choices=_CONFIG.SUPPORTED_FORMATS,
        default=_CONFIG.DEFAULT_FORMAT,
    )
    p.add_argument(
        "--quality",
        metavar="QUALITY",
        default=_CONFIG.DEFAULT_QUALITY,
        help="Audio bitrate (128, 192, 320) or video height (360, 480, 720, 1080, etc.)",
    )
    p.add_argument(
        "--limit",
        metavar="INT",
        type=int,
        help="Limit number of downloads from a URL (e.g. for playlists)",
    )
    p.add_argument("--max-results", metavar="INT", type=int, default=_CONFIG.DEFAULT_MAX_RESULTS)
    p.add_argument("--max-duration", metavar="INT", type=int, default=_CONFIG.MAX_DURATION_SECONDS)
    p.add_argument("--min-duration", metavar="INT", type=int, default=_CONFIG.MIN_DURATION_SECONDS)
    p.add_argument(
        "--fuzzy-threshold", metavar="INT", type=int, default=_CONFIG.DEFAULT_FUZZY_THRESHOLD
    )
    p.add_argument("--workers", metavar="INT", type=int, default=_CONFIG.DEFAULT_WORKERS)
    p.add_argument(
        "--delay",
        metavar="FLOAT",
        type=float,
        nargs=2,
        default=[_CONFIG.DEFAULT_DELAY_MIN, _CONFIG.DEFAULT_DELAY_MAX],
    )
    p.add_argument("--sources", metavar="LIST", default=",".join(_CONFIG.DEFAULT_SOURCES))
    p.add_argument(
        "--cookies-browser",
        metavar="BROWSER",
        choices=["chrome", "firefox", "edge", "safari"],
    )
    p.add_argument("--cookies", metavar="FILE", type=Path, help="Path to a cookies.txt file")
    p.add_argument("--proxy", metavar="URL")
    p.add_argument("--musicbrainz", action="store_true")

    p.add_argument("--acoustid-key", metavar="KEY", dest="acoustid_key")
    p.add_argument("--skip-fingerprint", action="store_true")
    p.add_argument("--force-fingerprint", action="store_true")
    p.add_argument(
        "--fingerprint-mode",
        choices=["lenient", "strict"],
        default="lenient",
        help="lenient: download everything, attempt AcoustID on all songs and "
        "report which ones could not be confirmed (default). "
        "strict: only download a song when AcoustID confirms it; "
        "unconfirmed songs are marked failed.",
    )
    p.add_argument(
        "--score-threshold", metavar="INT", type=int, default=_CONFIG.SCORE_THRESHOLD_REJECT
    )
    p.add_argument("--no-silence-check", action="store_true")

    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--update-json", action="store_true")
    p.add_argument(
        "--report",
        metavar="FORMAT",
        action="append",
        choices=["json", "csv", "m3u"],
        default=[],
        dest="report",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--interactive", action="store_true")
    p.add_argument(
        "--select",
        action="store_true",
        help="Interactive candidate selection: browse all results and pick which to download",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Enable audio preview in --select mode (requires ffplay)",
    )
    p.add_argument(
        "--video-preview",
        action="store_true",
        help="Enable video preview in --select mode (requires ffplay)",
    )
    p.add_argument(
        "--preview-seconds",
        metavar="INT",
        type=int,
        default=15,
        help="Preview duration in seconds (default: 15)",
    )
    p.add_argument("--log-file", metavar="FILE", type=Path)
    p.add_argument(
        "--match-title",
        metavar="REGEX",
        type=str,
        help="Include only videos matching this regex in the title",
    )
    p.add_argument(
        "--reject-title",
        metavar="REGEX",
        type=str,
        help="Exclude videos matching this regex in the title",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify the local library instead of downloading.",
    )
    mode.add_argument(
        "--repair",
        action="store_true",
        help="Verify the local library and automatically re-download missing or corrupted files.",
    )
    mode.add_argument(
        "--review",
        action="store_true",
        help="Interactively review fingerprint-unverified files: accept, delete, "
        "re-download, re-run fingerprint, or listen before deciding.",
    )

    p.add_argument(
        "--review-only-suspects",
        action="store_true",
        help="In --review, only show files AcoustID matched to a different song "
        "(suspected wrong downloads).",
    )

    p.add_argument(
        "--review-clip-seconds",
        metavar="INT",
        type=int,
        default=12,
        help="Length in seconds of the listening clip used by --review (default: 12)",
    )

    args = p.parse_args()

    if args.url:
        if args.verify or args.repair or args.review:
            p.error("--verify, --repair, and --review cannot be used with --url")
        if args.update_json:
            p.error("--update-json cannot be used with --url")
        if args.dry_run:
            p.error("--dry-run cannot be used with --url")

    if args.review_only_suspects and not args.review:
        p.error("--review-only-suspects requires --review")
    if args.review and (args.verify or args.repair):
        p.error("--review cannot be combined with --verify or --repair")

    args.workers = max(1, min(args.workers, _CONFIG.MAX_WORKERS))
    args.sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    return args
