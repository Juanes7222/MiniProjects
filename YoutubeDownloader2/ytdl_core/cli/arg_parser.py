"""
Argument parser for ytdl_core CLI.

All argparse definitions live here. The ``parse_args()`` function returns
a fully-validated ``argparse.Namespace``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import Config
from ..profiles import ProfileError, load_profile

_CONFIG = Config()


def _explicit_source_options(argv: list[str] | None) -> set[str]:
    option_map = {
        "--file": "file",
        "--data": "data",
        "--url": "url",
        "--retry": "retry",
    }
    tokens = list(argv) if argv is not None else sys.argv[1:]
    return {
        option_map[token.split("=", 1)[0]]
        for token in tokens
        if token.split("=", 1)[0] in option_map
    }


def _validate_profile_choices(
    parser: argparse.ArgumentParser,
    settings: dict[str, object],
    profile_name: str,
) -> None:
    actions = {action.dest: action for action in parser._actions}
    for key, value in settings.items():
        action = actions.get(key)
        if action is None or not action.choices:
            continue
        values = value if isinstance(value, list) else [value]
        invalid = [item for item in values if item not in action.choices]
        if invalid:
            parser.error(
                f"Invalid value for '{key}' in profile '{profile_name}': "
                f"{invalid[0]!r}. Expected one of {', '.join(map(str, action.choices))}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ytdl",
        description="YT Music Downloader v2.0 -- batch audio download with metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "Examples:\n"
            "  ytdl --file songs.json\n"
            "  ytdl --file songs.json --format mp3 --quality 320 "
            "--workers 3 --musicbrainz --report json\n"
            "  ytdl --file songs.json --acoustid-key KEY --quality 320\n"
            "  ytdl --file songs.json --skip-fingerprint --no-silence-check\n"
            '  ytdl --data \'{"Radiohead": ["Creep"]}\' --dry-run\n'
            "  ytdl --profile high-quality --file songs.json\n"
            "  ytdl --output ./downloads --retry"
        ),
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", metavar="PATH", type=Path)
    src.add_argument("--data", metavar="JSON_STR")
    src.add_argument(
        "--url",
        metavar="URL",
        type=str,
        help="Download a playlist, channel, or video directly by URL",
    )
    src.add_argument(
        "--retry",
        action="store_true",
        help="Retry songs saved in the output directory retry queue",
    )

    p.add_argument(
        "--profile",
        metavar="NAME",
        help="Apply a profile from profiles.toml",
    )
    p.add_argument(
        "--profiles-file",
        metavar="PATH",
        type=Path,
        help="Path to profiles.toml",
    )

    p.add_argument("--output", metavar="DIR", type=Path, default=Path(_CONFIG.DEFAULT_OUTPUT_DIR))
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
    decision = p.add_mutually_exclusive_group()
    decision.add_argument(
        "--jev",
        action="store_true",
        help="Use Jev through Vercel AI Gateway to choose the candidate for each song.",
    )
    decision.add_argument(
        "--kev",
        action="store_true",
        help="Use a local Kev server to choose the candidate for each song.",
    )
    p.add_argument(
        "--kev-url",
        metavar="URL",
        default="http://127.0.0.1:8009",
        help="Kev server URL (default: %(default)s).",
    )
    p.add_argument(
        "--kev-model",
        metavar="MODEL",
        default="kev-latest",
        help="Kev API model name; the server's --run option selects the checkpoint (default: %(default)s).",
    )
    p.add_argument(
        "--kev-run",
        metavar="CHECKPOINT",
        default="jaredpalmer/kev-4b",
        help="Checkpoint to download and serve (default: %(default)s).",
    )
    p.add_argument(
        "--kev-dir",
        metavar="PATH",
        type=Path,
        default=Path.home() / ".cache" / "ytdl" / "kev",
        help="Kev repository directory (default: %(default)s).",
    )
    p.add_argument(
        "--kev-port",
        metavar="INT",
        type=int,
        default=8009,
        help="Local Kev server port (default: %(default)s).",
    )
    p.add_argument(
        "--kev-startup-timeout",
        metavar="INT",
        type=int,
        default=900,
        help="Seconds to wait for Kev startup (default: %(default)s).",
    )
    p.add_argument(
        "--kev-skip-update",
        action="store_true",
        help="Do not pull the latest Kev repository changes.",
    )
    p.add_argument(
        "--kev-threshold",
        metavar="FLOAT",
        type=float,
        default=_CONFIG.JEV_DEFAULT_THRESHOLD,
        help="Minimum average Kev probability required to download (default: %(default)s).",
    )
    p.add_argument(
        "--kev-runs",
        metavar="INT",
        type=int,
        default=_CONFIG.JEV_DEFAULT_RUNS,
        help="Number of Kev evaluations per song (default: %(default)s).",
    )
    p.add_argument(
        "--jev-threshold",
        metavar="FLOAT",
        type=float,
        default=_CONFIG.JEV_DEFAULT_THRESHOLD,
        help="Minimum average Jev probability required to download (default: %(default)s).",
    )
    p.add_argument(
        "--jev-runs",
        metavar="INT",
        type=int,
        default=_CONFIG.JEV_DEFAULT_RUNS,
        help="Number of Jev evaluations per song (default: %(default)s).",
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

    known, _ = p.parse_known_args(argv)
    if known.profiles_file and not known.profile:
        p.error("--profiles-file requires --profile")

    settings: dict[str, object] = {}
    profiles_path = None
    if known.profile:
        allowed_keys = {
            action.dest
            for action in p._actions
            if action.dest not in {"help", "profile", "profiles_file"}
        }
        try:
            profiles_path, settings = load_profile(
                known.profile,
                known.profiles_file,
                allowed_keys,
            )
        except ProfileError as error:
            p.error(str(error))

        explicit_sources = _explicit_source_options(argv)
        if explicit_sources:
            for option in ("file", "data", "url", "retry"):
                settings.pop(option, None)
        profile_sources = [
            option for option in ("file", "data", "url", "retry") if settings.get(option)
        ]
        if len(profile_sources) > 1:
            p.error(
                f"Profile '{known.profile}' defines multiple sources: {', '.join(profile_sources)}"
            )
        _validate_profile_choices(p, settings, known.profile)
        p.set_defaults(**settings)

    args = p.parse_args(argv)
    args.profiles_path = profiles_path
    if known.report:
        args.report = known.report

    if args.retry and (args.verify or args.repair or args.review):
        p.error("--retry cannot be combined with --verify, --repair, or --review")

    sources = [args.file, args.data, args.url, args.retry]
    if sum(source is not None and source is not False for source in sources) != 1:
        p.error("exactly one of --file, --data, --url, or --retry is required")

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
    if not 1 <= args.kev_port <= 65535:
        p.error("--kev-port must be between 1 and 65535")
    if args.kev_startup_timeout < 1:
        p.error("--kev-startup-timeout must be greater than 0")
    if not 0 < args.jev_threshold <= 1:
        p.error("--jev-threshold must be greater than 0 and at most 1")
    if not 0 < args.kev_threshold <= 1:
        p.error("--kev-threshold must be greater than 0 and at most 1")
    if not 1 <= args.jev_runs <= _CONFIG.JEV_MAX_RUNS:
        p.error(f"--jev-runs must be between 1 and {_CONFIG.JEV_MAX_RUNS}")
    if not 1 <= args.kev_runs <= _CONFIG.JEV_MAX_RUNS:
        p.error(f"--kev-runs must be between 1 and {_CONFIG.JEV_MAX_RUNS}")
    if args.jev and args.url:
        p.error("--jev cannot be used with --url")
    if args.jev and (args.verify or args.repair or args.review):
        p.error("--jev cannot be combined with --verify, --repair, or --review")
    if args.jev and args.dry_run:
        p.error("--jev cannot be used with --dry-run")
    if args.kev and args.url:
        p.error("--kev cannot be used with --url")
    if args.kev and (args.verify or args.repair or args.review):
        p.error("--kev cannot be combined with --verify, --repair, or --review")
    if args.kev and args.dry_run:
        p.error("--kev cannot be used with --dry-run")

    args.workers = max(1, min(args.workers, _CONFIG.MAX_WORKERS))
    raw_sources = (
        args.sources.split(",")
        if isinstance(args.sources, str)
        else [str(source) for source in args.sources]
    )
    args.sources = [source.strip().lower() for source in raw_sources if source.strip()]
    return args
