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
from ..decider_server import (
    DEFAULT_GRAPH_B_BUCKETS as _DECIDER_DEFAULT_GRAPH_B_BUCKETS,
    DEFAULT_GRAPH_T_BUCKETS as _DECIDER_DEFAULT_GRAPH_T_BUCKETS,
)
from ..decider_server import DEFAULT_PACKAGE as _DECIDER_DEFAULT_PACKAGE
from ..profiles import ProfileError, load_profile

_CONFIG = Config()

# One flag per decision model, and the same three rules for all of them. Written
# as a table rather than six repeated blocks because the rules are not per-model:
# a decision model re-runs the search it is deciding between, so it cannot share
# a run with a single URL, with the post-download verifiers, or with a dry run.
_DECISION_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("jev", "--jev"),
    ("kev", "--kev"),
    ("decider", "--decider"),
)


def _reject_provider_conflicts(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for attribute, flag in _DECISION_PROVIDERS:
        if not getattr(args, attribute, False):
            continue
        if args.url:
            parser.error(f"{flag} cannot be used with --url")
        if args.verify or args.repair or args.review:
            parser.error(f"{flag} cannot be combined with --verify, --repair, or --review")
        if args.dry_run:
            parser.error(f"{flag} cannot be used with --dry-run")


# Former flag -> what to type instead. The old spellings are still accepted as
# argparse aliases on the same dest, so an existing command line keeps working;
# this only says so once, on stderr, rather than letting it rot unnoticed.
_DEPRECATED_FLAGS: dict[str, str] = {
    "--jev-runs": "--decision-runs",
    "--kev-runs": "--decision-runs",
    "--jev-threshold": "--decision-threshold",
    "--kev-threshold": "--decision-threshold",
    "--kev-timeout": "--decision-timeout",
}


def _warn_deprecated_flags(argv: list[str] | None) -> None:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    used = {
        replacement
        for token in tokens
        if (replacement := _DEPRECATED_FLAGS.get(token.split("=", 1)[0])) is not None
    }
    for replacement in sorted(used):
        old = [flag for flag, new in _DEPRECATED_FLAGS.items() if new == replacement]
        print(
            f"warning: {', '.join(sorted(old))} {'are' if len(old) > 1 else 'is'} "
            f"deprecated; use {replacement}",
            file=sys.stderr,
        )


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
    p.add_argument(
        "--max-duration",
        metavar="INT",
        type=int,
        default=None,
        help=(
            f"Max length in seconds (default: {_CONFIG.MAX_DURATION_SECONDS}, "
            "0 disables it). With --url on a single video the default is not "
            "applied unless this flag is given explicitly."
        ),
    )
    p.add_argument(
        "--min-duration",
        metavar="INT",
        type=int,
        default=None,
        help=(
            f"Min length in seconds (default: {_CONFIG.MIN_DURATION_SECONDS}, "
            "0 disables it). With --url on a single video the default is not "
            "applied unless this flag is given explicitly."
        ),
    )
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
        help=(
            "Optional extra pause before each song, in seconds. Off by default: "
            "requests are already paced per service, and only when a budget is "
            "actually exhausted."
        ),
    )
    p.add_argument(
        "--decision-timeout",
        "--kev-timeout",
        dest="decision_timeout",
        metavar="SECONDS",
        type=int,
        default=_CONFIG.DECISION_TIMEOUT_SECONDS,
        help=(
            "Per-evaluation timeout for the decision model. Raise it when the "
            "server is healthy but slow. A timeout is never retried: a slow answer "
            "needs more time, not another attempt. --kev-timeout is the former name "
            "and still works."
        ),
    )
    p.add_argument(
        "--decision-questions",
        metavar="INT",
        type=int,
        default=_CONFIG.DECISION_MAX_QUESTIONS,
        help=(
            "Question budget per evaluation. Cost scales with questions, so this is "
            "the knob that actually bounds decision latency."
        ),
    )
    p.add_argument(
        "--decision-in-flight",
        metavar="INT",
        type=int,
        default=_CONFIG.DECISION_MAX_IN_FLIGHT,
        help=(
            "How many decision evaluations may be in flight at once. kev.serve "
            "batches up to 64 queued requests into one model pass, so several at "
            "once finish in roughly one pass instead of N; 1 makes the model "
            "strictly serial at the cost of throughput. Bounded by VRAM as much as "
            "by speed (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--no-pipeline",
        action="store_true",
        help=(
            "Run each song end-to-end in one worker instead of staging search, "
            "verification, download and tagging separately."
        ),
    )
    p.add_argument(
        "--stage-workers",
        metavar="INT",
        nargs="*",
        type=int,
        help=(
            "Per-stage thread counts, in pipeline order: search verify download "
            "post. Auto-sized by default (e.g. --stage-workers 12 8 12 6)."
        ),
        dest="stage_workers",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and do not write the on-disk candidate/metadata/cover caches.",
    )
    p.add_argument(
        "--fingerprint-concurrency",
        metavar="INT",
        type=int,
        default=_CONFIG.FP_CONCURRENCY,
        help="How many 90-second partial downloads may be fingerprinted at once.",
    )
    p.add_argument("--sources", metavar="LIST", default=",".join(_CONFIG.DEFAULT_SOURCES))
    p.add_argument(
        "--use-aria2c",
        action="store_true",
        help=(
            "Hand transfers to aria2c when the binary is available. Measured on a "
            "~450 KB/s link it was 75%% SLOWER than yt-dlp's own downloader (22.3s "
            "vs 12.7s for the same file), because splitting one connection's worth "
            "of bandwidth costs more in connection setup than it recovers. Only "
            "worth trying where a single connection is throttled or latency is high."
        ),
    )
    p.add_argument(
        "--fragment-concurrency",
        metavar="INT",
        type=int,
        default=_CONFIG.FRAGMENT_CONCURRENCY,
        help=(
            "How many DASH fragments to fetch at once per file (default: "
            "%(default)s). Note this is currently inert: the configured YouTube "
            "player clients offer audio as a single progressive HTTPS file, so "
            "there are no fragments. It applies if formats arrive segmented."
        ),
    )
    p.add_argument(
        "--cookies-browser",
        metavar="BROWSER",
        choices=["chrome", "firefox", "edge", "safari"],
    )
    p.add_argument("--cookies", metavar="FILE", type=Path, help="Path to a cookies.txt file")
    p.add_argument("--proxy", metavar="URL")
    p.add_argument(
        "--musicbrainz",
        action="store_true",
        help="Enrich tags from MusicBrainz and use it as the duration reference.",
    )
    p.add_argument(
        "--no-channel-search",
        action="store_true",
        help="Do not search inside channels that previously delivered this artist.",
    )
    p.add_argument(
        "--backfill-channels",
        action="store_true",
        help="Resolve and store the publishing channel of past downloads so the "
        "learned channel trust model is active on this run.",
    )

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
    decision.add_argument(
        "--decider",
        action="store_true",
        help=(
            "Use a local Decider server to choose the candidate for each song. Speaks "
            "the same System One API as Kev, and unlike Kev it also runs on Apple "
            "Silicon and CPU."
        ),
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
        default="jaredpalmer/kev-4b@v1.0",
        help=(
            "Checkpoint to download and serve. Pinned to @v1.0 so the weights are "
            "reproducible; an unpinned name tracks the branch (default: %(default)s)."
        ),
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
        "--decider-url",
        metavar="URL",
        default="http://127.0.0.1:8010",
        help="Decider server URL (default: %(default)s).",
    )
    p.add_argument(
        "--decider-model",
        metavar="MODEL",
        default=None,
        help=(
            "Checkpoint name to send in the request body. Left unset, the name is read "
            "from the running server, which is the reliable option: decider.serve names "
            "the checkpoint after the version in its own config."
        ),
    )
    p.add_argument(
        "--decider-run",
        metavar="CHECKPOINT",
        default="Mapika/decider-4b@v2",
        help=(
            "Checkpoint to download and serve, as repo_id or repo_id@revision. Pinned "
            "to a tag so the weights are reproducible; an unpinned name tracks the "
            "branch. Note the tag is @v2, not @v2.1: the model card's weights version "
            "and the Hub tag are different numbers, and only the tag resolves "
            "(default: %(default)s). Smaller: Mapika/decider-2b. Largest and best on "
            "a big GPU: Mapika/decider-12b."
        ),
    )
    p.add_argument(
        "--decider-dir",
        metavar="PATH",
        type=Path,
        default=Path.home() / ".cache" / "ytdl" / "decider",
        help="Decider virtual environment directory (default: %(default)s).",
    )
    p.add_argument(
        "--decider-port",
        metavar="INT",
        type=int,
        default=8010,
        help="Local Decider server port (default: %(default)s).",
    )
    p.add_argument(
        "--decider-startup-timeout",
        metavar="INT",
        type=int,
        default=900,
        help="Seconds to wait for Decider startup (default: %(default)s).",
    )
    p.add_argument(
        "--decider-package",
        metavar="SPEC",
        default=_DECIDER_DEFAULT_PACKAGE,
        help=(
            "Python package providing decider.serve, pinned for reproducibility "
            "(default: %(default)s)."
        ),
    )
    p.add_argument(
        "--decider-skip-install",
        action="store_true",
        help="Do not create or update the Decider virtual environment.",
    )
    p.add_argument(
        "--decider-device",
        metavar="DEVICE",
        default=None,
        help=(
            "Device for decider.serve: cuda, cuda:1, mps, cpu. Unset lets it choose "
            "(CUDA, else MPS, else CPU), which is what you want unless you know why "
            "not."
        ),
    )
    p.add_argument(
        "--decider-graphs-t",
        metavar="LIST",
        default=_DECIDER_DEFAULT_GRAPH_T_BUCKETS,
        help=(
            "Length buckets for decider.serve's CUDA-graph grid (DECIDER_T_BUCKETS, "
            "comma-separated ints, empty string leaves decider.serve's default). The "
            "default narrows the grid from 89 captured graphs / ~25 GB VRAM on a "
            "4B checkpoint to 5 graphs / seconds of start-up; our client is strictly "
            "serial (one request in flight, short states), so larger batch and "
            "length cells would only sit unused on the GPU."
        ),
    )
    p.add_argument(
        "--decider-graphs-b",
        metavar="LIST",
        default=_DECIDER_DEFAULT_GRAPH_B_BUCKETS,
        help=(
            "Batch buckets for decider.serve's CUDA-graph grid "
            "(DECIDER_B_BUCKETS, comma-separated ints, empty string leaves decider.serve "
            "default 1,2,4,8,16,32). Decider's default is 1 because our client is "
            "serial; raise --decision-in-flight first, and only then touch this."
        ),
    )
    fused_group = p.add_mutually_exclusive_group()
    fused_group.add_argument(
        "--decider-kernels",
        dest="decider_kernels",
        action="store_const",
        const=True,
        default=None,
        help="Insist on the GDN Triton kernels without probing.",
    )
    fused_group.add_argument(
        "--no-decider-kernels",
        dest="decider_kernels",
        action="store_const",
        const=False,
        help="Decline the GDN Triton kernels; PyTorch's reference path is used instead.",
    )
    fused = p.add_mutually_exclusive_group()
    fused.add_argument(
        "--kev-fused",
        dest="kev_fused",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Insist on Kev's fused Qwen3.5 kernels without probing. They are worth "
            "about a third of the GPU time per batch, but some Triton/platform "
            "combinations import cleanly and then never return, so the default is to "
            "prove they answer first."
        ),
    )
    fused.add_argument(
        "--no-kev-fused",
        dest="kev_fused",
        action="store_const",
        const=False,
        help="Decline Kev's fused Qwen3.5 kernels without probing.",
    )
    p.add_argument(
        "--decision-threshold",
        "--jev-threshold",
        "--kev-threshold",
        dest="decision_threshold",
        metavar="FLOAT",
        type=float,
        default=_CONFIG.DECISION_DEFAULT_THRESHOLD,
        help=(
            "Minimum average probability required to download, for whichever decision "
            "model is in use (default: %(default)s). --jev-threshold and --kev-threshold "
            "are the former names and still work."
        ),
    )
    p.add_argument(
        "--decision-calibration-log",
        dest="decision_calibration_log",
        metavar="PATH",
        type=str,
        default=_CONFIG.DECISION_CALIBRATION_LOG,
        help=(
            "Append one JSONL record per song with every gate score, the heuristic "
            "score and what was selected. Use it to retune the gate floors against "
            "real data instead of against a sample; empty (the default) writes nothing."
        ),
    )
    p.add_argument(
        "--decision-runs",
        "--jev-runs",
        "--kev-runs",
        dest="decision_runs",
        metavar="INT",
        type=int,
        default=_CONFIG.DECISION_DEFAULT_RUNS,
        help=(
            "Evaluations per song (default: %(default)s). Above 1 this measures how "
            "steady the model is rather than asking it again for a different answer: "
            "these models are prefilled deterministically, so the extra passes "
            "essentially repeat the same forward pass and the verdict does not change. "
            "What it does buy is the spread across runs, which is what marks a "
            "candidate unstable enough to review. Raise --decision-questions or lower "
            "--decision-timeout before you raise this. --jev-runs and --kev-runs are "
            "the former names and still work."
        ),
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

    if args.max_duration is not None and args.max_duration < 0:
        p.error("--max-duration must be zero or greater")
    if args.min_duration is not None and args.min_duration < 0:
        p.error("--min-duration must be zero or greater")
    if args.review_only_suspects and not args.review:
        p.error("--review-only-suspects requires --review")
    if args.review and (args.verify or args.repair):
        p.error("--review cannot be combined with --verify or --repair")
    if not 1 <= args.kev_port <= 65535:
        p.error("--kev-port must be between 1 and 65535")
    if not 1 <= args.decider_port <= 65535:
        p.error("--decider-port must be between 1 and 65535")
    if args.kev_startup_timeout < 1:
        p.error("--kev-startup-timeout must be greater than 0")
    if args.decider_startup_timeout < 1:
        p.error("--decider-startup-timeout must be greater than 0")
    if not 0 < args.decision_threshold <= 1:
        p.error("--decision-threshold must be greater than 0 and at most 1")
    if not 1 <= args.decision_runs <= _CONFIG.DECISION_MAX_RUNS:
        p.error(f"--decision-runs must be between 1 and {_CONFIG.DECISION_MAX_RUNS}")
    if args.decision_timeout < 1:
        p.error("--decision-timeout must be greater than 0")
    if args.decision_in_flight < 1:
        p.error("--decision-in-flight must be at least 1")
    _reject_provider_conflicts(p, args)
    _warn_deprecated_flags(argv)

    args.workers = max(1, min(args.workers, _CONFIG.MAX_WORKERS))
    raw_sources = (
        args.sources.split(",")
        if isinstance(args.sources, str)
        else [str(source) for source in args.sources]
    )
    args.sources = [source.strip().lower() for source in raw_sources if source.strip()]
    return args
