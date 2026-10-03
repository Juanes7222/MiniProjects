"""
MusicDownloader — the library-facing core class.

This module contains NO Rich / CLI code. All user-facing output is delegated
to the DownloaderEvents instance supplied at construction time.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
from collections import Counter
from dataclasses import dataclass, field
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yt_dlp
from rapidfuzz import fuzz

from .channels import ChannelTrust, channel_url_for
from .config import Config
from .downloader import download_partial, execute_download
from .events import DownloaderEvents
from .fingerprint import (
    AcoustIDCircuitBreaker,
    FingerprintCache,
    FingerprintVerdict,
    release_fingerprint_slot,
    verify_fingerprint,
)
from .jev import JevClassifier, JevEvaluationError
from .kev import KevClassifier
from .metadata import CatalogContext, fetch_itunes_reference, fetch_musicbrainz
from .pipeline import StagePipeline
from .post_checks import check_duration, check_silence, embed_and_verify
from .reports import export_report, update_json_file
from .result import DownloadResult
from .search import search_all_sources, select_best_result
from .state import load_state, merge_state_detail, save_state, state_detail
from .statewriter import CoalescingStateWriter
from .utils import (
    apply_delay,
    compute_md5,
    migrate_legacy_audio_path,
    normalize_title,
    sanitize_filename,
    strip_featuring,
)
from .verifier import verify_library as _verify_library
from .ytdlp_options import build_ytdlp_base_opts, make_progress_hook, resolve_downloaded_file


def _resolve_stage_workers(configured: int, fallback: int) -> int:
    """Auto-size a pipeline stage when the config leaves it at 0."""
    return max(1, int(configured) if configured else fallback)


@dataclass
class _SongJob:
    """One song's state, carried through the pipeline stages.

    The stages of a song are the same code whether they run back-to-back (the
    single-song path) or spread across the pipeline's queues (the batch path).
    Splitting the work into stages and threading this object through them means
    there is one implementation of "download a song", not two that drift apart.
    """

    artist: str
    song: str
    output_dir: Path
    fmt: str
    quality: str
    skip_existing: bool
    state: dict
    state_lock: threading.Lock
    stop_event: threading.Event
    seen: set
    seen_lock: threading.Lock
    artist_counts: dict

    result: DownloadResult = field(init=False)
    key: str = field(init=False)
    safe_a: str = field(init=False)
    safe_s: str = field(init=False)

    mb: Optional[dict] = None
    best: Optional[dict] = None
    ranked: list = field(default_factory=list)
    src: Optional[str] = None
    url: str = ""
    dur_s: int = 0
    dl_file: Optional[Path] = None

    def __post_init__(self) -> None:
        self.result = DownloadResult(artist=self.artist, song=self.song)
        self.key = f"{self.artist}::{self.song}"
        self.safe_a = sanitize_filename(self.artist)
        self.safe_s = sanitize_filename(self.song)


class MusicDownloader:
    def __init__(
        self,
        config=None,
        events=None,
        acoustid_key=None,
        force_fingerprint=False,
        skip_fingerprint=False,
        require_fingerprint=False,
        no_silence_check=False,
        score_threshold=None,
        sources=None,
        workers=2,
        delay=(2.0, 5.0),
        max_results=5,
        fuzzy_threshold=65,
        max_duration=None,
        min_duration=None,
        musicbrainz=False,
        cookies_browser=None,
        cookies_file=None,
        proxy=None,
        use_jev=False,
        use_kev=False,
        jev_threshold=None,
        jev_runs=None,
        jev_classifier=None,
        kev_threshold=None,
        kev_runs=None,
        kev_url="http://127.0.0.1:8009",
        kev_model="kev-latest",
        kev_classifier=None,
        channel_search=True,
    ):
        if require_fingerprint and not acoustid_key:
            raise ValueError("Strict fingerprint verification requires an AcoustID key")
        if require_fingerprint and skip_fingerprint:
            raise ValueError("Strict fingerprint verification cannot be skipped")
        fpcalc_available = shutil.which("fpcalc") is not None
        if require_fingerprint and not fpcalc_available:
            raise RuntimeError("Strict fingerprint verification requires fpcalc on PATH")
        if use_jev and use_kev:
            raise ValueError("Jev and Kev cannot be enabled at the same time")

        self.config = config or Config()
        self.events = events or DownloaderEvents()
        self.acoustid_key = acoustid_key
        self.force_fingerprint = force_fingerprint
        self.skip_fingerprint = skip_fingerprint
        self.require_fingerprint = require_fingerprint
        self.no_silence_check = no_silence_check
        self.score_threshold = (
            score_threshold if score_threshold is not None else self.config.SCORE_THRESHOLD_REJECT
        )
        self.sources = list(sources) if sources is not None else list(self.config.DEFAULT_SOURCES)
        self.workers = max(1, min(workers, self.config.MAX_WORKERS))
        self.delay = delay
        self.max_results = max_results
        self.fuzzy_threshold = fuzzy_threshold
        self.max_duration = (
            max_duration if max_duration is not None else self.config.MAX_DURATION_SECONDS
        )
        self.max_duration_explicit = max_duration is not None
        self.min_duration = (
            min_duration if min_duration is not None else self.config.MIN_DURATION_SECONDS
        )
        self.min_duration_explicit = min_duration is not None
        self.musicbrainz = musicbrainz
        self.cookies_browser = cookies_browser
        self.cookies_file = cookies_file
        self.proxy = proxy
        self.use_jev = use_jev
        self.use_kev = use_kev
        self.decision_provider = "kev" if use_kev else "jev" if use_jev else "heuristic"
        self.jev_threshold = (
            jev_threshold if jev_threshold is not None else self.config.JEV_DEFAULT_THRESHOLD
        )
        self.kev_threshold = (
            kev_threshold if kev_threshold is not None else self.config.JEV_DEFAULT_THRESHOLD
        )
        self.jev_runs = jev_runs if jev_runs is not None else self.config.JEV_DEFAULT_RUNS
        self.kev_runs = kev_runs if kev_runs is not None else self.config.JEV_DEFAULT_RUNS
        self.decision_threshold = self.kev_threshold if use_kev else self.jev_threshold
        self.decision_runs = self.kev_runs if use_kev else self.jev_runs
        gate_floor = float(self.config.DECISION_GATE_FLOOR)
        stable_spread = float(self.config.DECISION_STABLE_SPREAD)
        max_candidates = int(self.config.DECISION_MAX_CANDIDATES)
        self.decision_classifier = kev_classifier or (
            KevClassifier(
                url=kev_url,
                model=kev_model,
                threshold=self.decision_threshold,
                gate_floor=gate_floor,
                stable_spread=stable_spread,
                max_candidates=max_candidates,
                timeout_seconds=self.config.DECISION_TIMEOUT_SECONDS,
                max_in_flight=self.config.DECISION_MAX_IN_FLIGHT,
                failure_threshold=self.config.DECISION_FAILURE_THRESHOLD,
                max_questions=self.config.DECISION_MAX_QUESTIONS,
                eval_headroom=self.config.DECISION_HEADROOM,
            )
            if use_kev
            else jev_classifier
            or (
                JevClassifier(
                    threshold=self.decision_threshold,
                    gate_floor=gate_floor,
                    stable_spread=stable_spread,
                    max_candidates=max_candidates,
                )
                if use_jev
                else None
            )
        )
        self.fpcalc_available = fpcalc_available
        self.channel_search = channel_search
        # Rebuilt from disk at the start of every run; see _load_channel_trust.
        self.channel_trust = ChannelTrust(self.config)
        # Bounds how many 90-second partials may be in flight at once. Sized so
        # the AcoustID budget, not this semaphore, is what throttles the run:
        # three concurrent partials at a few seconds each reaches well under the
        # three-per-second ceiling, so the old value of 3 was the real limit and
        # the published rate limit was never the constraint it appeared to be.
        self._fp_semaphore = threading.Semaphore(
            max(1, int(self.config.FP_CONCURRENCY))
        )
        self._circuit_breaker = AcoustIDCircuitBreaker(cooldown_seconds=60.0)
        self._selection_lock = threading.Lock()
        # Remembers AcoustID verdicts across runs and coalesces concurrent
        # duplicate checks within one.
        self._fingerprint_cache = FingerprintCache(self.config)
        # Answers every song of an artist from one release tracklist instead of
        # one MusicBrainz search per song.
        self._catalog = CatalogContext(self.config, musicbrainz=self.musicbrainz)
        # Injected by download_batch so a batch's stages do not each build their
        # own pool. None keeps the single-song path pool-free.
        self._search_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._alternate_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Set by download_batch so persist() coalesces onto a background writer
        # instead of rewriting the whole file under the shared lock.
        self._state_writer: Optional[CoalescingStateWriter] = None

    # -- shared executors ----------------------------------------------------

    def _alternate_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        pool = self._alternate_pool
        if pool is None:
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, int(self.config.FP_ALTERNATE_FANOUT)),
                thread_name_prefix="ytdl-fp-alternate",
            )
            self._alternate_pool = pool
        return pool

    def _close_executors(self) -> None:
        for attribute in ("_alternate_pool", "_search_executor"):
            pool = getattr(self, attribute, None)
            if pool is not None:
                pool.shutdown(wait=False)
                setattr(self, attribute, None)

    def _load_channel_trust(self, state):
        """(Re)build the learned channel trust model from the download state."""
        self.channel_trust = ChannelTrust.from_state(state, self.config)
        return self.channel_trust

    def _resolve_reference(self, mb_data, artist, song, allow_itunes=True):
        """
        Build a trustworthy duration reference for this song.

        MusicBrainz is asked first, then the iTunes catalogue. Either can return
        a same-titled recording by a *different* artist, whose duration is that
        recording's and not ours -- and applying a -35 "duration mismatch" to
        every correct candidate on the strength of someone else's recording
        actively demotes the right answer. Both sources are therefore checked
        against the requested title and artist, and a mismatch is surfaced to
        the user because it usually means the song list itself is wrong
        ("Con el alma en las manos" is a Jesús Manuel recording, not Miguel
        Morales; "Coqueta" is Heredero, not Jorge Veloza).

        Returns ``(reference_dict_or_None, warnings)`` where reference_dict has
        ``title``, ``artist``, ``album``, ``duration``, ``source``.
        """
        warnings: list[str] = []
        sources = [("MusicBrainz", mb_data)]
        if allow_itunes:
            sources.append(("iTunes", None))
        for source, payload in sources:
            if source == "iTunes":
                try:
                    payload = fetch_itunes_reference(artist, song)
                except Exception:
                    payload = None
            if not payload:
                continue

            duration = payload.get("duration_seconds" if source == "MusicBrainz" else "duration")
            ref_title = payload.get("title")
            ref_artist = payload.get("artist")
            if not ref_artist and source == "iTunes":
                ref_artist = payload.get("artist")

            if ref_artist:
                query = normalize_title(strip_featuring(artist))
                matched = normalize_title(strip_featuring(ref_artist))
                if (
                    matched
                    and fuzz.token_set_ratio(query, matched)
                    < self.config.MB_REFERENCE_MIN_ARTIST_MATCH
                ):
                    warnings.append(
                        f"{source}: '{ref_title or song}' is credited to {ref_artist}, "
                        f"not {artist} -- ignoring its reference duration. "
                        "Check the artist/song in your list."
                    )
                    continue

            if ref_title and song:
                query_song = normalize_title(strip_featuring(song))
                matched_song = normalize_title(strip_featuring(ref_title))
                if (
                    matched_song
                    and fuzz.token_set_ratio(query_song, matched_song)
                    < self.config.MB_REFERENCE_MIN_TITLE_MATCH
                ):
                    continue

            if duration:
                return (
                    {
                        "title": ref_title,
                        "artist": ref_artist,
                        "album": payload.get("album") or payload.get("album"),
                        "duration": int(duration),
                        "source": source,
                    },
                    warnings,
                )
        return None, warnings

    def download(self, artist, song, output_dir, fmt="mp3", quality="192", skip_existing=False):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(output_dir, self.config.STATE_FILE)
        self._load_channel_trust(state)
        return self._process_song(
            artist,
            song,
            output_dir,
            fmt,
            quality,
            skip_existing,
            state,
            threading.Lock(),
            threading.Event(),
            set(),
            threading.Lock(),
            {artist: 1},
        )

    def _pipeline_specs(self) -> list[tuple[str, int]]:
        """Stage sizes, each sized against the resource that stage contends for.

        Search and download are network-bound and want more threads than cores.
        Verification is bounded by the AcoustID budget and by how many 90-second
        partials are sensible in flight. The post-download stage decodes audio
        and tags files, so it wants about one worker per core -- oversubscribing
        it does not make it faster, it makes the transcodes running beside it
        slower too.
        """
        cpus = os.cpu_count() or 4
        return [
            ("search", _resolve_stage_workers(self.config.SEARCH_WORKERS, self.workers)),
            (
                "verify",
                _resolve_stage_workers(self.config.VERIFY_WORKERS, self.config.FP_CONCURRENCY),
            ),
            ("download", _resolve_stage_workers(self.config.DOWNLOAD_WORKERS, self.workers)),
            ("finalize", _resolve_stage_workers(self.config.POST_WORKERS, cpus)),
        ]

    def _run_stage(self, job: "_SongJob", stage: str) -> bool:
        """Run one stage. False means the song's fate is settled here."""
        if stage == "search":
            return self._stage_select(job)
        if stage == "verify":
            return self._stage_verify(job)
        if stage == "download":
            return self._stage_fetch(job)
        self._stage_finalize(job)
        return False

    def download_batch(
        self,
        songs,
        output_dir,
        fmt="mp3",
        quality="192",
        skip_existing=False,
        report_formats=None,
        update_json_path=None,
    ):
        """Download a batch as a stage-bounded pipeline.

        The stages -- search, verify, download, finalize -- each get their own
        pool sized against their own bottleneck, and hand work on through bounded
        queues. A slow stage therefore pushes back on the stage feeding it rather
        than letting the whole batch queue up behind it, and the batch never has
        every song's search, download and decode in flight at once.

        Results are reported as they complete but returned in the order the songs
        were requested, so a report for a 500-song batch is reproducible.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = [(a, s) for a, lst in songs.items() for s in lst if lst]
        self.events.on_session_start(len(pairs))
        artist_counts = dict(Counter(artist for artist, _ in pairs))
        state = load_state(output_dir, self.config.STATE_FILE)
        state_lock = threading.Lock()
        self._load_channel_trust(state)
        stop = threading.Event()
        seen, seen_lock = set(), threading.Lock()

        all_results: list[DownloadResult] = []
        results_lock = threading.Lock()
        slots: dict[str, DownloadResult] = {}
        start = time.monotonic()

        jobs = [
            _SongJob(
                artist=artist,
                song=song,
                output_dir=output_dir,
                fmt=fmt,
                quality=quality,
                skip_existing=skip_existing,
                state=state,
                state_lock=state_lock,
                stop_event=stop,
                seen=seen,
                seen_lock=seen_lock,
                artist_counts=artist_counts,
            )
            for artist, song in pairs
        ]

        # One background writer for the whole run: state writes are coalesced off
        # the lock that every stage shares.
        writer = CoalescingStateWriter(
            state,
            output_dir,
            self.config.STATE_FILE,
            flush_interval=self.config.STATE_FLUSH_INTERVAL,
            flush_batch=self.config.STATE_FLUSH_BATCH,
        ).start()
        self._state_writer = writer

        def _publish(job: _SongJob) -> None:
            with results_lock:
                all_results.append(job.result)
                slots[job.key] = job.result
            self.events.on_result(job.result)

        def _on_stage_error(job: "_SongJob", error: BaseException) -> None:
            if isinstance(job, _SongJob):
                job.result.status = "failed"
                job.result.reason = f"{type(error).__name__}: {error}"
                _publish(job)
            else:  # pragma: no cover - defensive
                self.events.on_warn(f"Pipeline stage error: {type(error).__name__}: {error}")

        try:
            if not self.config.PIPELINE_ENABLED or len(jobs) <= 1:
                # A one-song batch gains nothing from a pipeline, and running it
                # sequentially keeps interrupt handling and ordering trivial.
                for job in jobs:
                    if stop.is_set():
                        break
                    try:
                        self._process_song(
                            job.artist,
                            job.song,
                            job.output_dir,
                            job.fmt,
                            job.quality,
                            job.skip_existing,
                            job.state,
                            job.state_lock,
                            job.stop_event,
                            job.seen,
                            job.seen_lock,
                            job.artist_counts,
                        )
                    except Exception as error:  # noqa: BLE001 - reported per song
                        job.result.status = "failed"
                        job.result.reason = f"{type(error).__name__}: {error}"
                    _publish(job)
            else:
                self._search_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, self.config.SEARCH_WORKERS or self.workers),
                    thread_name_prefix="ytdl-search",
                )
                self._prime_catalogs(jobs)
                specs = self._pipeline_specs()

                def _make_stage(name: str):
                    def _run(job: _SongJob):
                        if self._run_stage(job, name):
                            return job
                        # Settled here -- downloaded, or failed, or skipped. Every
                        # song produces exactly one result however it ends, so it
                        # is published on the way out rather than by the last
                        # stage, which most songs never reach.
                        _publish(job)
                        return None

                    return _run

                pipeline = StagePipeline[_SongJob](
                    [
                        (name, _make_stage(name), workers) for name, workers in specs
                    ],
                    queue_depth=self.config.PIPELINE_QUEUE_DEPTH,
                    on_error=_on_stage_error,
                    stop_event=stop,
                )
                pipeline.run(jobs)
        except KeyboardInterrupt:
            stop.set()
            self.events.on_interrupted(len(all_results), len(pairs), time.monotonic() - start)
        finally:
            # Flush synchronously before anything else: the caller reads this file
            # as soon as download_batch returns.
            writer.flush()
            writer.close()
            self._state_writer = None
            self._close_executors()

        elapsed = time.monotonic() - start
        # Deterministic order, regardless of which stage happened to finish first.
        ordered = [slots.get(job.key, job.result) for job in jobs]
        self.events.on_session_complete(ordered, elapsed)
        if report_formats:
            export_report([r.to_dict() for r in ordered], output_dir, report_formats)
        if update_json_path:
            update_json_file(update_json_path, [r.to_dict() for r in ordered])
        return ordered

    def _prime_catalogs(self, jobs: list["_SongJob"]) -> None:
        """Load each artist's album before its songs start searching.

        One throttled catalogue lookup per artist, overlapping with other artists'
        work, instead of one per song sitting on a single song's critical path.
        Only worth doing when the song list actually names several artists.
        """
        if not self.musicbrainz:
            return
        by_artist: dict[str, list[str]] = {}
        for job in jobs:
            by_artist.setdefault(job.artist, []).append(job.song)
        if len(by_artist) < 2:
            return

        def _prime(artist: str, first_song: str) -> None:
            try:
                self._catalog.prime(artist, first_song, fetch=fetch_musicbrainz)
            except Exception:
                # Warming the cache is best-effort; a failure just means those
                # songs use the per-song lookup they always used.
                pass

        workers = min(len(by_artist), max(1, self.workers))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ytdl-catalog"
        ) as pool:
            # Submit every artist first, then collect. Collecting inside a
            # generator expression would submit-then-immediately-block on each
            # one in turn, which serialises exactly the work being warmed up.
            pending = [
                pool.submit(_prime, artist, songs[0]) for artist, songs in by_artist.items()
            ]
            for future in pending:
                try:
                    future.result()
                except Exception:
                    pass

    def download_url(
        self,
        url,
        output_dir,
        fmt="mp3",
        quality="192",
        max_downloads=None,
        skip_existing=False,
        match_title=None,
        reject_title=None,
    ):
        self.download_url_results(
            url,
            output_dir,
            fmt,
            quality,
            max_downloads,
            skip_existing,
            match_title,
            reject_title,
        )

    def download_url_results(
        self,
        url,
        output_dir,
        fmt="mp3",
        quality="192",
        max_downloads=None,
        skip_existing=False,
        match_title=None,
        reject_title=None,
    ) -> list[DownloadResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        match_re = re.compile(match_title, re.IGNORECASE) if match_title else None
        reject_re = re.compile(reject_title, re.IGNORECASE) if reject_title else None
        scan_opts = build_ytdlp_base_opts(
            output_dir,
            fmt,
            quality,
            quiet=False,
            no_warnings=False,
            progress_hook=None,
            skip_existing=False,
            max_downloads=max_downloads,
            cookies_browser=self.cookies_browser,
            cookies_file=self.cookies_file,
            proxy=self.proxy,
            enable_remote_components=True,
            youtube_player_clients=list(self.config.YOUTUBE_PLAYER_CLIENTS),
            noplaylist=False,
            for_scan=True,
        )
        try:
            with yt_dlp.YoutubeDL(scan_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            self.events.on_download_failed("URL", url, str(exc))
            return []
        if not info:
            return []
        is_single_video = not info.get("entries")
        if is_single_video:
            max_duration = 0 if not self.max_duration_explicit else self.max_duration
            min_duration = 0 if not self.min_duration_explicit else self.min_duration
        else:
            max_duration = self.max_duration
            min_duration = self.min_duration
            if not self.max_duration_explicit or not self.min_duration_explicit:
                hints = []
                if not self.max_duration_explicit:
                    hints.append(
                        f"longer than {self.max_duration}s are skipped "
                        f"(pass --max-duration 0 to include them)"
                    )
                if not self.min_duration_explicit:
                    hints.append(
                        f"shorter than {self.min_duration}s are skipped "
                        f"(pass --min-duration 0 to include them)"
                    )
                self.events.on_warn(f"Playlist/channel: entries {' and '.join(hints)}.")
        entries = list(self._iter_entries(info))
        urls = []
        for e in entries:
            if not e:
                continue
            title = e.get("title", "Unknown")
            uploader = e.get("uploader", "Unknown")
            if match_re and not match_re.search(title):
                self.events.on_warn(f"Skipped (no match): {title}")
                continue
            if reject_re and reject_re.search(title):
                self.events.on_warn(f"Skipped (rejected): {title}")
                continue
            if e.get("is_live"):
                self.events.on_warn(f"Skipped (live): {title}")
                continue
            dur = e.get("duration")
            if dur is not None:
                if min_duration and dur < min_duration:
                    self.events.on_warn(f"Skipped (too short): {title}")
                    continue
                if max_duration and dur > max_duration:
                    self.events.on_warn(f"Skipped (too long): {title}")
                    continue
            iu = e.get("webpage_url") or e.get("url")
            if not iu and e.get("id"):
                iu = f"https://www.youtube.com/watch?v={e['id']}"
            if is_single_video:
                # The requested URL *is* this entry — never drop it.
                if iu:
                    urls.append((uploader, title, iu))
                continue
            if iu and iu != url and "search?" not in iu:
                urls.append((uploader, title, iu))
        if not urls:
            self.events.on_warn("No suitable URLs found.")
            return []
        self.events.on_session_start(len(urls))
        all_results = []
        lock = threading.Lock()
        stop = threading.Event()
        start = time.monotonic()

        def _dl_one(ia, it, iu):
            if stop.is_set():
                return
            result = DownloadResult(artist=ia, song=it)
            target_file = migrate_legacy_audio_path(
                output_dir / sanitize_filename(ia) / f"{sanitize_filename(it)}.{fmt}"
            )
            progress_hook = make_progress_hook(self.events, ia, it)
            options = build_ytdlp_base_opts(
                output_dir=output_dir,
                fmt=fmt,
                quality=quality,
                quiet=True,
                no_warnings=True,
                progress_hook=progress_hook,
                skip_existing=skip_existing,
                cookies_browser=self.cookies_browser,
                cookies_file=self.cookies_file,
                proxy=self.proxy,
                enable_remote_components=True,
                youtube_player_clients=list(self.config.YOUTUBE_PLAYER_CLIENTS),
                noplaylist=True,
                output_template=target_file.with_suffix(".%(ext)s"),
                embed_thumbnail=True,
            )
            if skip_existing and target_file.exists():
                self.events.on_skip_existing(ia, it, target_file, True)
                result.status = "skipped"
                result.reason = "File exists"
                result.file_path = target_file
                with lock:
                    all_results.append(result)
                self.events.on_result(result)
                return
            self.events.on_download_start(ia, it, iu)
            try:
                with yt_dlp.YoutubeDL(options) as downloader:
                    info = downloader.extract_info(iu, download=True)
                    if not info:
                        raise RuntimeError("no info_dict")
                    downloaded_file = resolve_downloaded_file(
                        Path(downloader.prepare_filename(info)), fmt
                    )
                    if downloaded_file is None:
                        raise FileNotFoundError("no output file")
                    result.status = "downloaded"
                    result.file_path = downloaded_file
                    result.duration_seconds = info.get("duration")
            except Exception as exc:
                self.events.on_download_failed(ia, it, str(exc))
                result.status = "failed"
                result.reason = str(exc)
            with lock:
                all_results.append(result)
            self.events.on_result(result)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                fs = [pool.submit(_dl_one, a, t, u) for a, t, u in urls]
                for f in concurrent.futures.as_completed(fs):
                    try:
                        f.result()
                    except Exception:
                        pass
        except KeyboardInterrupt:
            stop.set()
        self.events.on_session_complete(all_results, time.monotonic() - start)
        return all_results

    def verify_library(self, songs, output_dir, fmt="mp3"):
        output_dir = Path(output_dir)
        state = load_state(output_dir, self.config.STATE_FILE)
        state_lock = threading.Lock()
        # A verify pass rewrites the same whole-file document once per song, so
        # it needs the same coalescing the batch path uses; otherwise
        # re-verifying a large library pays O(N^2) writes under a shared lock.
        writer = CoalescingStateWriter(
            state,
            output_dir,
            self.config.STATE_FILE,
            flush_interval=self.config.STATE_FLUSH_INTERVAL,
            flush_batch=self.config.STATE_FLUSH_BATCH,
        ).start()
        self._state_writer = writer
        try:
            return _verify_library(
                songs,
                output_dir,
                fmt,
                self.workers,
                self.acoustid_key,
                self.config,
                self._circuit_breaker,
                self._fp_semaphore,
                self.musicbrainz,
                self.events,
                self._persist,
                state,
                state_lock,
                require_fingerprint=self.require_fingerprint,
                state_filename=self.config.STATE_FILE,
            )
        finally:
            writer.flush()
            writer.close()
            self._state_writer = None

    def _process_song(
        self,
        artist,
        song,
        output_dir,
        fmt,
        quality,
        skip_existing,
        state,
        state_lock,
        stop_event,
        seen,
        seen_lock,
        artist_counts,
    ):
        """Run every stage of one song back to back.

        This is the single-song path, and the sequential reference the batch
        pipeline is measured against. The batch drives these same stage methods,
        so a song behaves identically either way.
        """
        job = _SongJob(
            artist=artist,
            song=song,
            output_dir=Path(output_dir),
            fmt=fmt,
            quality=quality,
            skip_existing=skip_existing,
            state=state,
            state_lock=state_lock,
            stop_event=stop_event,
            seen=seen,
            seen_lock=seen_lock,
            artist_counts=artist_counts,
        )
        if not self._stage_select(job):
            return job.result
        if not self._stage_verify(job):
            return job.result
        if not self._stage_fetch(job):
            return job.result
        self._stage_finalize(job)
        return job.result

    # -- stage 1: catalogue reference, search, rank, decide -------------------

    def _stage_select(self, job: _SongJob) -> bool:
        """Everything up to and including choosing a candidate. False = stop."""
        artist, song = job.artist, job.song
        result = job.result

        with job.seen_lock:
            if artist not in job.seen:
                job.seen.add(artist)
                self.events.on_artist_start(artist, job.artist_counts.get(artist, 0))

        if job.stop_event.is_set():
            result.status = "skipped"
            result.reason = "Interrupted"
            return False

        expected = migrate_legacy_audio_path(
            job.output_dir / job.safe_a / f"{job.safe_s}.{job.fmt}"
        )
        with job.state_lock:
            existing = job.state.get("downloads", {}).get(job.key)
        if job.skip_existing and existing and existing.get("status") == "downloaded":
            md5s = existing.get("md5")
            if expected.exists():
                if md5s:
                    if compute_md5(expected) == md5s:
                        self.events.on_skip_existing(artist, song, expected, True)
                        result.status = "skipped"
                        result.file_path = expected
                        result.md5 = md5s
                        return False
                    self.events.on_md5_mismatch(artist, song)
                else:
                    self.events.on_skip_existing(artist, song, expected, False)
                    result.status = "skipped"
                    result.file_path = expected
                    return False

        # An explicit --delay is still honoured, but the default is no tax at
        # all. Request pacing is not missing here: every source paces itself at
        # its own call site from per-service token buckets (see ratelimit), which
        # block only when a budget is genuinely exhausted instead of charging
        # every song a fixed toll -- and which also pace the several extractions
        # one song performs, not just the song itself.
        if self.delay[1] > 0:
            apply_delay(self.delay[0], self.delay[1])

        job.mb = self._resolve_catalog(artist, song)
        if self.musicbrainz:
            self.events.on_musicbrainz_result(artist, song, bool(job.mb), job.mb or {})

        best, ranked, src = self._search_and_select(
            artist,
            song,
            job.output_dir,
            job.state,
            job.state_lock,
            job.key,
            result,
            job.stop_event,
            job.mb,
            executor=self._search_executor,
        )
        job.best, job.ranked, job.src = best, ranked, src
        if best is None:
            return False

        job.url = best.get("webpage_url") or best.get("url", "")
        job.dur_s = int(best.get("duration") or 0)
        result.source = src
        result.url = job.url
        result.matched_title = best.get("title") or ""
        result.fuzzy_score = int(
            fuzz.token_sort_ratio(f"{artist} {song}".lower(), (best.get("title") or "").lower())
        )
        result.duration_seconds = job.dur_s
        result.heuristic_score = int(best.get("_heuristic_score", best.get("_composite_score", 0)))
        result.composite_score = best.get("_composite_score", 0)
        result.score_breakdown = best.get("_score_breakdown", {})
        self._apply_decision(result, best)
        result.candidates_ranked = len(ranked)
        if result.selection_method == "heuristic":
            result.selection_method = self.decision_provider
        return True

    def _resolve_catalog(self, artist: str, song: str) -> Optional[dict]:
        """Catalogue data for a song, from its album's tracklist when available.

        One MusicBrainz search answers one song, and the API is throttled to one
        request per second process-wide -- which used to serialise a whole album
        behind fifteen sequential lookups. An artist's songs are usually an
        album, so once any of them resolves to a release the rest are read off
        that release's tracklist: two throttled calls for the album instead of
        one per song, and an exact tracklist rather than a search-ranked guess.
        """
        if not self.musicbrainz:
            return None

        from_tracklist = self._catalog.from_tracklist(artist, song)
        if from_tracklist is not None:
            return from_tracklist

        data = fetch_musicbrainz(artist, song)
        if data and data.get("release_id"):
            self._catalog.remember_release(artist, data["release_id"])
        return data

    # -- stage 2: fingerprint the leading candidate ---------------------------

    def _stage_verify(self, job: _SongJob) -> bool:
        """Fingerprint check plus the strict-verification gate. False = stop."""
        result, best = job.result, job.best
        artist, song = job.artist, job.song

        fp_ok, fp_conf, fp_title, fp_label = self._fingerprint_check(
            artist, song, job.url, job.output_dir, best, job.ranked, result
        )
        # Reported on the same scale the gate uses, so the line the user reads
        # ("high confidence") and the reason fingerprinting was skipped agree.
        sc = result.heuristic_score
        self.events.on_verification_status(
            artist,
            song,
            sc,
            "high confidence"
            if sc >= self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT
            else "moderate"
            if sc >= self.config.SCORE_THRESHOLD_REJECT
            else "low",
            fp_label,
        )
        result.fingerprint_verified = fp_ok
        result.fingerprint_confidence = fp_conf
        result.fingerprint_matched_title = fp_title
        result.fingerprint_label = fp_label
        if self.require_fingerprint and not fp_ok:
            self.events.on_warn(
                f"Strict fingerprint: cannot confirm {artist} -- {song} "
                f"({fp_label or 'no match'}). Download blocked."
            )
            result.status = "failed"
            result.reason = "Fingerprint did not confirm the song"
            self._persist(
                job.state,
                job.state_lock,
                job.key,
                "failed",
                job.url,
                None,
                None,
                job.output_dir,
                state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                result=result,
            )
            return False

        if result.url:
            job.url = result.url
            job.dur_s = result.duration_seconds or job.dur_s

        # The fingerprint stage can promote a different candidate, and the
        # download stage can replace it again; both make the originally selected
        # entry the wrong source of the publisher and thumbnail we record.
        best_url = (best or {}).get("webpage_url") or (best or {}).get("url")
        if result.url and result.url != best_url:
            for entry, _, _ in job.ranked:
                if (entry.get("webpage_url") or entry.get("url")) == result.url:
                    job.best = entry
                    break
        return True

    # -- stage 3: download the winner, falling through to alternates ----------

    def _stage_fetch(self, job: _SongJob) -> bool:
        """Download, falling through to the next candidate on failure."""
        result = job.result
        (job.output_dir / job.safe_a).mkdir(parents=True, exist_ok=True)

        dl_file, err, used = self._download_with_fallback(
            job.ranked,
            job.best,
            job.artist,
            job.song,
            job.output_dir,
            job.fmt,
            job.quality,
            job.stop_event,
            job.state,
            job.state_lock,
            job.url,
            job.dur_s,
            result,
        )

        if used is not None:
            # The winning candidate became unplayable; carry the replacement
            # candidate's identity and score into the report.
            job.best = self._adopt_candidate(result, used)
            job.url = result.url or job.url
            job.dur_s = result.duration_seconds

        job.dl_file = dl_file
        if dl_file is None:
            self.events.on_download_failed(job.artist, job.song, err)
            result.reason = err
            self._persist(
                job.state,
                job.state_lock,
                job.key,
                "failed",
                job.url,
                None,
                None,
                job.output_dir,
                state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                result=result,
            )
            return False
        return True

    # -- stage 4: post-download checks, tagging, persistence -----------------

    def _stage_finalize(self, job: _SongJob) -> None:
        best = job.best or {}
        self._post_download_checks(
            job.dl_file,
            job.artist,
            job.song,
            job.url,
            best.get("thumbnail"),
            job.fmt,
            job.dur_s,
            job.state,
            job.state_lock,
            job.key,
            job.result,
            job.output_dir,
            job.mb,
            channel=best.get("channel") or best.get("uploader"),
            channel_url=channel_url_for(best),
        )

    def _download_with_fallback(
        self,
        ranked,
        best,
        artist,
        song,
        output_dir,
        fmt,
        quality,
        stop_event,
        state,
        state_lock,
        url,
        dur_s,
        result,
    ):
        """
        Download the winning candidate, falling through to the next one on failure.

        A dead URL is usually a dead *candidate*, not a transient network fault:
        DRM-protected SoundCloud rips and removed videos fail identically on
        every retry. Re-running the same URL three times with backoff wastes the
        whole budget and then reports failure, even when candidate #2 was
        perfectly playable.

        Returns ``(file, error, used_candidate)`` where ``used_candidate`` is
        ``best`` on success and the replacement entry when we fell through, or
        None when every candidate failed.
        """
        tried: set[str] = set()
        candidates = [best] + [entry for entry, _, _ in (ranked or [])[1:]]
        last_error = ""

        for index, entry in enumerate(candidates):
            if entry is None:
                continue
            entry_url = entry.get("webpage_url") or entry.get("url") or url
            if not entry_url or entry_url in tried:
                continue
            if index > 0:
                entry_score = entry.get("_composite_score", 0)
                if entry_score < self.score_threshold:
                    break
            tried.add(entry_url)

            if index > 0:
                self.events.on_warn(
                    f"{artist} -- {song}: primary source unusable, trying "
                    f"{entry.get('title') or entry_url} (score {entry.get('_composite_score', 0)})"
                )
                # The replacement needs its own verification pass, not the
                # fingerprint verdict we collected for a different file.
                result.fingerprint_verified = False
                result.fingerprint_confidence = 0.0
                result.fingerprint_matched_title = None
                result.fingerprint_label = "not verified (alternate candidate)"

            self.events.on_download_start(artist, song, entry_url)
            dl_file, err = execute_download(
                entry_url,
                output_dir,
                fmt,
                quality,
                artist,
                song,
                self.events,
                self.config,
                stop_event,
                state,
                state_lock,
                self.cookies_browser,
                self.cookies_file,
                self.proxy,
            )
            if dl_file is not None:
                return dl_file, "", (entry if index > 0 else None)
            last_error = err
            if stop_event.is_set():
                break

        return None, last_error, None

    def _search_and_select(
        self,
        artist,
        song,
        output_dir,
        state,
        state_lock,
        key,
        result,
        stop_event,
        mb_data,
        executor=None,
    ):
        opts = {
            "max_results": self.max_results,
            # Fetch budget is deliberately decoupled from the presentation
            # budget above; see search.search_with_variants.
            "fetch_multiplier": self.config.FETCH_MULTIPLIER,
            "min_fetch_per_query": self.config.MIN_FETCH_PER_QUERY,
            "cookies_browser": self.cookies_browser,
            "cookies_file": self.cookies_file,
            "proxy": self.proxy,
            "config": self.config,
        }
        reference, reference_warnings = self._resolve_reference(
            mb_data, artist, song, allow_itunes=self.musicbrainz
        )
        for message in reference_warnings:
            self.events.on_warn(message)
        mb_duration = reference["duration"] if reference else None
        if reference and reference.get("album") and isinstance(mb_data, dict):
            # Prefer the album the reference actually came from: it seeds the
            # exact catalogue tracklist lookup.
            mb_data = {**mb_data, "album": reference["album"]}
        best, ranked, src = None, [], None
        if stop_event.is_set():
            result.status = "skipped"
            return best, ranked, src
        self.events.on_search_start(artist, song, "parallel sources")
        search_kwargs = {"executor": executor} if executor is not None else {}
        raw = search_all_sources(
            artist,
            song,
            self.sources,
            opts,
            mb_data=mb_data,
            channel_trust=self.channel_trust if self.channel_search else None,
            config=self.config,
            **search_kwargs,
        )
        if raw:
            found, ranked = select_best_result(
                raw,
                artist,
                song,
                mb_duration,
                self.config,
                None,
                None,
                self.min_duration,
                self.max_duration,
                self.score_threshold,
                self.channel_trust if self.channel_search else None,
            )
            has_sel = hasattr(self.events, "selector_fn") and callable(self.events.selector_fn)
            has_con = hasattr(self.events, "confirm_fn") and callable(self.events.confirm_fn)
            if not ranked:
                self.events.on_search_failed(artist, song, self.sources)
            elif self.decision_classifier is not None:
                decision_best = None
                decision_ranked = None
                decision_error = None
                try:
                    decision_best, decision_ranked = self.decision_classifier.select(
                        artist,
                        song,
                        [entry for entry, _, _ in ranked],
                        reference_metadata=mb_data,
                        runs=self.decision_runs,
                    )
                except JevEvaluationError as exc:
                    decision_error = exc
                    # Reported once per run, not once per song: a provider that
                    # has been given up on has nothing new to say 999 more times.
                    notice = (
                        self.decision_classifier.gate.give_up_notice()
                        if hasattr(self.decision_classifier, "gate")
                        else None
                    )
                    if notice:
                        self.events.on_warn(notice)

                if decision_ranked:
                    ranked = decision_ranked
                result.selection_method = self.decision_provider
                self.events.on_candidates_scored(artist, song, ranked)

                if decision_best is None:
                    # A vetoed or under-threshold best still beats a silent gap:
                    # report *why* it was not taken, which the atomic dimensions
                    # now make answerable ("origin" vs "studio"), not just a score.
                    def heuristic_score(entry: dict) -> int:
                        value = entry.get("_heuristic_score")
                        if value is None:
                            value = entry.get("_composite_score")
                        return int(value or 0)

                    # Prefer a candidate the model did not veto; only consider a
                    # vetoed one when nothing else exists.
                    fallback_pool = [
                        item[0] for item in ranked if item[0].get("_decision_eligible", True)
                    ] or [item[0] for item in ranked]
                    fallback_entry = (
                        max(fallback_pool, key=heuristic_score) if fallback_pool else None
                    )
                    fallback_score = heuristic_score(fallback_entry) if fallback_entry else 0
                    fallback_threshold = max(
                        self.score_threshold,
                        self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT,
                    )
                    if fallback_entry and fallback_score >= fallback_threshold:
                        found = fallback_entry
                        result.selection_method = f"{self.decision_provider}-fallback"
                        detail = (
                            str(decision_error)
                            if decision_error
                            else self._decision_rejection_reason(fallback_entry)
                        )
                        self.events.on_warn(
                            f"{self.decision_provider}: {detail}; using heuristic fallback"
                        )
                    else:
                        result.reason = (
                            str(decision_error)
                            if decision_error
                            else (
                                f"{self.decision_provider.capitalize()} found no candidate at "
                                f"or above {self.decision_threshold:.2f}"
                                + (
                                    f" ({self._decision_rejection_reason(fallback_entry)})"
                                    if fallback_entry
                                    else ""
                                )
                                + ", and no valid heuristic candidate remains"
                            )
                        )
                        self.events.on_warn(result.reason)
                        self._persist(
                            state,
                            state_lock,
                            key,
                            "failed",
                            None,
                            None,
                            None,
                            output_dir,
                            state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                            result=result,
                        )
                        return None, ranked, None
                else:
                    found = decision_best
            else:
                self.events.on_candidates_scored(artist, song, ranked)
            if ranked:
                if has_sel:
                    with self._selection_lock:
                        if stop_event.is_set():
                            result.status = "skipped"
                            result.reason = "Interrupted"
                            return best, ranked, src
                        chosen = self.events.selector_fn(artist, song, ranked)
                    if chosen is None:
                        result.status = "skipped"
                        result.reason = "User skipped"
                        return best, ranked, src
                    best, src = chosen, chosen.get("_source", "unknown")
                elif has_con:
                    if found is None:
                        self.events.on_search_failed(artist, song, self.sources)
                        result.reason = "No valid result"
                        self._persist(
                            state,
                            state_lock,
                            key,
                            "failed",
                            None,
                            None,
                            None,
                            output_dir,
                            state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                            result=result,
                        )
                        return best, ranked, src
                    if not self.events.confirm_fn(artist, song, found):
                        result.status = "skipped"
                        result.reason = "User skipped"
                        return best, ranked, src
                    best, src = found, found.get("_source", "unknown")
                else:
                    if found:
                        best, src = found, found.get("_source", "unknown")
        else:
            for s in self.sources:
                self.events.on_no_results(artist, song, s)
        if best is None:
            self.events.on_search_failed(artist, song, self.sources)
            result.reason = "No valid result"
            self._persist(
                state,
                state_lock,
                key,
                "failed",
                None,
                None,
                None,
                output_dir,
                state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                result=result,
            )
        return best, ranked, src

    def _fingerprint_check(self, artist, song, url, output_dir, best, ranked, result):
        fp_ok, fp_conf, fp_title, fp_label = False, 0.0, None, "disabled"
        # The threshold is calibrated against the heuristic's 0-165 scale, so it
        # must be compared with the heuristic score. `_composite_score` means
        # different things depending on the provider: with a decision model
        # running it holds a probability x100, so the same 70 would read as a
        # different decision in the two modes.
        sc = result.heuristic_score
        threshold = self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT
        needs = (
            self.force_fingerprint
            or self.require_fingerprint
            or (
                bool(self.acoustid_key)
                and not self.skip_fingerprint
                and self.fpcalc_available
                and sc < threshold
            )
        )
        if (
            self.acoustid_key
            and sc >= threshold
            and not self.require_fingerprint
            and not self.force_fingerprint
        ):
            fp_label = f"skipped -- heuristic {sc} >= {threshold}"
        elif self.acoustid_key and not self.fpcalc_available:
            fp_label = "disabled -- fpcalc not found"
        elif self.skip_fingerprint:
            fp_label = "disabled -- --skip-fingerprint"
        if needs:
            self.events.on_fingerprint_start(artist, song, self.config.PARTIAL_DOWNLOAD_SECONDS)
            # A remembered verdict costs nothing, and the same upload is a
            # candidate for several songs of an artist and for every --retry.
            cached = self._fingerprint_cache.get(url, artist, song)
            if cached is not None:
                fp_ok, fp_conf, fp_title = cached.as_tuple()
                fp_label = f"verified {fp_conf:.0%} conf. (cached)" if fp_ok else cached.matched_title
                self.events.on_fingerprint_result(artist, song, fp_ok, fp_conf, fp_title)
            elif self._fingerprint_cache.coalesce(url, artist, song):
                # Another worker is already spending one of three requests per
                # second on this exact check; wait for its answer rather than
                # duplicating the request.
                cached = self._fingerprint_cache.get(url, artist, song)
                if cached is not None:
                    fp_ok, fp_conf, fp_title = cached.as_tuple()
                    fp_label = f"verified {fp_conf:.0%} conf. (shared)" if fp_ok else cached.matched_title
                else:
                    fp_ok, fp_conf, fp_title = False, 0.0, None
                    fp_label = "fingerprint check did not complete"
            else:
                try:
                    fp_ok, fp_conf, fp_title, fp_label = self._fingerprint_one(
                        url, artist, song, output_dir
                    )
                finally:
                    release_fingerprint_slot(self._fingerprint_cache, url, artist, song)
                if fp_conf > 0.4 or self.require_fingerprint:
                    if fp_conf > 0.4:
                        self.events.on_fingerprint_low_confidence(artist, song, fp_title)
                    fp_ok, fp_conf, fp_title, fp_label = self._try_next_fp(
                        ranked, artist, song, output_dir, result
                    )
                elif not fp_ok:
                    self.events.on_fingerprint_no_match(artist, song)
                    if fp_label is None:
                        fp_label = "no AcoustID match"
        return fp_ok, fp_conf, fp_title, fp_label

    def _fingerprint_one(
        self,
        url: str,
        artist: str,
        song: str,
        output_dir: Path,
    ) -> tuple[bool, float, Optional[str], Optional[str]]:
        """Partial-download one candidate, fingerprint it, and cache the verdict.

        The semaphore wraps the partial download as well as the lookup. That is
        deliberate: the AcoustID rate limit only paces the API call, and pacing
        the download around it too would have wasted the budget -- but bounding
        concurrent partials is still what keeps a batch from putting dozens of
        90-second clips in flight at once.
        """
        partial = None
        try:
            with self._fp_semaphore:
                partial = download_partial(
                    url,
                    output_dir,
                    self.events,
                    self.cookies_browser,
                    self.cookies_file,
                    self.proxy,
                    self.config,
                )
                if partial is None:
                    self.events.on_fingerprint_partial_failed(artist, song)
                    return False, 0.0, None, "partial download failed"

                verified, confidence, title = verify_fingerprint(
                    partial,
                    artist,
                    song,
                    self.acoustid_key,
                    self.config,
                    self._circuit_breaker,
                    on_warn=self.events.on_warn,
                    on_info=self.events.on_info,
                    on_fingerprint_error=self.events.on_fingerprint_error,
                )
                label = (
                    f"verified {confidence:.0%} conf."
                    if verified
                    else title
                    if title and confidence > 0
                    else "no AcoustID match"
                )
                self.events.on_fingerprint_result(artist, song, verified, confidence, title)
                self._fingerprint_cache.put(
                    url, artist, song, FingerprintVerdict(verified, confidence, title)
                )
                return verified, confidence, title, label
        finally:
            if partial and partial.exists():
                partial.unlink(missing_ok=True)

    @staticmethod
    def _adopt_candidate(result, entry):
        """Re-point ``result`` at a replacement candidate and copy its verdict.

        A candidate can be replaced twice: once when the fingerprint stage
        promotes a better match, and again when the download stage finds the
        winner unplayable. Both times every number we recorded has to travel
        with the candidate, otherwise the state ends up describing a song we
        did not actually download.
        """
        result.url = entry.get("webpage_url") or entry.get("url") or result.url
        result.source = entry.get("_source", result.source)
        result.matched_title = entry.get("title") or result.matched_title
        result.duration_seconds = int(entry.get("duration") or 0) or result.duration_seconds
        result.heuristic_score = int(
            entry.get("_heuristic_score", entry.get("_composite_score", 0))
        )
        result.composite_score = entry.get("_composite_score", result.composite_score)
        result.score_breakdown = entry.get("_score_breakdown", result.score_breakdown)
        MusicDownloader._copy_decision(result, entry)
        result.fallback_used = True
        return entry

    @staticmethod
    def _copy_decision(result: "DownloadResult", entry: dict) -> None:
        """Move a decision model's verdict from a candidate entry onto the result."""
        result.decision_probability = entry.get("_decision_probability")
        result.decision_samples = list(entry.get("_decision_samples") or [])
        result.decision_runs = int(entry.get("_decision_runs") or 0)
        result.decision_threshold = entry.get("_decision_threshold")
        result.decision_dimensions = {
            key: float(value) for key, value in (entry.get("_decision_dimensions") or {}).items()
        }
        result.decision_failed_gates = list(entry.get("_decision_failed_gates") or [])
        result.decision_spread = float(entry.get("_decision_spread") or 0.0)
        result.decision_stable = bool(entry.get("_decision_stable", True))
        result.decision_choice_probability = entry.get("_decision_choice_probability")
        result.decision_confidence = entry.get("_decision_confidence")

    @staticmethod
    def _decision_rejection_reason(entry: dict) -> str:
        """Explain a rejection with the dimension that caused it."""
        failed = list(entry.get("_decision_failed_gates") or [])
        if failed:
            return f"rejected on {'/'.join(failed)}"
        if not entry.get("_decision_stable", True):
            return f"unstable across runs (spread {float(entry.get('_decision_spread') or 0):.0%})"
        return f"best score {float(entry.get('_decision_probability') or 0):.0%} is below threshold"

    def _apply_decision(self, result: "DownloadResult", entry: dict) -> None:
        """Copy a decision model's verdict, then judge whether it needs a human.

        The model is not just "yes/no" here: it reports per-dimension
        probabilities and run-to-run spread. A candidate that only just cleared
        the threshold, or whose answers moved between runs, is a case for review
        rather than an automatic download.
        """
        self._copy_decision(result, entry)
        result.decision_needs_review = self._needs_review(result)

    def _needs_review(self, result: "DownloadResult") -> bool:
        """Whether a human should look at this pick before trusting it."""
        if result.decision_threshold is None or result.decision_probability is None:
            return False
        if result.decision_failed_gates or not result.decision_stable:
            return True
        margin = self.config.DECISION_REVIEW_MARGIN
        probability = float(result.decision_probability)
        # Only just above threshold, or only just below it: either way the model
        # did not really settle the question.
        return (
            result.decision_threshold <= probability < result.decision_threshold + margin
            or result.decision_threshold - margin <= probability < result.decision_threshold
        )

    def _diversified_alternates(self, ranked) -> list:
        """Pick alternate candidates worth a second fingerprint.

        Taking the next *N* by score spends the partial-download and AcoustID
        budget on near-duplicates of the candidate that just failed -- a live
        rip and its "radio edit" score almost identically and answer identically.
        Choosing by marginal relevance instead means each speculative check is
        a genuinely different recording, which is the only way the fan-out can
        change the outcome.

        Ties fall back to score order, so the choice stays deterministic.
        """
        pool = []
        for entry, score, _ in ranked[1:]:
            if score < self.score_threshold:
                break
            title = normalize_title(strip_featuring((entry.get("title") or "").lower()))
            if title:
                pool.append((title, int(score or 0), entry))
        if not pool:
            return []

        chosen: list[dict] = []
        chosen_titles: list[str] = []
        remaining = list(pool)
        while remaining and len(chosen) < max(1, int(self.config.FP_ALTERNATE_FANOUT)):
            best_entry = None
            best_rank = None
            for index, (title, score, entry) in enumerate(remaining):
                similarity = max(
                    (fuzz.token_set_ratio(title, already) for already in chosen_titles), default=0
                )
                # Higher is better: reward low similarity to what is already
                # chosen, and use score only to break ties.
                rank = (1000 - similarity) * 1000 + score
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_entry = index
            title, _score, entry = remaining.pop(best_entry or 0)
            chosen.append(entry)
            chosen_titles.append(title)
        return chosen

    def _try_next_fp(self, ranked, artist, song, output_dir, result):
        """Fingerprint alternates in parallel and adopt the first real match.

        Done one at a time this cost, per alternate, a full partial download plus
        a rate-limited lookup plus a fixed 0.35s sleep -- all serialised. Running
        the top alternates concurrently turns the worst case from the sum of
        those latencies into roughly the slowest one. The AcoustID budget still
        serialises the API calls themselves, which is correct: that limit is
        real.
        """
        alternates = self._diversified_alternates(ranked)
        if not alternates:
            return False, 0.0, None, "low confidence (no alternate match)"

        pool = self._alternate_executor()

        def _check(entry: dict):
            url = entry.get("webpage_url") or entry.get("url", "")
            if not url:
                return entry, False, 0.0, None
            cached = self._fingerprint_cache.get(url, artist, song)
            if cached is not None:
                return entry, cached.verified, cached.confidence, cached.matched_title
            try:
                verified, confidence, title, _label = self._fingerprint_one(
                    url, artist, song, output_dir
                )
            except Exception:
                return entry, False, 0.0, None
            return entry, verified, confidence, title

        # Submit in rank order so the best alternate is checked first; results
        # are collected in that same order, which keeps selection deterministic.
        futures = [pool.submit(_check, entry) for entry in alternates]
        for future in futures:
            try:
                entry, verified, confidence, title = future.result()
            except Exception:
                continue
            if verified:
                self._adopt_candidate(result, entry)
                return True, confidence, title, f"verified next candidate {confidence:.0%}"
        return False, 0.0, None, "low confidence (no alternate match)"

    def _post_download_checks(
        self,
        downloaded_file,
        artist,
        song,
        url,
        thumbnail,
        fmt,
        expected_duration,
        state,
        state_lock,
        key,
        result,
        output_dir,
        musicbrainz_data,
        channel=None,
        channel_url=None,
    ):
        duration_ok, actual_duration, failure = check_duration(
            downloaded_file,
            expected_duration,
            artist,
            song,
            self.events,
        )
        result.duration_verified = duration_ok
        if failure:
            result.reason = failure
            self._persist(
                state,
                state_lock,
                key,
                "failed",
                url,
                None,
                None,
                output_dir,
                state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                result=result,
            )
            return result

        silence_ratio = 0.0
        if not self.no_silence_check:
            silence_ratio, _, silence_failure = check_silence(
                downloaded_file,
                artist,
                song,
                self.config,
                self.events,
            )
            result.silence_ratio = silence_ratio
            if silence_failure:
                result.reason = silence_failure
                self._persist(
                    state,
                    state_lock,
                    key,
                    "failed",
                    url,
                    None,
                    None,
                    output_dir,
                    state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                    result=result,
                )
                return result

        self.events.on_post_check_summary(
            artist,
            song,
            duration_ok,
            actual_duration,
            silence_ratio,
            not self.no_silence_check,
        )
        if not embed_and_verify(
            downloaded_file,
            song,
            artist,
            url,
            thumbnail,
            fmt,
            musicbrainz_data,
            self.events,
        ):
            result.reason = "Metadata integrity check failed"
            self._persist(
                state,
                state_lock,
                key,
                "failed",
                url,
                None,
                None,
                output_dir,
                state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
                result=result,
            )
            return result

        md5 = compute_md5(downloaded_file)
        result.status = "downloaded"
        result.file_path = downloaded_file
        result.file_size_bytes = downloaded_file.stat().st_size
        result.md5 = md5
        result.musicbrainz_enriched = bool(musicbrainz_data)
        result.album = musicbrainz_data.get("album") if musicbrainz_data else None
        result.year = musicbrainz_data.get("year") if musicbrainz_data else None
        result.genre = musicbrainz_data.get("genre") if musicbrainz_data else None
        result.silence_ratio = silence_ratio
        result.duration_verified = duration_ok
        self._persist(
            state,
            state_lock,
            key,
            "downloaded",
            url,
            str(downloaded_file),
            md5,
            output_dir,
            fingerprint_verified=result.fingerprint_verified,
            fingerprint_confidence=result.fingerprint_confidence,
            fingerprint_label=result.fingerprint_label,
            state_filename=self.config.STATE_FILE,
                writer=self._state_writer,
            result=result,
            channel=channel,
            channel_url=channel_url,
        )
        # Fold this observation into the live model so later songs in the same
        # run already benefit from the channel that just proved itself.
        if channel:
            self.channel_trust.add(
                channel,
                artist=artist,
                channel_url=channel_url or "",
                verified=bool(result.fingerprint_verified),
            )
        return result

    @staticmethod
    def _iter_entries(data):
        entries = data.get("entries")
        if entries:
            for e in entries:
                if e:
                    yield from MusicDownloader._iter_entries(e)
            return
        yield data

    @staticmethod
    def _persist(
        state,
        lock,
        key,
        status,
        url,
        file_path,
        md5,
        output_dir,
        fingerprint_verified=False,
        fingerprint_confidence=0.0,
        fingerprint_label=None,
        preserve_timestamp=False,
        state_filename=None,
        channel=None,
        channel_url=None,
        result=None,
        preserve_fields=None,
        writer=None,
    ):
        MusicDownloader._persist_state(
            state,
            lock,
            key,
            status,
            url,
            file_path,
            md5,
            output_dir,
            fingerprint_verified=fingerprint_verified,
            fingerprint_confidence=fingerprint_confidence,
            fingerprint_label=fingerprint_label,
            preserve_timestamp=preserve_timestamp,
            state_filename=state_filename,
            channel=channel,
            channel_url=channel_url,
            result=result,
            preserve_fields=preserve_fields,
            writer=writer,
        )

    @staticmethod
    def _persist_state(
        state,
        lock,
        key,
        status,
        url,
        file_path,
        md5,
        output_dir,
        fingerprint_verified=False,
        fingerprint_confidence=0.0,
        fingerprint_label=None,
        preserve_timestamp=False,
        state_filename=None,
        channel=None,
        channel_url=None,
        result=None,
        preserve_fields=None,
        writer=None,
    ):
        with lock:
            downloads = state.setdefault("downloads", {})
            existing = downloads.get(key)
            timestamp = (
                existing.get("timestamp")
                if preserve_timestamp and existing and "timestamp" in existing
                else datetime.now(timezone.utc).isoformat()
            )
            entry = {
                "status": status,
                "url": url,
                "file_path": file_path,
                "md5": md5,
                "fingerprint_verified": fingerprint_verified,
                "fingerprint_confidence": fingerprint_confidence,
                "fingerprint_label": fingerprint_label,
                "timestamp": timestamp,
            }
            # Publisher provenance is what makes the learned channel trust model
            # work across runs; keep any previously recorded values on failure
            # paths so a retry does not erase what we already know.
            resolved_channel = channel or (existing or {}).get("channel")
            resolved_channel_url = channel_url or (existing or {}).get("channel_url")
            if resolved_channel:
                entry["channel"] = resolved_channel
            if resolved_channel_url:
                entry["channel_url"] = resolved_channel_url
            # The result carries the full story of the attempt -- which title won,
            # how the heuristic scored it, what Jev/Kev decided about it -- so the
            # run can be audited from the state file alone.
            if result is not None:
                merge_state_detail(entry, existing, state_detail(result), preserve_fields)
            downloads[key] = entry
        # The mutation above is the cheap part and stays under the lock. The
        # write is not: rewriting the whole document and fsyncing it while every
        # other worker waits on this same lock is what turned a 500-song batch
        # into an O(N^2) stall. The writer snapshots under the lock and does the
        # serialisation and I/O on a background thread instead.
        if writer is not None:
            writer.record()
        else:
            save_state(state, output_dir, state_filename)
