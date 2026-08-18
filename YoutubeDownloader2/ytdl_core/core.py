"""
MusicDownloader — the library-facing core class.

This module contains NO Rich / CLI code. All user-facing output is delegated
to the DownloaderEvents instance supplied at construction time.
"""

from __future__ import annotations

import concurrent.futures
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import yt_dlp
from rapidfuzz import fuzz
from yt_dlp.utils import DownloadError, ExtractorError

from .config import Config
from .events import DownloaderEvents
from .fingerprint import (
    AcoustIDCircuitBreaker,
    has_excessive_silence,
    verify_duration,
    verify_fingerprint,
)
from .metadata import embed_metadata, fetch_musicbrainz
from .reports import export_report, update_json_file
from .result import DownloadResult
from .search import search_all_sources, select_best_result
from .state import load_state, save_state
from .utils import apply_delay, compute_md5, sanitize_filename
from .ytdlp_options import build_ytdlp_base_opts, resolve_downloaded_file


class MusicDownloader:
    def __init__(
        self,
        config: Optional[Config] = None,
        events: Optional[DownloaderEvents] = None,
        acoustid_key: Optional[str] = None,
        force_fingerprint: bool = False,
        skip_fingerprint: bool = False,
        no_silence_check: bool = False,
        score_threshold: Optional[int] = None,
        sources: Optional[list[str]] = None,
        workers: int = 2,
        delay: tuple[float, float] = (2.0, 5.0),
        max_results: int = 5,
        fuzzy_threshold: int = 65,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        musicbrainz: bool = False,
        cookies_browser: Optional[str] = None,
        cookies_file: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self.config = config or Config()
        self.events = events or DownloaderEvents()
        self.acoustid_key = acoustid_key
        self.force_fingerprint = force_fingerprint
        self.skip_fingerprint = skip_fingerprint
        self.no_silence_check = no_silence_check
        self.score_threshold = (
            score_threshold if score_threshold is not None else self.config.SCORE_THRESHOLD_REJECT
        )
        self.sources = sources or list(self.config.DEFAULT_SOURCES)
        self.workers = max(1, min(workers, self.config.MAX_WORKERS))
        self.delay = delay
        self.max_results = max_results
        self.fuzzy_threshold = fuzzy_threshold
        self.max_duration = max_duration or self.config.MAX_DURATION_SECONDS
        self.min_duration = min_duration or self.config.MIN_DURATION_SECONDS
        self.musicbrainz = musicbrainz
        self.cookies_browser = cookies_browser
        self.cookies_file = cookies_file
        self.proxy = proxy
        self.fpcalc_available: bool = shutil.which("fpcalc") is not None
        self._fp_semaphore = threading.Semaphore(2)

        self._circuit_breaker = AcoustIDCircuitBreaker(
            cooldown_seconds=60.0,
        )

        self._selection_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        artist: str,
        song: str,
        output_dir: Path,
        fmt: str = "mp3",
        quality: str = "192",
        skip_existing: bool = False,
    ) -> DownloadResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(output_dir)
        state_lock = threading.Lock()
        stop_event = threading.Event()
        return self._process_song(
            artist=artist,
            song=song,
            output_dir=output_dir,
            fmt=fmt,
            quality=quality,
            skip_existing=skip_existing,
            state=state,
            state_lock=state_lock,
            stop_event=stop_event,
            seen_artists=set(),
            seen_artists_lock=threading.Lock(),
            all_pairs=[(artist, song)],
        )

    def download_batch(
        self,
        songs: dict[str, list[str]],
        output_dir: Path,
        fmt: str = "mp3",
        quality: str = "192",
        skip_existing: bool = False,
        report_formats: Optional[list[str]] = None,
        update_json_path: Optional[Path] = None,
    ) -> list[DownloadResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = [(artist, song) for artist, lst in songs.items() for song in lst if lst]
        total = len(pairs)
        self.events.on_session_start(total)

        state = load_state(output_dir)
        state_lock = threading.Lock()
        stop_event = threading.Event()
        seen_artists: set[str] = set()
        seen_artists_lock = threading.Lock()
        all_results: list[DownloadResult] = []
        results_lock = threading.Lock()
        start = time.monotonic()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(
                        self._process_song,
                        artist=artist,
                        song=song,
                        output_dir=output_dir,
                        fmt=fmt,
                        quality=quality,
                        skip_existing=skip_existing,
                        state=state,
                        state_lock=state_lock,
                        stop_event=stop_event,
                        seen_artists=seen_artists,
                        seen_artists_lock=seen_artists_lock,
                        all_pairs=pairs,
                    ): (artist, song)
                    for artist, song in pairs
                    if not stop_event.is_set()
                }
                for fut in concurrent.futures.as_completed(futures):
                    artist, song = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = DownloadResult(
                            artist=artist,
                            song=song,
                            status="failed",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    with results_lock:
                        all_results.append(result)
                    self.events.on_result(result)

        except KeyboardInterrupt:
            stop_event.set()
            elapsed = time.monotonic() - start
            self.events.on_interrupted(len(all_results), total, elapsed)
            save_state(state, output_dir)
            self.events.on_session_complete(all_results, elapsed)
            return all_results

        elapsed = time.monotonic() - start
        self.events.on_session_complete(all_results, elapsed)

        if report_formats:
            export_report([r.to_dict() for r in all_results], output_dir, report_formats)
        if update_json_path:
            update_json_file(update_json_path, [r.to_dict() for r in all_results])

        return all_results

    def download_url(
        self,
        url: str,
        output_dir: Path,
        fmt: str = "mp3",
        quality: str = "192",
        max_downloads: Optional[int] = None,
        skip_existing: bool = False,
        match_title: Optional[str] = None,
        reject_title: Optional[str] = None,
    ) -> None:
        """Download directly from an arbitrary URL (playlist, channel, shorts, video, etc.)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        match_pattern = re.compile(match_title, re.IGNORECASE) if match_title else None
        reject_pattern = re.compile(reject_title, re.IGNORECASE) if reject_title else None

        def _session_progress_hook(d: dict) -> None:
            status = d.get("status")
            if status == "downloading":
                total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded_b = d.get("downloaded_bytes") or 0
                pct = (downloaded_b / total_b * 100.0) if total_b else 0.0
                self.events.on_download_progress(
                    "URL",
                    url[:30],
                    pct,
                    d.get("speed") or 0.0,
                    downloaded_b,
                    total_b,
                )
            elif status == "finished":
                total_b = d.get("total_bytes") or d.get("downloaded_bytes") or 0
                self.events.on_download_progress("URL", url[:30], 100.0, 0.0, total_b, total_b)
            elif status == "processing":
                self.events.on_info("[cyan]Processing playlist item...[/cyan]")

        scan_opts = build_ytdlp_base_opts(
            output_dir=output_dir,
            fmt=fmt,
            quality=quality,
            quiet=False,
            no_warnings=False,
            progress_hook=None,
            skip_existing=False,
            max_downloads=max_downloads,
            cookies_browser=self.cookies_browser,
            cookies_file=self.cookies_file,
            proxy=self.proxy,
            enable_remote_components=True,
            youtube_player_clients=["web"],
            noplaylist=False,
            for_scan=True,
        )

        try:
            with yt_dlp.YoutubeDL(scan_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            self.events.on_download_failed("URL", url, str(exc))
            return

        if not info:
            return

        entries = list(self._iter_entries(info))

        urls_to_download: list[tuple[str, str, str]] = []
        for entry in entries:
            if not entry:
                continue

            title = entry.get("title", "Unknown")
            uploader = entry.get("uploader", "Unknown")

            if match_pattern and not match_pattern.search(title):
                self.events.on_warn(f"[yellow]  Skipped (doesn't match --match-title): {title}[/yellow]")
                continue

            if reject_pattern and reject_pattern.search(title):
                self.events.on_warn(f"[yellow]  Skipped (matches --reject-title): {title}[/yellow]")
                continue

            if entry.get("is_live"):
                self.events.on_warn(f"[yellow]  Skipped (live stream): {title}[/yellow]")
                continue

            duration = entry.get("duration")
            if duration is not None:
                if self.min_duration and duration < self.min_duration:
                    self.events.on_warn(
                        f"[yellow]  Skipped (too short, {int(duration)}s < {self.min_duration}s): {title}[/yellow]"
                    )
                    continue
                if self.max_duration and duration > self.max_duration:
                    self.events.on_warn(
                        f"[yellow]  Skipped (too long, {int(duration)}s > {self.max_duration}s): {title}[/yellow]"
                    )
                    continue

            item_url = entry.get("webpage_url") or entry.get("url")
            if not item_url and entry.get("id"):
                item_url = f"https://www.youtube.com/watch?v={entry['id']}"

            if item_url and item_url != url and "search?" not in item_url:
                urls_to_download.append((uploader, title, item_url))

        if not urls_to_download:
            self.events.on_warn("[yellow]No suitable URLs found to download after applying filters.[/yellow]")
            return

        total = len(urls_to_download)
        self.events.on_session_start(total)

        all_results: list[DownloadResult] = []
        results_lock = threading.Lock()
        stop_event = threading.Event()
        start = time.monotonic()

        def _process_single_url(item_artist: str, item_title: str, item_url: str) -> None:
            if stop_event.is_set():
                return

            result = DownloadResult(artist=item_artist, song=item_title)

            def _item_progress_hook(d: dict) -> None:
                status = d.get("status")
                if status == "downloading":
                    total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded_b = d.get("downloaded_bytes") or 0
                    pct = (downloaded_b / total_b * 100.0) if total_b else 0.0
                    self.events.on_download_progress(
                        item_artist,
                        item_title[:30],
                        pct,
                        d.get("speed") or 0.0,
                        downloaded_b,
                        total_b,
                    )
                elif status == "finished":
                    total_b = d.get("total_bytes") or d.get("downloaded_bytes") or 0
                    self.events.on_download_progress(
                        item_artist,
                        item_title[:30],
                        100.0,
                        0.0,
                        total_b,
                        total_b,
                    )
                elif status == "processing":
                    self.events.on_info(f"[cyan]Processing: {item_title}[/cyan]")

            item_opts = build_ytdlp_base_opts(
                output_dir=output_dir,
                fmt=fmt,
                quality=quality,
                quiet=True,
                no_warnings=True,
                progress_hook=_item_progress_hook,
                skip_existing=skip_existing,
                cookies_browser=self.cookies_browser,
                cookies_file=self.cookies_file,
                proxy=self.proxy,
                enable_remote_components=True,
                youtube_player_clients=["web"],
                noplaylist=True,
                for_scan=False,
            )

            safe_artist = sanitize_filename(item_artist)
            safe_title = sanitize_filename(item_title)
            guess_file = output_dir / safe_artist / f"{safe_title}.{fmt}"

            if skip_existing and guess_file.exists():
                self.events.on_skip_existing(item_artist, item_title, guess_file, True)
                result.status = "skipped"
                result.reason = "File exists"
                result.file_path = guess_file
                with results_lock:
                    all_results.append(result)
                self.events.on_result(result)
                return

            self.events.on_download_start(item_artist, item_title, item_url)

            try:
                with yt_dlp.YoutubeDL(item_opts) as ydl_item:
                    info_dict = ydl_item.extract_info(item_url, download=True)
                    if not info_dict:
                        raise RuntimeError("yt-dlp returned no info_dict")

                    source_file = Path(ydl_item.prepare_filename(info_dict))
                    downloaded_file = resolve_downloaded_file(source_file, fmt)

                    if downloaded_file is None:
                        raise FileNotFoundError(
                            f"yt-dlp finished but no output file was found for {item_artist} - {item_title}"
                        )

                    result.status = "downloaded"
                    result.file_path = downloaded_file
                    result.duration_seconds = info_dict.get("duration")

            except Exception as exc:
                self.events.on_download_failed(item_artist, item_title, str(exc))
                result.status = "failed"
                result.reason = str(exc)

            with results_lock:
                all_results.append(result)
            self.events.on_result(result)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [
                    executor.submit(_process_single_url, artist, title, item_url)
                    for artist, title, item_url in urls_to_download
                ]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception:
                        pass
        except KeyboardInterrupt:
            stop_event.set()

        elapsed = time.monotonic() - start
        self.events.on_session_complete(all_results, elapsed)

    def verify_library(
        self,
        songs: dict[str, list[str]],
        output_dir: Path,
        fmt: str = "mp3",
    ) -> list[DownloadResult]:
        """
        Validates the existence, size, duration, and fingerprint of local audio tracks.
        Implements progressive state saving to preserve progress upon interruption.
        Includes all requested songs in the final result, regardless of when they were verified.
        """
        output_dir = Path(output_dir)
        state = load_state(output_dir)

        results_map = {}
        pairs_to_process = []

        # Build the complete map and separate items that actually need processing
        for artist, lst in songs.items():
            if not lst:
                continue
            for song in lst:
                key = f"{artist}::{song}"
                entry_state = state.get("downloads", {}).get(key, {})
                status = entry_state.get("status")

                # If it's already verified, construct a successful DownloadResult immediately
                if status == "verified":
                    res = DownloadResult(artist=artist, song=song, status="verified")
                    res.file_path = Path(entry_state["file_path"]) if entry_state.get("file_path") else None
                    res.md5 = entry_state.get("md5")
                    res.fingerprint_verified = entry_state.get("fingerprint_verified", False)
                    results_map[(artist, song)] = res
                else:
                    # Initialize as skipped, add to queue for processing
                    results_map[(artist, song)] = DownloadResult(
                        artist=artist, song=song, status="skipped", reason="Not verified in state"
                    )
                    # Only process if it was marked as downloaded
                    if status == "downloaded":
                        pairs_to_process.append((artist, song))

        total = len(pairs_to_process)
        self.events.on_session_start(total, is_verify=True)

        state_lock = threading.Lock()
        stop_event = threading.Event()
        start = time.monotonic()

        seen_artists: set[str] = set()
        seen_artists_lock = threading.Lock()

        def _verify_single(artist: str, song: str) -> DownloadResult:
            with seen_artists_lock:
                if artist not in seen_artists:
                    seen_artists.add(artist)
                    count = sum(1 for a, _ in pairs_to_process if a == artist)
                    self.events.on_artist_start(artist, count)

            result = DownloadResult(artist=artist, song=song)

            if stop_event.is_set():
                result.status = "skipped"
                result.reason = "Interrupted"
                return result

            safe_artist = sanitize_filename(artist)
            safe_song = sanitize_filename(song)
            expected_file = output_dir / safe_artist / f"{safe_song}.{fmt}"

            if not expected_file.exists():
                result.status = "failed"
                result.reason = "File does not exist"
                return result

            try:
                size = expected_file.stat().st_size
                if size < 1024 * 50:
                    result.status = "failed"
                    result.reason = "File is extremely small (<50KB)"
                    result.file_path = expected_file
                    return result
                result.file_size_bytes = size
            except Exception:
                result.status = "failed"
                result.reason = "Could not read file size"
                return result

            duration_ok, actual_duration = verify_duration(expected_file, 0)
            if not duration_ok and actual_duration == 0:
                result.status = "failed"
                result.reason = "Corrupted file or missing metadata"
                result.file_path = expected_file
                return result

            result.duration_seconds = actual_duration

            if self.acoustid_key:
                apply_delay(0.2, 0.5)
                if getattr(self, "musicbrainz", False):
                    try:
                        mb_data = fetch_musicbrainz(artist, song)
                        if mb_data:
                            result.musicbrainz_enriched = True
                            if "duration_seconds" in mb_data:
                                result.duration_seconds = mb_data["duration_seconds"]
                    except Exception:
                        pass

                with self._fp_semaphore:
                    fp_ok, fp_conf, fp_title = verify_fingerprint(
                        expected_file,
                        artist,
                        song,
                        self.acoustid_key,
                        self.config,
                        self._circuit_breaker,
                        on_warn=self.events.on_warn,
                        on_info=self.events.on_info,
                        on_fingerprint_error=self.events.on_fingerprint_error,
                    )

                result.fingerprint_verified = fp_ok
                result.fingerprint_confidence = fp_conf
                result.fingerprint_matched_title = fp_title

                if not fp_ok and fp_conf > 0:
                    result.status = "failed"
                    result.reason = f"Fingerprint mismatch: Found '{fp_title}' ({fp_conf*100:.1f}%)"
                    result.file_path = expected_file
                    return result

            result.status = "verified"
            result.file_path = expected_file
            return result

        if pairs_to_process:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(_verify_single, artist, song): (artist, song)
                    for artist, song in pairs_to_process
                }
                try:
                    for fut in concurrent.futures.as_completed(futures):
                        artist, song = futures[fut]
                        key = f"{artist}::{song}"

                        try:
                            res = fut.result()
                        except Exception as exc:
                            res = DownloadResult(
                                artist=artist, song=song, status="failed", reason=f"Exception: {str(exc)}"
                            )

                        results_map[(artist, song)] = res
                        self.events.on_result(res)

                        if res.status != "skipped":
                            with state_lock:
                                existing = state.get("downloads", {}).get(key, {})

                            current_md5 = existing.get("md5")
                            if not current_md5 and res.file_path and Path(res.file_path).exists():
                                current_md5 = compute_md5(Path(res.file_path))

                            fingerprint_verified = getattr(res, "fingerprint_verified", False)

                            self._persist(
                                state,
                                state_lock,
                                key,
                                res.status,
                                existing.get("url"),
                                str(res.file_path) if res.file_path else existing.get("file_path"),
                                current_md5,
                                output_dir,
                                fingerprint_verified=fingerprint_verified,
                                preserve_timestamp=True,
                            )

                except KeyboardInterrupt:
                    stop_event.set()
                    for fut in futures:
                        fut.cancel()

        all_results = list(results_map.values())
        elapsed = time.monotonic() - start
        self.events.on_session_complete(all_results, elapsed)

        return all_results

    # ------------------------------------------------------------------
    # Internal: single-song processing pipeline
    # ------------------------------------------------------------------

    def _process_song(
        self,
        artist: str,
        song: str,
        output_dir: Path,
        fmt: str,
        quality: str,
        skip_existing: bool,
        state: dict,
        state_lock: threading.Lock,
        stop_event: threading.Event,
        seen_artists: set,
        seen_artists_lock: threading.Lock,
        all_pairs: list[tuple[str, str]],
    ) -> DownloadResult:
        with seen_artists_lock:
            if artist not in seen_artists:
                seen_artists.add(artist)
                count = sum(1 for a, _ in all_pairs if a == artist)
                self.events.on_artist_start(artist, count)

        result = DownloadResult(artist=artist, song=song)
        key = f"{artist}::{song}"

        if stop_event.is_set():
            result.status = "skipped"
            result.reason = "Interrupted"
            return result

        safe_artist = sanitize_filename(artist)
        safe_song = sanitize_filename(song)
        expected_file = output_dir / safe_artist / f"{safe_song}.{fmt}"

        with state_lock:
            existing = state.get("downloads", {}).get(key)

        if skip_existing and existing and existing.get("status") == "downloaded":
            stored_md5 = existing.get("md5")
            if expected_file.exists():
                if stored_md5:
                    if compute_md5(expected_file) == stored_md5:
                        self.events.on_skip_existing(artist, song, expected_file, True)
                        result.status = "skipped"
                        result.file_path = expected_file
                        result.md5 = stored_md5
                        return result
                    self.events.on_md5_mismatch(artist, song)
                else:
                    self.events.on_skip_existing(artist, song, expected_file, False)
                    result.status = "skipped"
                    result.file_path = expected_file
                    return result

        apply_delay(self.delay[0], self.delay[1])

        search_opts = {
            "max_results": self.max_results,
            "cookies_browser": self.cookies_browser,
            "cookies_file": self.cookies_file,
            "proxy": self.proxy,
        }

        mb_duration_seconds: Optional[int] = None

        if getattr(self, "musicbrainz", False):
            try:
                mb_data = fetch_musicbrainz(artist, song)
                if mb_data:
                    self.events.on_musicbrainz_result(artist, song, True, mb_data)
                    mb_duration_seconds = mb_data.get("duration_seconds")
            except Exception as mb_exc:
                self.events.on_warn(f"[yellow]MusicBrainz failed: {mb_exc}[/yellow]")

        best_result: Optional[dict] = None
        ranked_candidates: list[tuple[dict, int, dict]] = []
        chosen_source: Optional[str] = None

        if stop_event.is_set():
            result.status = "skipped"
            return result

        self.events.on_search_start(artist, song, "parallel sources")
        raw = search_all_sources(artist, song, self.sources, search_opts)

        if raw:
            found, ranked = select_best_result(
                results=raw,
                artist=artist,
                song=song,
                mb_duration_seconds=mb_duration_seconds,
                config=self.config,
                console=None,
                console_lock=None,
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                score_threshold=self.score_threshold,
            )

            has_selector = hasattr(self.events, "selector_fn") and callable(self.events.selector_fn)
            has_confirm = hasattr(self.events, "confirm_fn") and callable(self.events.confirm_fn)

            # Only report "search failed" when there's nothing to choose from
            if not ranked:
                self.events.on_search_failed(artist, song, self.sources)
            else:
                self.events.on_candidates_scored(artist, song, ranked)

            if ranked:
                if hasattr(self.events, "selector_fn") and callable(self.events.selector_fn):
                    with self._selection_lock:
                        if stop_event.is_set():
                            result.status = "skipped"
                            result.reason = "Interrupted"
                            return result
                        chosen = self.events.selector_fn(artist, song, ranked)
                    if chosen is None:
                        result.status = "skipped"
                        result.reason = "User skipped in select mode"
                        return result
                    best_result = chosen
                    ranked_candidates = ranked
                    chosen_source = chosen.get("_source", "unknown")
                elif has_confirm:
                    if found is None:
                        self.events.on_search_failed(artist, song, self.sources)
                        result.reason = "No valid result found after all sources"
                        self._persist(state, state_lock, key, "failed", None, None, None, output_dir)
                        return result
                    if not self.events.confirm_fn(artist, song, found):
                        result.status = "skipped"
                        result.reason = "User skipped in interactive mode"
                        return result
                    best_result = found
                    ranked_candidates = ranked
                    chosen_source = best_result.get("_source", "unknown")
                else:
                    if found is not None:
                        best_result = found
                        ranked_candidates = ranked
                        chosen_source = best_result.get("_source", "unknown")
        else:
            for source in self.sources:
                self.events.on_no_results(artist, song, source)

        if best_result is None:
            self.events.on_search_failed(artist, song, self.sources)
            result.reason = "No valid result found after all sources"
            self._persist(state, state_lock, key, "failed", None, None, None, output_dir)
            return result

        url = best_result.get("webpage_url") or best_result.get("url", "")
        matched_title = best_result.get("title") or ""
        duration_s = int(best_result.get("duration") or 0)
        thumbnail_url = best_result.get("thumbnail")
        composite_score = best_result.get("_composite_score", 0)
        score_breakdown = best_result.get("_score_breakdown", {})

        result.source = chosen_source
        result.url = url
        result.matched_title = matched_title
        result.fuzzy_score = int(
            fuzz.token_sort_ratio(f"{artist} {song}".lower(), matched_title.lower())
        )
        result.duration_seconds = duration_s
        result.composite_score = composite_score
        result.score_breakdown = score_breakdown

        fp_verified = False
        fp_confidence = 0.0
        fp_matched_title: Optional[str] = None
        fp_label = "disabled"

        needs_fp = (
            self.force_fingerprint
            or (
                bool(self.acoustid_key)
                and not self.skip_fingerprint
                and self.fpcalc_available
                and composite_score < self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT
            )
        )

        if self.acoustid_key and composite_score >= self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT:
            fp_label = f"skipped -- score {composite_score} >= threshold"
        elif self.acoustid_key and not self.fpcalc_available:
            fp_label = "disabled -- fpcalc not found"
        elif self.skip_fingerprint:
            fp_label = "disabled -- --skip-fingerprint"

        if needs_fp:
            partial_path: Optional[Path] = None
            self.events.on_fingerprint_start(artist, song, self.config.PARTIAL_DOWNLOAD_SECONDS)
            try:
                with self._fp_semaphore:
                    partial_path = self._download_partial(url, output_dir)
                    if partial_path is None:
                        self.events.on_fingerprint_partial_failed(artist, song)
                        fp_label = "partial download failed"
                    else:
                        is_match, conf, fp_title = verify_fingerprint(
                            partial_path,
                            artist,
                            song,
                            self.acoustid_key,
                            self.config,
                            self._circuit_breaker,
                            on_warn=self.events.on_warn,
                            on_info=self.events.on_info,
                            on_fingerprint_error=self.events.on_fingerprint_error,
                        )
                        time.sleep(0.35)
                        fp_confidence = conf
                        fp_matched_title = fp_title
                        self.events.on_fingerprint_result(artist, song, is_match, conf, fp_title)

                        if is_match:
                            fp_verified = True
                            fp_label = f"verified {conf:.0%} conf."
                        elif conf > 0.4:
                            fp_label = f"low confidence ({fp_title})"
                            self.events.on_fingerprint_low_confidence(artist, song, fp_title)
                            for cand_r, cand_s, _bd in ranked_candidates[1:]:
                                if cand_s < self.score_threshold:
                                    break
                                next_url = cand_r.get("webpage_url") or cand_r.get("url", "")
                                next_partial: Optional[Path] = None
                                try:
                                    next_partial = self._download_partial(next_url, output_dir)
                                    if next_partial:
                                        n_ok, n_conf, n_title = verify_fingerprint(
                                            next_partial,
                                            artist,
                                            song,
                                            self.acoustid_key,
                                            self.config,
                                            self._circuit_breaker,
                                            on_warn=self.events.on_warn,
                                            on_info=self.events.on_info,
                                            on_fingerprint_error=self.events.on_fingerprint_error,
                                        )
                                        time.sleep(0.35)
                                        if n_ok:
                                            best_result = cand_r
                                            url = next_url
                                            matched_title = cand_r.get("title") or ""
                                            duration_s = int(cand_r.get("duration") or 0)
                                            fp_verified = True
                                            fp_confidence = n_conf
                                            fp_matched_title = n_title
                                            fp_label = f"verified next candidate {n_conf:.0%}"
                                            result.url = url
                                            result.matched_title = matched_title
                                            result.duration_seconds = duration_s
                                            result.composite_score = cand_r.get("_composite_score", 0)
                                            result.score_breakdown = cand_r.get("_score_breakdown", {})
                                            break
                                finally:
                                    if next_partial and next_partial.exists():
                                        next_partial.unlink(missing_ok=True)
                                break
                        else:
                            self.events.on_fingerprint_no_match(artist, song)
                            fp_label = "no AcoustID match"
            finally:
                if partial_path and partial_path.exists():
                    partial_path.unlink(missing_ok=True)

        sc = result.composite_score
        score_label = (
            "high confidence"
            if sc >= self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT
            else "moderate"
            if sc >= self.config.SCORE_THRESHOLD_REJECT
            else "low"
        )
        self.events.on_verification_status(artist, song, sc, score_label, fp_label)
        result.fingerprint_verified = fp_verified
        result.fingerprint_confidence = fp_confidence
        result.fingerprint_matched_title = fp_matched_title

        (output_dir / safe_artist).mkdir(parents=True, exist_ok=True)
        is_video = fmt == "mp4"
        output_template = str(
            output_dir / safe_artist / f"{safe_song}.mp4"
            if is_video
            else f"{safe_song}.%(ext)s"
        )
        self.events.on_download_start(artist, song, url)

        def _progress_hook(d: dict) -> None:
            status = d.get("status")
            if status == "downloading":
                total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded_b = d.get("downloaded_bytes") or 0
                pct = (downloaded_b / total_b * 100.0) if total_b else 0.0
                self.events.on_download_progress(
                    artist, song, pct, d.get("speed") or 0.0, downloaded_b, total_b
                )
            elif status == "finished":
                total_b = d.get("total_bytes") or d.get("downloaded_bytes") or 0
                self.events.on_download_progress(
                    artist, song, 100.0, 0.0, total_b, total_b
                )
            elif status == "processing":
                self.events.on_info(f"[cyan]Processing: {song}[/cyan]")

        if is_video:
            ydl_opts: Any = {
                "format": f"bestvideo[height<={quality}]+bestaudio/bestvideo[height<={quality}]/best",
                "outtmpl": output_template,
                "merge_output_format": "mp4",
                "quiet": True,
                "no_warnings": True,
                "progress_hooks": [_progress_hook],
                "postprocessors": [
                    {"key": "FFmpegMetadata"},
                    {"key": "EmbedThumbnail", "already_have_thumbnail": False},
                ],
                "noplaylist": True,
                "writethumbnail": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["web"],
                    }
                },
            }
        else:
            ydl_opts: Any = {
                "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
                "progress_hooks": [_progress_hook],
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": fmt, "preferredquality": quality},
                    {"key": "FFmpegMetadata"},
                    {"key": "EmbedThumbnail", "already_have_thumbnail": False},
                ],
                "noplaylist": True,
                "writethumbnail": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["web"],
                    }
                },
            }

        node_path = shutil.which("node")
        if node_path:
            ydl_opts["js_runtimes"] = {
                "node": {"path": node_path}
            }

        ydl_opts["remote_components"] = ["ejs:github"]

        if self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)
        if self.cookies_file:
            ydl_opts["cookiefile"] = str(self.cookies_file)
        if self.proxy:
            ydl_opts["proxy"] = self.proxy

        downloaded_file: Optional[Path] = None
        last_error = ""

        for attempt in range(1, self.config.RETRY_ATTEMPTS + 1):
            if stop_event.is_set():
                break
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
                    if info_dict:
                        source_file = Path(ydl.prepare_filename(info_dict))
                        downloaded_file = resolve_downloaded_file(source_file, fmt)

                if downloaded_file and downloaded_file.exists():
                    break

            except DownloadError as exc:
                last_error = f"DownloadError: {exc}"
            except ExtractorError as exc:
                last_error = f"ExtractorError: {exc}"
            except OSError as exc:
                if exc.errno == 28:
                    self.events.on_disk_full()
                    stop_event.set()
                    save_state(state, output_dir)
                    result.reason = "Disk full"
                    return result
                last_error = f"OSError: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.config.RETRY_ATTEMPTS:
                wait = self.config.RETRY_BACKOFF_BASE**attempt
                self.events.on_download_retry(
                    artist, song, attempt, self.config.RETRY_ATTEMPTS, last_error, wait
                )
                time.sleep(wait)

        if downloaded_file is None or not downloaded_file.exists():
            self.events.on_download_failed(artist, song, last_error)
            result.reason = last_error
            self._persist(state, state_lock, key, "failed", url, None, None, output_dir, needs_fp)
            return result

        dur_ok, actual_dur = verify_duration(downloaded_file, duration_s)
        result.duration_verified = dur_ok
        self.events.on_duration_check(artist, song, duration_s, actual_dur, dur_ok)

        if not dur_ok and duration_s > 0:
            discrepancy = abs(actual_dur - duration_s) / max(duration_s, 1)
            if discrepancy > 0.40:
                downloaded_file.unlink(missing_ok=True)
                result.reason = f"Duration discrepancy {discrepancy:.0%}"
                self._persist(state, state_lock, key, "failed", url, None, None, output_dir, needs_fp)
                return result

        silence_ratio = 0.0
        if not self.no_silence_check:
            is_excessive, silence_ratio = has_excessive_silence(downloaded_file, self.config)
            result.silence_ratio = silence_ratio
            self.events.on_silence_check(artist, song, silence_ratio, is_excessive)
            if is_excessive:
                self.events.on_silence_rejected(artist, song, silence_ratio)
                downloaded_file.unlink(missing_ok=True)
                result.reason = f"Excessive silence ({silence_ratio:.1%})"
                self._persist(state, state_lock, key, "failed", url, None, None, output_dir, needs_fp)
                return result

        self.events.on_post_check_summary(artist, song, dur_ok, actual_dur, silence_ratio)

        mb_data: Optional[dict] = None
        mb_enriched = False
        if self.musicbrainz:
            mb_data = fetch_musicbrainz(artist, song)
            mb_enriched = mb_data is not None
            self.events.on_musicbrainz_result(artist, song, mb_enriched, mb_data or {})

        extra = {
            "source_url": url,
            "album": mb_data.get("album") if mb_data else None,
            "year": mb_data.get("year") if mb_data else None,
            "genre": mb_data.get("genre") if mb_data else None,
            "track_num": mb_data.get("track_num") if mb_data else None,
            "mb_id": mb_data.get("mb_id") if mb_data else None,
            "cover_url": mb_data.get("cover_url") if mb_data else None,
        }

        embed_ok = embed_metadata(
            downloaded_file,
            song,
            artist,
            extra,
            thumbnail_url,
            fmt,
            lambda msg: self.events.on_warn(msg),
        )

        if not embed_ok:
            self.events.on_metadata_error(artist, song, downloaded_file.name)
            downloaded_file.unlink(missing_ok=True)
            result.reason = "Metadata integrity check failed"
            self._persist(state, state_lock, key, "failed", url, None, None, output_dir, needs_fp)
            return result

        md5 = compute_md5(downloaded_file)
        file_size = downloaded_file.stat().st_size

        result.status = "downloaded"
        result.file_path = downloaded_file
        result.file_size_bytes = file_size
        result.md5 = md5
        result.musicbrainz_enriched = mb_enriched
        result.album = extra.get("album")
        result.year = extra.get("year")
        result.genre = extra.get("genre")
        result.silence_ratio = silence_ratio
        result.duration_verified = dur_ok

        self._persist(
            state,
            state_lock,
            key,
            "downloaded",
            url,
            str(downloaded_file),
            md5,
            output_dir,
            needs_fp,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _iter_entries(self, data: dict[str, Any]):
        """Recursively flatten yt-dlp playlist/channel results into video entries."""
        entries = data.get("entries")
        if entries:
            for e in entries:
                if not e:
                    continue
                yield from self._iter_entries(e)
            return

        yield data

    def _download_partial(self, url: str, output_dir: Path) -> Optional[Path]:
        token = uuid4().hex[:8]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        partial_base = output_dir / f"_partial_{token}"
        expected_mp3 = partial_base.with_suffix(".mp3")

        ydl_opts: Any = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "outtmpl": str(partial_base) + ".%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "socket_timeout": 30,
            "retries": 5,
            "fragment_retries": 5,
            "extractor_retries": 5,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
            "extractor_args": {
                "youtube": {
                    "player_client": ["web"],
                }
            },
            "remote_components": ["ejs:github"],
        }

        node_path = shutil.which("node")
        if node_path:
            ydl_opts["js_runtimes"] = {
                "node": {"path": node_path}
            }

        if self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)
        if self.cookies_file:
            ydl_opts["cookiefile"] = str(self.cookies_file)
        if self.proxy:
            ydl_opts["proxy"] = self.proxy

        def _hook(d: dict) -> None:
            status = d.get("status")
            if status == "downloading":
                total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded_b = d.get("downloaded_bytes") or 0
                pct = (downloaded_b / total_b * 100.0) if total_b else 0.0
                self.events.on_download_progress(
                    "Partial",
                    url[:30],
                    pct,
                    d.get("speed") or 0.0,
                    downloaded_b,
                    total_b,
                )
            elif status == "finished":
                total_b = d.get("total_bytes") or d.get("downloaded_bytes") or 0
                self.events.on_download_progress("Partial", url[:30], 100.0, 0.0, total_b, total_b)

        ydl_opts["progress_hooks"] = [_hook]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return None

                source_file = Path(ydl.prepare_filename(info))
                final_mp3 = source_file.with_suffix(".mp3")

                if final_mp3.exists():
                    return final_mp3

                if expected_mp3.exists():
                    return expected_mp3

                for candidate in sorted(
                    output_dir.glob(f"_partial_{token}.*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ):
                    if candidate.exists():
                        return candidate

        except Exception:
            return None

        return None

    @staticmethod
    def _persist(
        state: dict,
        lock: threading.Lock,
        key: str,
        status: str,
        url: Optional[str],
        file_path: Optional[str],
        md5: Optional[str],
        output_dir: Path,
        fingerprint_verified: bool = False,
        preserve_timestamp: bool = False,
    ) -> None:
        with lock:
            downloads = state.setdefault("downloads", {})
            existing = downloads.get(key)

            if preserve_timestamp and existing and "timestamp" in existing:
                ts = existing["timestamp"]
            else:
                ts = datetime.now(timezone.utc).isoformat()

            downloads[key] = {
                "status": status,
                "url": url,
                "file_path": file_path,
                "md5": md5,
                "fingerprint_verified": fingerprint_verified,
                "timestamp": ts,
            }
            save_state(state, output_dir)
