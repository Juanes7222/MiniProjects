"""
Library verification: validates existence, size, duration, and fingerprint
of local audio tracks.

Standalone functions extracted from ``MusicDownloader.verify_library`` so
they can be tested and reused independently.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Optional

from .config import Config
from .events import DownloaderEvents
from .fingerprint import AcoustIDCircuitBreaker, verify_duration, verify_fingerprint
from .metadata import fetch_musicbrainz
from .result import DownloadResult
from .state import load_state
from .utils import apply_delay, compute_md5, sanitize_filename


def _verify_single(
    artist: str,
    song: str,
    output_dir: Path,
    fmt: str,
    acoustid_key: Optional[str],
    config: Config,
    circuit_breaker: AcoustIDCircuitBreaker,
    fp_semaphore: threading.Semaphore,
    musicbrainz: bool,
    events: DownloaderEvents,
    stop_event: threading.Event,
) -> DownloadResult:
    """Verify a single audio file on disk."""
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

    if acoustid_key:
        apply_delay(0.2, 0.5)
        if musicbrainz:
            try:
                mb_data = fetch_musicbrainz(artist, song)
                if mb_data:
                    result.musicbrainz_enriched = True
                    if "duration_seconds" in mb_data:
                        result.duration_seconds = mb_data["duration_seconds"]
            except Exception:
                pass

        with fp_semaphore:
            fp_ok, fp_conf, fp_title = verify_fingerprint(
                expected_file,
                artist,
                song,
                acoustid_key,
                config,
                circuit_breaker,
                on_warn=events.on_warn,
                on_info=events.on_info,
                on_fingerprint_error=events.on_fingerprint_error,
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


def verify_library(
    songs: dict[str, list[str]],
    output_dir: Path,
    fmt: str,
    workers: int,
    acoustid_key: Optional[str],
    config: Config,
    circuit_breaker: AcoustIDCircuitBreaker,
    fp_semaphore: threading.Semaphore,
    musicbrainz: bool,
    events: DownloaderEvents,
    persist_fn: callable,
    state: dict,
    state_lock: threading.Lock,
) -> list[DownloadResult]:
    """
    Verify a local library of audio files.

    Parameters
    ----------
    persist_fn:
        Callable to persist state (typically ``MusicDownloader._persist``).
    state / state_lock:
        Shared state dict and its lock.

    Returns
    -------
    List of ``DownloadResult`` for every song in *songs*.
    """
    results_map: dict[tuple[str, str], DownloadResult] = {}
    pairs_to_process: list[tuple[str, str]] = []

    for artist, lst in songs.items():
        if not lst:
            continue
        for song in lst:
            key = f"{artist}::{song}"
            entry_state = state.get("downloads", {}).get(key, {})
            status = entry_state.get("status")

            if status == "verified":
                res = DownloadResult(artist=artist, song=song, status="verified")
                res.file_path = (
                    Path(entry_state["file_path"]) if entry_state.get("file_path") else None
                )
                res.md5 = entry_state.get("md5")
                res.fingerprint_verified = entry_state.get("fingerprint_verified", False)
                results_map[(artist, song)] = res
            else:
                results_map[(artist, song)] = DownloadResult(
                    artist=artist, song=song, status="skipped", reason="Not verified in state"
                )
                if status == "downloaded":
                    pairs_to_process.append((artist, song))

    total = len(pairs_to_process)
    events.on_session_start(total, is_verify=True)

    stop_event = threading.Event()
    start = time.monotonic()

    seen_artists: set[str] = set()
    seen_artists_lock = threading.Lock()

    def _verify_with_artist_tracking(artist: str, song: str) -> DownloadResult:
        with seen_artists_lock:
            if artist not in seen_artists:
                seen_artists.add(artist)
                count = sum(1 for a, _ in pairs_to_process if a == artist)
                events.on_artist_start(artist, count)
        return _verify_single(
            artist, song, output_dir, fmt, acoustid_key, config,
            circuit_breaker, fp_semaphore, musicbrainz, events, stop_event,
        )

    if pairs_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_verify_with_artist_tracking, artist, song): (artist, song)
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
                            artist=artist,
                            song=song,
                            status="failed",
                            reason=f"Exception: {str(exc)}",
                        )

                    results_map[(artist, song)] = res
                    events.on_result(res)

                    if res.status != "skipped":
                        with state_lock:
                            existing = state.get("downloads", {}).get(key, {})

                        current_md5 = existing.get("md5")
                        if not current_md5 and res.file_path and Path(res.file_path).exists():
                            current_md5 = compute_md5(Path(res.file_path))

                        fingerprint_verified = getattr(res, "fingerprint_verified", False)

                        persist_fn(
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
    events.on_session_complete(all_results, elapsed)

    return all_results
