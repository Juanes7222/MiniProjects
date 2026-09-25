from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config
from .events import DownloaderEvents
from .fingerprint import AcoustIDCircuitBreaker, verify_duration, verify_fingerprint
from .metadata import fetch_musicbrainz
from .result import DownloadResult
from .utils import apply_delay, compute_md5, sanitize_filename

_MIN_FILE_SIZE = 50 * 1024
_FINGERPRINT_ERRORS = frozenset(
    {
        "no_key",
        "circuit_breaker_open",
        "rate_limit_exceeded",
        "fingerprint_error",
        "max_retries_exceeded",
    }
)


def _failed_result(artist: str, song: str, reason: str) -> DownloadResult:
    return DownloadResult(artist=artist, song=song, status="failed", reason=reason)


def _verify_single(
    artist: str,
    song: str,
    output_dir: Path,
    fmt: str,
    acoustid_key: Optional[str],
    config: Config,
    circuit_breaker: AcoustIDCircuitBreaker,
    fingerprint_semaphore: threading.Semaphore,
    musicbrainz: bool,
    events: DownloaderEvents,
    stop_event: threading.Event,
    require_fingerprint: bool = False,
) -> DownloadResult:
    if stop_event.is_set():
        return DownloadResult(artist=artist, song=song, status="skipped", reason="Interrupted")

    expected_file = output_dir / sanitize_filename(artist) / f"{sanitize_filename(song)}.{fmt}"
    if not expected_file.is_file():
        return _failed_result(artist, song, "File does not exist")

    try:
        file_size = expected_file.stat().st_size
    except OSError:
        return _failed_result(artist, song, "Could not read file size")

    if file_size < _MIN_FILE_SIZE:
        result = _failed_result(artist, song, "File is extremely small (<50KB)")
        result.file_path = expected_file
        return result

    duration_ok, actual_duration = verify_duration(expected_file, 0)
    if not duration_ok and actual_duration == 0:
        result = _failed_result(artist, song, "Corrupted file or missing metadata")
        result.file_path = expected_file
        return result

    result = DownloadResult(
        artist=artist,
        song=song,
        duration_seconds=actual_duration,
        file_path=expected_file,
        file_size_bytes=file_size,
    )

    if acoustid_key:
        apply_delay(0.2, 0.5)
        if musicbrainz:
            try:
                musicbrainz_data = fetch_musicbrainz(artist, song)
            except Exception:
                musicbrainz_data = None
            if musicbrainz_data:
                result.musicbrainz_enriched = True
                if "duration_seconds" in musicbrainz_data:
                    result.duration_seconds = musicbrainz_data["duration_seconds"]

        with fingerprint_semaphore:
            fingerprint_ok, confidence, matched_title = verify_fingerprint(
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

        result.fingerprint_verified = fingerprint_ok
        result.fingerprint_confidence = confidence
        result.fingerprint_matched_title = matched_title
        if fingerprint_ok:
            result.fingerprint_label = f"verified {confidence:.0%}"
        elif matched_title in _FINGERPRINT_ERRORS:
            result.fingerprint_label = matched_title
        elif confidence > 0:
            result.fingerprint_label = f"no match ({matched_title or 'unknown'})"
        else:
            result.fingerprint_label = "no match"

        if not fingerprint_ok and (require_fingerprint or confidence > 0):
            reason = (
                "Fingerprint did not confirm the song"
                if require_fingerprint
                else f"Fingerprint mismatch: Found '{matched_title}' ({confidence * 100:.1f}%)"
            )
            result.status = "failed"
            result.reason = reason
            return result

    result.status = "verified"
    return result


def _restore_cached_result(
    artist: str,
    song: str,
    output_dir: Path,
    fmt: str,
    entry_state: dict,
    require_fingerprint: bool,
) -> Optional[DownloadResult]:
    if entry_state.get("status") != "verified":
        return None
    if require_fingerprint and not entry_state.get("fingerprint_verified", False):
        return None

    raw_path = entry_state.get("file_path")
    if not raw_path:
        return None
    file_path = Path(raw_path)
    expected_file = output_dir / sanitize_filename(artist) / f"{sanitize_filename(song)}.{fmt}"
    if file_path != expected_file or not file_path.is_file():
        return None

    try:
        file_size = file_path.stat().st_size
        if file_size < _MIN_FILE_SIZE:
            return None
        _, duration = verify_duration(file_path, 0)
        if duration == 0:
            return None
        stored_md5 = entry_state.get("md5")
        if not stored_md5:
            return None
        current_md5 = compute_md5(file_path)
        if current_md5 != stored_md5:
            return None
    except OSError:
        return None

    result = DownloadResult(
        artist=artist,
        song=song,
        status="verified",
        duration_seconds=duration,
        file_path=file_path,
        file_size_bytes=file_size,
        md5=current_md5,
        fingerprint_verified=entry_state.get("fingerprint_verified", False),
        fingerprint_confidence=entry_state.get("fingerprint_confidence", 0.0),
        fingerprint_matched_title=entry_state.get("fingerprint_matched_title"),
    )
    if result.fingerprint_verified:
        result.fingerprint_label = entry_state.get("fingerprint_label", "verified (stored)")
    return result


def verify_library(
    songs: dict[str, list[str]],
    output_dir: Path,
    fmt: str,
    workers: int,
    acoustid_key: Optional[str],
    config: Config,
    circuit_breaker: AcoustIDCircuitBreaker,
    fingerprint_semaphore: threading.Semaphore,
    musicbrainz: bool,
    events: DownloaderEvents,
    persist_fn: Callable[..., None],
    state: dict,
    state_lock: threading.Lock,
    require_fingerprint: bool = False,
    state_filename: str | None = None,
) -> list[DownloadResult]:
    output_dir = Path(output_dir)
    results_map: dict[tuple[str, str], DownloadResult] = {}
    pairs_to_process: list[tuple[str, str]] = []

    for artist, artist_songs in songs.items():
        for song in artist_songs:
            key = f"{artist}::{song}"
            entry_state = state.get("downloads", {}).get(key, {})
            cached_result = _restore_cached_result(
                artist,
                song,
                output_dir,
                fmt,
                entry_state,
                require_fingerprint,
            )
            if cached_result:
                results_map[(artist, song)] = cached_result
            else:
                results_map[(artist, song)] = DownloadResult(
                    artist=artist,
                    song=song,
                    status="skipped",
                    reason="Not verified in state",
                )
                pairs_to_process.append((artist, song))

    events.on_session_start(len(pairs_to_process), is_verify=True)
    stop_event = threading.Event()
    start = time.monotonic()
    artist_counts = Counter(artist for artist, _ in pairs_to_process)
    seen_artists: set[str] = set()
    seen_artists_lock = threading.Lock()

    def verify_with_artist_tracking(artist: str, song: str) -> DownloadResult:
        with seen_artists_lock:
            if artist not in seen_artists:
                seen_artists.add(artist)
                events.on_artist_start(artist, artist_counts[artist])
        return _verify_single(
            artist,
            song,
            output_dir,
            fmt,
            acoustid_key,
            config,
            circuit_breaker,
            fingerprint_semaphore,
            musicbrainz,
            events,
            stop_event,
            require_fingerprint,
        )

    if pairs_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(verify_with_artist_tracking, artist, song): (artist, song)
                for artist, song in pairs_to_process
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    artist, song = futures[future]
                    key = f"{artist}::{song}"
                    try:
                        result = future.result()
                    except Exception as error:
                        result = _failed_result(
                            artist,
                            song,
                            f"Exception: {type(error).__name__}: {error}",
                        )

                    results_map[(artist, song)] = result
                    events.on_result(result)

                    if result.status == "skipped":
                        continue
                    with state_lock:
                        existing = dict(state.get("downloads", {}).get(key, {}))
                    current_md5 = (
                        compute_md5(Path(result.file_path))
                        if result.file_path and Path(result.file_path).is_file()
                        else existing.get("md5")
                    )
                    persistence_options: dict[str, Any] = {
                        "fingerprint_verified": result.fingerprint_verified,
                        "fingerprint_confidence": result.fingerprint_confidence,
                        "fingerprint_label": result.fingerprint_label,
                        "preserve_timestamp": True,
                    }
                    if state_filename is not None:
                        persistence_options["state_filename"] = state_filename
                    persist_fn(
                        state,
                        state_lock,
                        key,
                        result.status,
                        existing.get("url"),
                        str(result.file_path) if result.file_path else existing.get("file_path"),
                        current_md5,
                        output_dir,
                        **persistence_options,
                    )
            except KeyboardInterrupt:
                stop_event.set()
                for future in futures:
                    future.cancel()

    results = list(results_map.values())
    events.on_session_complete(results, time.monotonic() - start)
    return results
