"""
Audio fingerprinting, silence detection, and duration verification.

Standalone functions extracted from MusicDownloader for testability
and reuse outside the main class.
"""

from __future__ import annotations

import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import acoustid
import mutagen
from rapidfuzz import fuzz

from .config import Config


class AcoustIDCircuitBreaker:
    """
    Thread-safe circuit breaker for the AcoustID API.

    When too many rate-limit errors accumulate, the breaker opens and
    suspends all fingerprinting for a configurable cooldown period.
    """

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        self._open = False
        self._cooldown_until = 0.0
        self._lock = threading.Lock()
        self._cooldown_duration = cooldown_seconds

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._open:
                if time.time() < self._cooldown_until:
                    return True
                # Cooldown expired — close the breaker
                self._open = False
            return False

    def trip(self) -> None:
        """Open the circuit breaker."""
        with self._lock:
            if not self._open:
                self._open = True
                self._cooldown_until = time.time() + self._cooldown_duration


def verify_fingerprint(
    partial_path: Path,
    artist: str,
    song: str,
    acoustid_key: str,
    config: Config,
    circuit_breaker: AcoustIDCircuitBreaker,
    on_warn: Optional[callable] = None,
    on_info: Optional[callable] = None,
    on_fingerprint_error: Optional[callable] = None,
) -> tuple[bool, float, str]:
    """
    Verify an audio file's fingerprint against AcoustID.

    Returns
    -------
    (is_match, confidence, matched_title)
    """
    if not acoustid_key:
        return False, 0.0, "no_key"

    if circuit_breaker.is_open:
        return False, 0.0, "circuit_breaker_open"

    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            # Random delay to be polite to the API
            stop_time = random.uniform(3.0, 7.0)
            if on_warn:
                on_warn(
                    f"[yellow]Applying random delay of {stop_time:.2f}s "
                    f"before AcoustID request...[/yellow]"
                )
            time.sleep(stop_time)

            results = list(acoustid.match(acoustid_key, str(partial_path), meta="recordings"))
            best_conf = 0.0
            best_title = ""

            for score, _rec_id, title, a in results:
                if score < config.FINGERPRINT_MIN_CONFIDENCE:
                    continue
                a_sim = fuzz.token_sort_ratio(artist.lower(), (a or "").lower())
                t_sim = fuzz.token_sort_ratio(song.lower(), (title or "").lower())
                if a_sim > 75 and t_sim > 75:
                    if on_info:
                        on_info(
                            f"[green]Fingerprint match: '{a} - {title}' "
                            f"with confidence {score:.2f} "
                            f"(artist sim: {a_sim}, title sim: {t_sim})[/green]"
                        )
                    return True, score, title or ""
                if score > best_conf:
                    best_conf = score
                    best_title = f"{a} -- {title}"
                    if on_info:
                        on_info(
                            f"[yellow]Best match found: {best_title} "
                            f"(confidence: {best_conf:.2f})[/yellow]"
                        )
            return False, best_conf, best_title

        except Exception as exc:
            err_str = str(exc).lower()

            if "error" in err_str or "rate limit" in err_str or "429" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2**attempt)
                    if on_warn:
                        on_warn(
                            f"[yellow]AcoustID rate limit hit. "
                            f"Local retry in {sleep_time}s...[/yellow]"
                        )
                    time.sleep(sleep_time)
                    continue
                else:
                    circuit_breaker.trip()
                    if on_warn:
                        on_warn(
                            f"[red]CRITICAL: AcoustID API blocked. "
                            f"Circuit Breaker OPEN. "
                            f"Suspending all fingerprinting for "
                            f"{int(circuit_breaker._cooldown_duration)} seconds.[/red]"
                        )
                    return False, 0.0, "rate_limit_exceeded"

            if on_fingerprint_error:
                on_fingerprint_error(artist, song, str(exc))
            return False, 0.0, "fingerprint_error"

    return False, 0.0, "max_retries_exceeded"


def has_excessive_silence(file_path: Path, config: Config) -> tuple[bool, float]:
    """
    Detect excessive silence in an audio file using ffmpeg's silencedetect filter.

    Returns
    -------
    (is_excessive, silence_ratio)
    """
    try:
        min_dur_sec = config.SILENCE_MIN_DURATION_MS / 1000.0
        thresh_db = config.SILENCE_THRESHOLD_DB

        cmd = [
            "ffmpeg",
            "-v",
            "info",
            "-nostdin",
            "-i",
            str(file_path),
            "-af",
            f"silencedetect=noise={thresh_db}dB:d={min_dur_sec}",
            "-f",
            "null",
            "-",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        silences = [
            float(match) for match in re.findall(r"silence_duration: ([\d\.]+)", res.stderr)
        ]
        total_silence_sec = sum(silences)

        dur_match = re.search(r"Duration: (\d{2}):(\d{2}):([\d\.]+)", res.stderr)
        if not dur_match:
            return False, 0.0

        h, m, s = dur_match.groups()
        total_dur_sec = int(h) * 3600 + int(m) * 60 + float(s)

        if total_dur_sec <= 0:
            return False, 0.0

        ratio = total_silence_sec / total_dur_sec
        return ratio > config.EXCESSIVE_SILENCE_RATIO, ratio
    except Exception:
        return False, 0.0


def verify_duration(
    path: Path, expected: int, tolerance: float = 0.20
) -> tuple[bool, int]:
    """
    Compare the actual audio duration against the expected value.

    Returns
    -------
    (is_ok, actual_seconds)
    """
    try:
        info = mutagen.File(str(path))  # type: ignore
        if info is None or info.info is None:
            return False, 0
        actual = int(info.info.length)
        if expected == 0:
            return True, actual
        ratio = abs(actual - expected) / max(expected, 1)
        return ratio <= tolerance, actual
    except Exception:
        return False, 0
