"""
Audio fingerprinting, silence detection, and duration verification.

Standalone functions extracted from MusicDownloader for testability
and reuse outside the main class.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import acoustid
import mutagen
from rapidfuzz import fuzz

from .cache import caches
from .config import Config
from .ratelimit import CircuitBreaker, full_jitter_backoff, is_rate_limit_error, limiters


def shutil_which(command: str) -> str | None:
    return shutil.which(command)

# AcoustID's free tier allows three requests per second, process-wide. pyacoustid
# ships its own limiter for the same rule; two limiters on one budget means two
# locks, two queues and no clearer picture of what is actually being spent, so we
# disable the library's and own the budget explicitly. That also lets us pace the
# API call alone instead of the whole fingerprint block.
try:  # pragma: no cover - depends on the installed pyacoustid version
    acoustid.REQUEST_INTERVAL = 0
except Exception:  # pragma: no cover
    pass


def find_fpcalc() -> str | None:
    """Locate the Chromaprint ``fpcalc`` binary, or None.

    Checks ``PATH`` first, then the directories a checkout would plausibly have
    dropped the binary in. The second check is not a convenience: a bundled
    ``fpcalc.exe`` sitting in the project root is invisible to ``shutil.which``,
    so without it the whole verification stage silently disables itself. That
    failure is invisible by construction -- there is no AcoustID key to be
    refused, no fingerprint cache to miss, and every result still comes back
    "downloaded" -- and it took the learned channel-trust model down with it,
    since ``fingerprint_verified`` was never set, so its verified multiplier never
    fired either.

    Resolved to an absolute path. On Windows ``shutil.which`` answers with a
    *relative* name when the binary happens to sit in the current directory, and
    pyacoustid hands whatever it gets straight to ``subprocess`` -- so a path that
    works from one directory and silently breaks from another is not a path.
    """
    found = shutil_which("fpcalc")
    if found and os.path.isfile(found):
        return str(Path(found).resolve())
    package_root = Path(__file__).resolve().parent
    roots = (package_root, package_root.parent, package_root.parent / "tools")
    names = ("fpcalc.exe", "fpcalc") if os.name == "nt" else ("fpcalc",)
    for root in roots:
        for name in names:
            candidate = root / name
            try:
                if candidate.is_file():
                    return str(candidate.resolve())
            except OSError:
                continue
    return None


_fpcalc_command: str | None = None


def configure_fpcalc() -> str | None:
    """Point pyacoustid at the discovered ``fpcalc``, once. Returns it or None.

    Worth being precise about what this does and does not buy.

    pyacoustid picks its backend with ``have_audioread and have_chromaprint and
    not force_fpcalc`` -- note the conjunction. ``have_chromaprint`` reports
    whether the ``chromaprint`` **C extension** imported, and where it did not,
    the pure-Python ``audioread`` FFT is never preferred regardless of
    ``force_fpcalc``; the command-line tool runs either way. Measured here with
    the flag both ways over 16 MB files, the timings are indistinguishable
    (0.5-0.9 s), because both calls take the same path.

    So the load-bearing part of this function is not the flag, it is the
    **absolute path**: ``acoustid.FPCALC_COMMAND`` is what the library spawns, and
    before it was pointed at a resolved binary the whole verification stage
    depended on the process's working directory containing it. ``force_fpcalc``
    is kept as a guarantee rather than an optimisation -- it pins the tool even on
    an interpreter where the C extension *is* present, where the default would be
    the Python FFT.
    """
    global _fpcalc_command
    if _fpcalc_command is None:
        found = find_fpcalc()
        _fpcalc_command = found or ""
        if found:
            acoustid.FPCALC_COMMAND = found
    return _fpcalc_command or None


def acoustid_bucket(config: Optional[Config] = None):
    """The process-wide AcoustID request bucket."""
    cfg = config or Config()
    return limiters.bucket(
        "acoustid",
        cfg.ACOUSTID_RATE_PER_SECOND,
        burst=cfg.ACOUSTID_BURST,
    )


class AcoustIDCircuitBreaker(CircuitBreaker):
    """Circuit breaker for the AcoustID API.

    Repeated throttling widens the cooldown instead of re-serving a fixed one, and
    the first request after a cooldown is admitted alone as a probe, so a
    recovered service is not immediately buried again by every queued worker.
    """

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        super().__init__(cooldown_seconds=cooldown_seconds, max_cooldown_seconds=900.0)


@dataclass(frozen=True)
class FingerprintVerdict:
    """The outcome of one AcoustID lookup."""

    verified: bool
    confidence: float
    matched_title: Optional[str]

    def as_tuple(self) -> tuple[bool, float, Optional[str]]:
        return self.verified, self.confidence, self.matched_title


class FingerprintCache:
    """Remembers the verdict for a (url, artist, song) triple.

    The same upload turns up as a candidate for several songs of an artist, and
    the same song is re-checked on every ``--retry``. Each of those checks costs
    one of three requests per second, so a remembered verdict is budget the run
    does not have to spend. Keyed by URL *and* by what we are looking for: the
    same file verified as one song says nothing about another.
    """

    def __init__(self, config: Optional[Config] = None, enabled: Optional[bool] = None) -> None:
        cfg = config or Config()
        self._cache = caches.get("fingerprint")
        if enabled is not None:
            self._cache.enabled = bool(enabled) and cfg.CACHE_ENABLED
        self._ttl = cfg.FINGERPRINT_CACHE_TTL
        self._lock = threading.Lock()
        self._in_flight: dict[str, threading.Event] = {}

    @staticmethod
    def _key(url: str, artist: str, song: str) -> str:
        return caches.get("fingerprint").make_key(url, artist.lower(), song.lower())

    def get(self, url: str, artist: str, song: str) -> Optional[FingerprintVerdict]:
        if not self._cache.enabled or not url:
            return None
        payload = self._cache.get_json(self._key(url, artist, song))
        if not isinstance(payload, (list, tuple)) or len(payload) != 3:
            return None
        verified, confidence, title = payload
        return FingerprintVerdict(bool(verified), float(confidence), title)

    def put(self, url: str, artist: str, song: str, verdict: FingerprintVerdict) -> None:
        if not self._cache.enabled or not url:
            return
        self._cache.put_json(
            self._key(url, artist, song),
            [verdict.verified, verdict.confidence, verdict.matched_title],
            self._ttl,
        )

    def coalesce(self, url: str, artist: str, song: str, timeout: float = 120.0) -> bool:
        """Wait for another worker already checking this exact triple.

        Returns True when the caller should skip its own check and re-read the
        cache. Without this, N workers holding the same candidate each spend a
        request to learn the same answer.
        """
        if not self._cache.enabled or not url:
            return False
        key = self._key(url, artist, song)
        with self._lock:
            event = self._in_flight.get(key)
            if event is None:
                self._in_flight[key] = event = threading.Event()
                return False
        event.wait(timeout)
        return True


def release_fingerprint_slot(cache: FingerprintCache, url: str, artist: str, song: str) -> None:
    """Signal waiters that a coalesced fingerprint check has finished."""
    key = cache._key(url, artist, song)  # noqa: SLF001 - same module family
    with cache._lock:  # noqa: SLF001
        event = cache._in_flight.pop(key, None)
    if event is not None:
        event.set()


def _artist_stem(name: str) -> str:
    """Return the leading artist name, dropping feat./ft./featuring clauses.

    'Barak feat. Marcos Yaroide' -> 'Barak'
    """
    if not name:
        return ""
    return re.split(r"\s+(?:feat\.?|ft\.?|featuring|con)\s+", name, flags=re.IGNORECASE)[0].strip()


def _interruptible_sleep(seconds: float, circuit_breaker: AcoustIDCircuitBreaker) -> None:
    """Sleep in short slices, giving up as soon as the breaker opens.

    Waiting out a full backoff while the circuit is already open is time the run
    cannot get back.
    """
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        if circuit_breaker.is_open:
            return
        slice_seconds = min(0.25, remaining)
        time.sleep(slice_seconds)
        remaining -= slice_seconds


def _score_matches(results, artist: str, song: str, config: Config, on_info) -> tuple[bool, float, str]:
    """Pick the best AcoustID result for the requested artist and song."""
    best_conf = 0.0
    best_title = ""
    for score, _rec_id, title, a in results:
        if score < config.FINGERPRINT_MIN_CONFIDENCE:
            continue
        a_sim = fuzz.token_sort_ratio(_artist_stem(artist).lower(), _artist_stem(a).lower())
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


def _lookup(
    acoustid_key: str,
    path: Path,
    expected_duration: Optional[int],
    *,
    force_fpcalc: bool = False,
):
    """Fingerprint *path* and look it up, declaring the **track's** duration.

    ``acoustid.match`` reads the duration off the file it is given. For a
    90-second excerpt that is 90 seconds, and AcoustID then searches for a
    recording roughly that long, which is not the recording the audio came from.
    So the lookup is done here instead: fingerprint exactly as ``match`` would,
    but pass the duration the caller knows the whole track has.

    Falls back to ``acoustid.match`` when no duration is known, which is the
    honest behaviour -- a wrong duration is worse than the file's own, and the
    caller's figure comes from the search result or the catalogue.

    Mirrors ``acoustid.match(parse=True)`` so the caller keeps getting the same
    ``(score, id, title, artist)`` tuples.
    """
    if not expected_duration or int(expected_duration) <= 0:
        return acoustid.match(
            acoustid_key, str(path), meta="recordings", force_fpcalc=force_fpcalc
        )
    duration, fingerprint = acoustid.fingerprint_file(str(path), force_fpcalc=force_fpcalc)
    response = acoustid.lookup(
        acoustid_key, fingerprint, int(expected_duration), meta="recordings"
    )
    return acoustid.parse_lookup_result(response)


def verify_fingerprint(
    partial_path: Path,
    artist: str,
    song: str,
    acoustid_key: str,
    config: Config,
    circuit_breaker: AcoustIDCircuitBreaker,
    expected_duration: Optional[int] = None,
    on_warn: Optional[Callable[[str], None]] = None,
    on_info: Optional[Callable[[str], None]] = None,
    on_fingerprint_error: Optional[Callable[[str, str, str], None]] = None,
) -> tuple[bool, float, str]:
    """
    Verify an audio file's fingerprint against AcoustID.

    ``expected_duration`` is the length of the **whole track**, and it is not
    optional in practice. AcoustID narrows its candidate set by the duration it is
    told, so a fingerprint taken from a 90-second excerpt submitted with that
    excerpt's own 90 seconds finds nothing: measured on one track, declaring 90,
    120, 180, 240 and 300 seconds all returned no match, while declaring the
    track's real 340 seconds matched at 0.95 from the very same 90 seconds of
    audio. The window is tight -- 400 and 600 were also misses -- so it has to be
    the real figure, not a generous one.

    Returns
    -------
    (is_match, confidence, matched_title)
    """
    if not acoustid_key:
        return False, 0.0, "no_key"

    if not circuit_breaker.allow():
        return False, 0.0, "circuit_breaker_open"

    # Use the command-line tool when we have it. pyacoustid would otherwise pick
    # its pure-Python audioread backend, because the chromaprint C extension is
    # absent, and pay an FFT in Python for every 90-second clip.
    fpcalc = configure_fpcalc()

    bucket = acoustid_bucket(config)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Paces the API call itself -- the fpcalc/partial work around it runs
            # at full speed instead of waiting behind the rate limit.
            bucket.acquire()

            results = list(_lookup(
                acoustid_key,
                partial_path,
                expected_duration,
                force_fpcalc=bool(fpcalc),
            ))
            circuit_breaker.record_success()
            return _score_matches(results, artist, song, config, on_info)

        except Exception as exc:
            message = str(exc)
            if is_rate_limit_error(message) or "error" in message.lower():
                # The service told us to slow down. Spend the budget for the
                # window it asked for up front so the other in-flight requests
                # do not immediately retry in lockstep behind this one.
                bucket.penalty(config.RETRY_BACKOFF_BASE)
                if attempt < max_retries - 1:
                    sleep_time = full_jitter_backoff(
                        attempt + 1,
                        base=config.RETRY_BACKOFF_BASE,
                        cap=config.RETRY_BACKOFF_CAP,
                    )
                    if on_warn:
                        on_warn(
                            f"[yellow]AcoustID rate limit hit. "
                            f"Local retry in {sleep_time:.1f}s...[/yellow]"
                        )
                    _interruptible_sleep(sleep_time, circuit_breaker)
                    continue
                circuit_breaker.trip()
                if on_warn:
                    on_warn(
                        f"[red]CRITICAL: AcoustID API blocked. "
                        f"Circuit Breaker OPEN. "
                        f"Suspending all fingerprinting for "
                        f"{int(circuit_breaker.cooldown_remaining())} seconds.[/red]"
                    )
                return False, 0.0, "rate_limit_exceeded"

            if on_fingerprint_error:
                on_fingerprint_error(artist, song, str(exc))
            circuit_breaker.release_probe()
            return False, 0.0, "fingerprint_error"

    circuit_breaker.release_probe()
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
            # Decode single-threaded. This runs in the CPU-bound stage alongside
            # yt-dlp's own transcodes, and letting every silencedetect grab
            # every core is what makes both of them slow.
            "-threads",
            "1",
            "-i",
            str(file_path),
            # Decimate before filtering. Deciding whether a passage is quieter
            # than -50 dB does not need 44.1 kHz stereo: ffmpeg resamples once,
            # the filter then works on ~1/11th of the samples, and the whole
            # decode gets correspondingly cheaper. This is a second full pass
            # over an already-encoded file -- the audio was decoded once already
            # to produce it -- so its cost is pure overhead, and it was the
            # largest single item in the post-download stage.
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(config.SILENCE_DETECT_SAMPLE_RATE),
            "-af",
            f"silencedetect=noise={thresh_db}dB:d={min_dur_sec}",
            "-f",
            "null",
            "-",
        ]

        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        # ffmpeg writes progress to one long stderr stream; scanning it for the
        # two patterns we need is cheaper than splitting every line into a list
        # and matching each one.
        silences = [float(m) for m in re.findall(r"silence_duration: ([\d.]+)", res.stderr)]
        total_silence_sec = sum(silences)

        dur_match = re.search(r"Duration: (\d{2}):(\d{2}):([\d.]+)", res.stderr)
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


def verify_duration(path: Path, expected: int, tolerance: float = 0.20) -> tuple[bool, int]:
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
