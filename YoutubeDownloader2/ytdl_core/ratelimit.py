"""
Rate limiting, jittered backoff and circuit breaking for remote services.

Every remote service this project talks to publishes a budget, and every one of
them is a *global* resource rather than a per-song one: three requests per
second to AcoustID is three per second for the whole run, no matter how many
songs are in flight. Sleeping blindly is the wrong tool for that -- it burns
wall-clock on songs that could have run, and it still fails to protect the
service when the sleep is shorter than the real round-trip.

The three primitives here replace the fixed sleeps that used to be scattered
across the pipeline:

* :class:`TokenBucket` -- "I may burst up to *burst* requests, then sustain
  *rate* per second". Blocks only when the caller would actually exceed the
  budget, and never holds its lock while sleeping, so queued threads keep
  draining instead of convoying behind one sleeper.
* :func:`full_jitter_backoff` -- exponential backoff with full jitter. Without
  jitter, N workers that fail together retry together, and the retry storm is
  what keeps a struggling service down.
* :class:`CircuitBreaker` -- CLOSED -> OPEN -> HALF_OPEN with an exponentially
  growing cooldown, so a service that has banned us is left alone for longer and
  longer while we are actually healthy.
"""

from __future__ import annotations

import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Optional

__all__ = [
    "TokenBucket",
    "CircuitBreaker",
    "RateLimiterRegistry",
    "limiters",
    "full_jitter_backoff",
    "parse_retry_after",
    "is_rate_limit_error",
]


class TokenBucket:
    """Thread-safe token bucket.

    Parameters
    ----------
    rate_per_second:
        Sustained rate. Tokens accrue at this rate; ``0`` or less disables
        limiting entirely, which is how a caller opts out.
    burst:
        Maximum tokens held at once. Defaults to ``max(1, rate)`` so a service
        with a per-second budget can absorb a short burst but not a stampede.
    clock / sleep:
        Injectable for tests. ``sleep`` must be the *blocking* sleep; the
        bucket releases its lock before calling it.
    """

    __slots__ = (
        "_rate",
        "_burst",
        "_tokens",
        "_updated",
        "_blocked_until",
        "_lock",
        "_clock",
        "_sleep",
    )

    def __init__(
        self,
        rate_per_second: float,
        burst: Optional[float] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rate = max(0.0, float(rate_per_second))
        self._burst = max(1.0, float(burst if burst is not None else max(1.0, self._rate)))
        self._tokens = self._burst
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    @property
    def rate_per_second(self) -> float:
        return self._rate

    def _refill_locked(self, now: float) -> None:
        if now < self._blocked_until:
            # A penalty is in force: no accrual, so the budget really is closed.
            self._updated = now
            return
        elapsed = now - self._updated
        if elapsed <= 0:
            return
        self._updated = now
        if self._rate <= 0:
            return
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)

    def _wait_seconds_locked(self, now: float) -> float:
        """How long until one token could plausibly be available."""
        deficit = 1.0 - self._tokens
        by_rate = deficit / self._rate if self._rate > 0 else 0.0
        by_penalty = max(0.0, self._blocked_until - now)
        return max(by_rate, by_penalty)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take *tokens* if they are available right now. Never blocks."""
        if self._rate <= 0:
            return True
        with self._lock:
            now = self._clock()
            self._refill_locked(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until *tokens* are available. Returns the seconds spent waiting.

        The wait happens *outside* the lock on purpose. Holding a lock across a
        sleep converts a rate limit into a mutex: every other thread piles up
        behind the sleeper and then re-checks to find the bucket still empty.
        """
        if self._rate <= 0:
            return 0.0

        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._refill_locked(now)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                delay = max(0.001, self._wait_seconds_locked(now))
            self._sleep(delay)
            waited += delay

    def penalty(self, seconds: float) -> None:
        """Close the budget for *seconds* after the service asked us to back off.

        The tokens already in hand are spent as well, so a burst of requests that
        all came back throttled does not immediately retry in lockstep -- each
        one would otherwise find a full bucket waiting for it.
        """
        if self._rate <= 0 or seconds <= 0:
            return
        with self._lock:
            now = self._clock()
            self._refill_locked(now)
            self._tokens = 0.0
            self._blocked_until = max(self._blocked_until, now + float(seconds))

    def reset(self) -> None:
        with self._lock:
            self._tokens = self._burst
            self._updated = self._clock()
            self._blocked_until = 0.0


class CircuitBreaker:
    """CLOSED -> OPEN -> HALF_OPEN breaker with an exponential cooldown.

    ``is_open`` answers "is this service currently refusing us?". It goes back
    to ``False`` once the cooldown expires, which is what lets a caller probe
    again. :meth:`allow` is the stricter gate: in HALF_OPEN it admits exactly
    one probe so a recovering service is not immediately buried again by every
    queued worker.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"

    def __init__(
        self,
        cooldown_seconds: float = 60.0,
        *,
        max_cooldown_seconds: float = 900.0,
        factor: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Kept under the historical private name: the fingerprint module and its
        # tests both read it to describe the suspension window to the user.
        self._cooldown_duration = float(cooldown_seconds)
        self._base_cooldown = float(cooldown_seconds)
        self._max_cooldown = float(max_cooldown_seconds)
        self._factor = max(1.0, float(factor))
        self._clock = clock
        self._state = self.CLOSED
        self._open_until = 0.0
        self._consecutive_failures = 0
        self._probe_in_flight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def cooldown_seconds(self) -> float:
        """The cooldown the current/last OPEN window used."""
        with self._lock:
            return self._cooldown_duration

    def cooldown_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._open_until - self._clock())

    @property
    def is_open(self) -> bool:
        """True only while the service is actively refusing us."""
        with self._lock:
            if self._state == self.OPEN:
                if self._clock() >= self._open_until:
                    # Cooldown served: let exactly one caller probe the service.
                    self._state = self.HALF_OPEN
                    self._probe_in_flight = False
                    return False
                return True
            return False

    def allow(self) -> bool:
        """Whether the caller may make a request right now."""
        with self._lock:
            if self._state == self.CLOSED:
                return True
            if self._state == self.OPEN:
                if self._clock() >= self._open_until:
                    self._state = self.HALF_OPEN
                    self._probe_in_flight = True
                    return True
                return False
            # HALF_OPEN: one probe at a time.
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def trip(self) -> None:
        """Record a failure and open (or widen) the circuit."""
        with self._lock:
            if self._state == self.OPEN:
                # Already refusing: do not extend the window and do not count a
                # second failure, or a burst of parallel errors would compound
                # into an ever-growing cooldown.
                return
            self._consecutive_failures += 1
            self._cooldown_duration = min(
                self._max_cooldown,
                self._base_cooldown * (self._factor ** (self._consecutive_failures - 1)),
            )
            self._state = self.OPEN
            self._probe_in_flight = False
            self._open_until = self._clock() + self._cooldown_duration

    def record_success(self) -> None:
        """A request got through: the service is healthy, reset everything."""
        with self._lock:
            self._state = self.CLOSED
            self._consecutive_failures = 0
            self._probe_in_flight = False
            self._open_until = 0.0
            self._cooldown_duration = self._base_cooldown

    def release_probe(self) -> None:
        """Give up a HALF_OPEN probe without recording a failure."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._probe_in_flight = False


class RateLimiterRegistry:
    """Named, lazily-created token buckets -- one per remote service.

    Services are registered by name so a caller asks for ``"acoustid"`` rather
    than carrying a rate around, and so the budget is genuinely process-wide:
    two downloaders in one process share one AcoustID bucket instead of each
    believing it may spend three requests per second.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def bucket(self, name: str, rate_per_second: float, burst: Optional[float] = None) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(name)
            if bucket is None:
                bucket = TokenBucket(rate_per_second, burst)
                self._buckets[name] = bucket
            return bucket

    def get(self, name: str) -> Optional[TokenBucket]:
        with self._lock:
            return self._buckets.get(name)

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


#: Process-wide registry. The documented budgets, applied by name.
limiters = RateLimiterRegistry()


def full_jitter_backoff(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 60.0,
    rng: Optional[random.Random] = None,
) -> float:
    """Exponential backoff with *full* jitter.

    ``attempt`` is 1-based. The window grows as ``base * 2 ** (attempt - 1)``
    and we sleep a uniform sample inside it. Full jitter (rather than
    ``base * 2 ** attempt`` with a small additive wobble) is what actually
    de-synchronises a fleet: identical deterministic sleeps turn N simultaneous
    failures into N simultaneous retries.
    """
    attempt = max(1, int(attempt))
    window = min(float(cap), float(base) * (2 ** (attempt - 1)))
    if window <= 0:
        return 0.0
    source = rng or random
    return source.uniform(0.0, window)


def parse_retry_after(value: Optional[str], *, now: Optional[float] = None) -> Optional[float]:
    """Seconds to wait from a ``Retry-After`` header.

    Handles both documented forms: delta-seconds and an HTTP-date. Returns
    ``None`` when the header is absent or unparseable, so the caller falls back
    to its own backoff rather than trusting a garbage value.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    reference = time.time() if now is None else now
    return max(0.0, when.timestamp() - reference)


_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "too many requests",
    "throttl",
    "quota exceeded",
    "slow down",
)


def is_rate_limit_error(message: Optional[str]) -> bool:
    """True when *message* reads like a service asking us to slow down."""
    if not message:
        return False
    lowered = str(message).lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _utcnow_iso() -> str:  # pragma: no cover - trivial, kept for symmetry
    return datetime.now(timezone.utc).isoformat()
