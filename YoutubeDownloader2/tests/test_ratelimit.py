"""Tests for ytdl_core.ratelimit: budgets, backoff, and circuit breaking."""

from __future__ import annotations

import threading
import time

from ytdl_core.ratelimit import (
    CircuitBreaker,
    RateLimiterRegistry,
    TokenBucket,
    full_jitter_backoff,
    is_rate_limit_error,
    limiters,
    parse_retry_after,
)


class FakeClock:
    """A clock that only moves when the test says so."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestTokenBucket:
    def test_burst_is_free_then_rate_applies(self):
        clock = FakeClock()
        bucket = TokenBucket(3.0, burst=3, clock=clock.time, sleep=clock.sleep)

        for _ in range(3):
            assert bucket.acquire() == 0.0

        assert bucket.acquire() == pytest_approx(1 / 3)
        assert sum(clock.slept) == pytest_approx(1 / 3)

    def test_zero_rate_disables_limiting(self):
        clock = FakeClock()
        bucket = TokenBucket(0.0, clock=clock.time, sleep=clock.sleep)
        for _ in range(100):
            bucket.acquire()
        assert clock.slept == []

    def test_sleep_happens_outside_the_lock(self):
        """The bucket's lock must be free while a caller is sleeping.

        Sleeping under the lock turns a rate limit into a mutex: every other
        thread queues behind the sleeper and then finds the bucket still empty.
        The check below is exact -- the injected sleep tries to take the same
        lock non-blockingly, which can only succeed if the bucket released it.
        """
        clock = FakeClock()
        bucket = TokenBucket(1.0, burst=1, clock=clock.time)
        bucket.acquire()  # drain the burst so the next call has to wait

        lock_was_free: list[bool] = []

        def sleep(seconds: float) -> None:
            taken = bucket._lock.acquire(blocking=False)
            lock_was_free.append(taken)
            if taken:
                bucket._lock.release()
            clock.sleep(seconds)

        bucket._sleep = sleep
        bucket.acquire()

        assert lock_was_free == [True]

    def test_penalty_blocks_then_reopens(self):
        clock = FakeClock()
        bucket = TokenBucket(10.0, burst=5, clock=clock.time, sleep=clock.sleep)
        for _ in range(5):
            bucket.acquire()

        bucket.penalty(2.0)
        # Nothing may be served during the penalty window.
        assert bucket.try_acquire() is False
        clock.now += 1.0
        assert bucket.try_acquire() is False

        clock.now += 1.5
        assert bucket.try_acquire() is True

    def test_penalty_drains_tokens_already_held(self):
        clock = FakeClock()
        bucket = TokenBucket(10.0, burst=5, clock=clock.time, sleep=clock.sleep)
        bucket.acquire()
        bucket.penalty(1.0)
        assert bucket.try_acquire() is False


    def test_concurrent_callers_share_one_budget(self):
        """N threads must not collectively exceed the published rate."""
        bucket = TokenBucket(100.0, burst=1)
        start = time.monotonic()
        threads = [
            threading.Thread(target=lambda: [bucket.acquire() for _ in range(2)])
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.monotonic() - start
        # 10 acquisitions at 100/s with a burst of 1 needs ~0.09s of pacing.
        assert 0.05 <= elapsed < 5.0

    def test_penalty_drains_the_budget(self):
        clock = FakeClock()
        bucket = TokenBucket(3.0, burst=3, clock=clock.time, sleep=clock.sleep)
        bucket.acquire()
        bucket.penalty(5.0)
        # The next acquire has to wait out the penalty plus normal refilling.
        assert bucket.acquire() > 0.0

    def test_try_acquire_never_blocks(self):
        clock = FakeClock()
        bucket = TokenBucket(1.0, burst=1, clock=clock.time, sleep=clock.sleep)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False
        assert clock.slept == []

    def test_reset_refills(self):
        clock = FakeClock()
        bucket = TokenBucket(1.0, burst=1, clock=clock.time, sleep=clock.sleep)
        bucket.acquire()
        bucket.reset()
        assert bucket.acquire() == 0.0


class TestCircuitBreaker:
    def test_starts_closed(self):
        assert CircuitBreaker().is_open is False

    def test_trip_opens(self):
        breaker = CircuitBreaker(cooldown_seconds=60)
        breaker.trip()
        assert breaker.is_open is True

    def test_cooldown_expiry_allows_a_probe(self):
        breaker = CircuitBreaker(cooldown_seconds=0.05)
        breaker.trip()
        assert breaker.is_open is True
        time.sleep(0.1)
        assert breaker.is_open is False
        assert breaker.state == CircuitBreaker.HALF_OPEN

    def test_trip_is_idempotent(self):
        """A burst of parallel errors must not compound into a longer cooldown."""
        breaker = CircuitBreaker(cooldown_seconds=60)
        breaker.trip()
        first = breaker.cooldown_remaining()
        for _ in range(5):
            breaker.trip()
        assert breaker.is_open is True
        assert breaker.cooldown_remaining() <= first

    def test_repeated_failures_widen_the_cooldown(self):
        """Each served cooldown must be longer than the last.

        A fixed cooldown retries a struggling service just as hard, just as
        often; widening it is what lets the breaker actually stop hammering.
        """
        clock = FakeClock()
        breaker = CircuitBreaker(
            cooldown_seconds=10.0, max_cooldown_seconds=100.0, factor=2.0, clock=clock.time
        )
        cooldowns = []
        for _ in range(4):
            breaker.trip()
            cooldowns.append(breaker.cooldown_remaining())
            clock.now += breaker.cooldown_remaining() + 1.0
            assert breaker.allow() is True  # the half-open probe
            breaker.trip()  # probe failed too
        assert cooldowns == sorted(cooldowns)
        assert cooldowns[-1] > cooldowns[0]

    def test_cooldown_is_capped(self):
        clock = FakeClock()
        breaker = CircuitBreaker(
            cooldown_seconds=1.0, max_cooldown_seconds=4.0, factor=10.0, clock=clock.time
        )
        for _ in range(10):
            breaker.trip()
            clock.now += breaker.cooldown_remaining() + 1.0
            breaker.allow()
        assert breaker.cooldown_remaining() <= 4.0

    def test_half_open_admits_exactly_one_probe(self):
        breaker = CircuitBreaker(cooldown_seconds=0.01)
        breaker.trip()
        time.sleep(0.05)
        assert breaker.allow() is True
        assert breaker.allow() is False

    def test_success_resets_everything(self):
        breaker = CircuitBreaker(cooldown_seconds=0.01)
        breaker.trip()
        breaker.allow()
        breaker.record_success()
        assert breaker.state == CircuitBreaker.CLOSED
        assert breaker.cooldown_remaining() == 0.0
        assert breaker.allow() is True

    def test_release_probe_without_failure(self):
        breaker = CircuitBreaker(cooldown_seconds=0.01)
        breaker.trip()
        time.sleep(0.05)
        assert breaker.allow() is True
        breaker.release_probe()
        assert breaker.allow() is True


class TestBackoff:
    def test_window_grows_and_is_capped(self):
        import random

        rng = random.Random(1234)
        first = full_jitter_backoff(1, base=1.0, cap=30.0, rng=rng)
        later = full_jitter_backoff(6, base=1.0, cap=30.0, rng=rng)
        assert 0 <= first <= 1.0
        assert 0 <= later <= 30.0

    def test_jitter_actually_varies(self):
        import random

        rng = random.Random(7)
        samples = {full_jitter_backoff(4, base=1.0, cap=8.0, rng=rng) for _ in range(20)}
        assert len(samples) > 1

    def test_is_never_negative(self):
        import random

        rng = random.Random(99)
        assert all(full_jitter_backoff(0, base=0.0, rng=rng) == 0.0 for _ in range(5))


class TestRetryAfter:
    def test_delta_seconds(self):
        assert parse_retry_after("5") == 5.0

    def test_http_date(self):
        assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=1445412480.0) == 0.0
        assert parse_retry_after("Wed, 21 Oct 2015 07:28:30 GMT", now=1445412480.0) == 30.0

    def test_unparseable_is_none(self):
        assert parse_retry_after("soon") is None
        assert parse_retry_after("") is None
        assert parse_retry_after(None) is None

    def test_never_negative(self):
        assert parse_retry_after("-5") == 0.0


class TestRateLimitDetection:
    def test_recognises_throttling(self):
        for message in ("HTTP 429", "rate limit exceeded", "Too Many Requests", "throttled"):
            assert is_rate_limit_error(message) is True

    def test_ignores_ordinary_failures(self):
        for message in ("Video unavailable", "HTTP Error 404", "private video", ""):
            assert is_rate_limit_error(message) is False


class TestRegistry:
    def test_same_name_shares_one_bucket(self):
        registry = RateLimiterRegistry()
        first = registry.bucket("svc", 2.0)
        second = registry.bucket("svc", 2.0)
        assert first is second

    def test_different_names_are_independent(self):
        registry = RateLimiterRegistry()
        assert registry.bucket("a", 1.0) is not registry.bucket("b", 1.0)

    def test_clear_forgets_everything(self):
        registry = RateLimiterRegistry()
        registry.bucket("svc", 1.0)
        registry.clear()
        assert registry.get("svc") is None

    def test_module_registry_is_usable(self):
        bucket = limiters.bucket("unit-test-service", 5.0)
        assert bucket.rate_per_second == 5.0


def pytest_approx(value: float, tolerance: float = 1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tolerance

        def __repr__(self):  # pragma: no cover - failure output only
            return f"approx({value})"

    return _Approx()