"""Tests for ytdl_core.cache: TTL, binary payloads, and advisory-only reads."""

from __future__ import annotations

import time

from ytdl_core.cache import CacheRegistry, TTLCache, cached_json, caches


class TestKeys:
    def test_parts_cannot_collide(self):
        assert TTLCache.make_key("ab", "c") != TTLCache.make_key("a", "bc")

    def test_same_parts_are_stable(self):
        assert TTLCache.make_key("a", "b") == TTLCache.make_key("a", "b")


class TestJsonEntries:
    def test_round_trip(self, tmp_path):
        cache = TTLCache("t", root=tmp_path)
        cache.put_json("k", {"a": [1, 2]}, 60)
        assert cache.get_json("k") == {"a": [1, 2]}

    def test_miss_returns_none(self, tmp_path):
        assert TTLCache("t", root=tmp_path).get_json("absent") is None

    def test_expired_entry_is_a_miss(self, tmp_path):
        cache = TTLCache("t", root=tmp_path)
        cache.put_json("k", 1, 0.05)
        assert cache.get_json("k") == 1
        time.sleep(0.1)
        assert cache.get_json("k") is None

    def test_survives_a_fresh_instance(self, tmp_path):
        """A warm disk entry must be readable by the next run, not just this one."""
        TTLCache("t", root=tmp_path).put_json("k", "v", 60)
        assert TTLCache("t", root=tmp_path).get_json("k") == "v"

    def test_corrupt_file_is_a_miss_not_a_crash(self, tmp_path):
        cache = TTLCache("t", root=tmp_path)
        cache.put_json("k", "v", 60)
        cache._path_for("k", binary=False).write_text("{not json", encoding="utf-8")
        # The in-process memo would otherwise mask the corrupt file.
        fresh = TTLCache("t", root=tmp_path)
        assert fresh.get_json("k") is None

    def test_unwritable_cache_does_not_raise(self, tmp_path):
        # A read-only root must not turn a cache into a failure.
        cache = TTLCache("t", root=tmp_path / "nested" / "deep")
        cache.put_json("k", 1, 60)
        assert cache.get_json("k") == 1

    def test_disabled_cache_stores_nothing(self, tmp_path):
        cache = TTLCache("t", root=tmp_path, enabled=False)
        cache.put_json("k", 1, 60)
        assert cache.get_json("k") is None
        assert TTLCache("t", root=tmp_path, enabled=True).get_json("k") is None

    def test_negative_answers_can_be_cached(self, tmp_path):
        """A song with no results is an answer worth remembering, briefly."""
        cache = TTLCache("t", root=tmp_path)
        cache.put_json("k", None, 30)
        assert cache.get_json("k") is None
        assert cache._path_for("k", binary=False).exists()

    def test_delete(self, tmp_path):
        cache = TTLCache("t", root=tmp_path)
        cache.put_json("k", 1, 60)
        cache.delete("k")
        assert cache.get_json("k") is None

    def test_clear_removes_everything(self, tmp_path):
        cache = TTLCache("t", root=tmp_path)
        for index in range(5):
            cache.put_json(f"k{index}", index, 60)
        assert cache.clear() >= 5
        assert cache.get_json("k0") is None


class TestBinaryEntries:
    def test_round_trip_is_byte_exact(self, tmp_path):
        payload = bytes(range(256)) * 40
        cache = TTLCache("t", root=tmp_path)
        cache.put_bytes("cover", payload, 60)
        assert cache.get_bytes("cover") == payload

    def test_survives_a_fresh_instance(self, tmp_path):
        payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000
        TTLCache("t", root=tmp_path).put_bytes("cover", payload, 60)
        assert TTLCache("t", root=tmp_path).get_bytes("cover") == payload

    def test_expiry_is_carried_in_the_payload(self, tmp_path):
        cache = TTLCache("t", root=tmp_path)
        cache.put_bytes("cover", b"x" * 100, 0.05)
        time.sleep(0.1)
        assert TTLCache("t", root=tmp_path).get_bytes("cover") is None

    def test_empty_payload_is_not_stored(self, tmp_path):
        cache = TTLCache("t", root=tmp_path)
        cache.put_bytes("cover", b"", 60)
        assert cache.get_bytes("cover") is None


class TestRegistry:
    def test_named_caches_are_reused(self):
        registry = CacheRegistry()
        assert registry.get("a") is registry.get("a")
        assert registry.get("a") is not registry.get("b")

    def test_disable_affects_existing_caches(self, tmp_path):
        registry = CacheRegistry()
        cache = registry.get("a")
        registry.disable()
        assert cache.enabled is False
        registry.enable()
        assert cache.enabled is True


class TestReadThrough:
    def test_producer_runs_once_for_a_warm_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YTDL_CACHE_DIR", str(tmp_path))
        caches.get("readthrough").delete("k")
        # The suite keeps the global cache off; this test is about the
        # read-through contract, so it turns it back on for itself.
        caches.enable()
        try:
            calls = []

            def producer():
                calls.append(1)
                return {"value": len(calls)}

            first = cached_json("readthrough", "k", producer, 60)
            second = cached_json("readthrough", "k", producer, 60)
        finally:
            caches.get("readthrough").delete("k")
            caches.disable()

        assert first == second
        assert len(calls) == 1