"""
Content-addressed disk cache with per-entry TTL.

The pipeline asks the same questions over and over. A run that fails five songs
repeats five complete multi-source searches; a ``--retry`` re-run re-asks
MusicBrainz and iTunes for metadata that has not changed; and every track of an
album asks coverartarchive for byte-identical cover art. None of that is
load-bearing work and all of it costs a rate-limited round trip.

Entries are keyed by a caller-supplied string, hashed to a path, and written
atomically so a crash mid-write cannot leave a half-parsed entry behind. Each
entry carries its own expiry, so a search (stale in days) and a fingerprint
verdict (a fact about one specific file) can share a cache without either
expiring at the wrong time.

The cache is advisory throughout: any read failure -- missing file, corrupt
JSON, clock skew -- is a miss, never an error. A cache that can break the
pipeline is worse than no cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["TTLCache", "CacheRegistry", "caches", "cache_root", "cached_json", "MISS"]

MISS: Any = object()

_EXPIRY_BYTES = 8


def _env_disabled() -> bool:
    for name in ("YTDL_NO_CACHE", "YTDL_CACHE_DISABLED"):
        if os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def cache_root() -> Path:
    """Root of the on-disk cache, shared with the Kev model cache."""
    override = os.environ.get("YTDL_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "ytdl"


class TTLCache:
    """A namespaced, TTL-bounded, thread-safe cache.

    Reads go through a small in-process memo in front of the disk, so the
    repeated lookups one run makes never re-parse a file.
    """

    def __init__(
        self,
        namespace: str,
        *,
        root: Optional[Path] = None,
        enabled: Optional[bool] = None,
        max_memory_entries: int = 512,
    ) -> None:
        self.namespace = namespace
        self.root = Path(root) if root is not None else cache_root() / namespace
        self.enabled = (not _env_disabled()) if enabled is None else bool(enabled)
        self._memo: dict[str, tuple[float, Any]] = {}
        self._max_memory_entries = max(0, int(max_memory_entries))
        self._lock = threading.Lock()

    # -- keys -----------------------------------------------------------------

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Build a stable key from arbitrary parts.

        Parts are stringified and joined with a separator that cannot occur in a
        normalised title, so ``("ab", "c")`` and ``("a", "bc")`` cannot collide.
        """
        return "\x1f".join(str(part) for part in parts)

    def _path_for(self, key: str, binary: bool) -> Path:
        digest = hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()
        suffix = "bin" if binary else "json"
        return self.root / digest[:2] / f"{digest}.{suffix}"

    # -- memory layer ---------------------------------------------------------

    def _memo_get(self, key: str) -> Any:
        if self._max_memory_entries <= 0:
            return MISS
        with self._lock:
            entry = self._memo.get(key)
            if entry is None:
                return MISS
            expires, value = entry
            if expires <= time.time():
                self._memo.pop(key, None)
                return MISS
            return value

    def _memo_put(self, key: str, value: Any, expires: float) -> None:
        if self._max_memory_entries <= 0:
            return
        with self._lock:
            if len(self._memo) >= self._max_memory_entries:
                # Cheap bounded eviction: drop the oldest insertion. The memo is
                # an optimisation, so an approximate policy is the right trade.
                oldest = next(iter(self._memo), None)
                if oldest is not None:
                    self._memo.pop(oldest, None)
            self._memo[key] = (expires, value)

    def _memo_drop(self, key: str) -> None:
        with self._lock:
            self._memo.pop(key, None)

    # -- json entries ---------------------------------------------------------

    def get_json(self, key: str) -> Any:
        """Return the cached value, or ``None`` on a miss."""
        if not self.enabled:
            return None
        memoized = self._memo_get(key)
        if memoized is not MISS:
            return memoized
        try:
            raw = self._path_for(key, binary=False).read_text(encoding="utf-8")
            envelope = json.loads(raw)
        except (OSError, ValueError):
            return None
        if not isinstance(envelope, dict) or "v" not in envelope:
            return None
        expires = float(envelope.get("exp") or 0.0)
        if expires <= time.time():
            self.delete(key)
            return None
        self._memo_put(key, envelope["v"], expires)
        return envelope["v"]

    def put_json(self, key: str, value: Any, ttl: float) -> None:
        if not self.enabled or ttl <= 0:
            return
        expires = time.time() + float(ttl)
        self._memo_put(key, value, expires)
        envelope = {"exp": expires, "v": value}
        self._write(self._path_for(key, binary=False), json.dumps(envelope, ensure_ascii=False))

    # -- binary entries (cover art) ------------------------------------------

    def get_bytes(self, key: str) -> Optional[bytes]:
        if not self.enabled:
            return None
        memoized = self._memo_get(key)
        if memoized is not MISS:
            return bytes(memoized) if isinstance(memoized, (bytes, bytearray)) else None
        try:
            blob = self._path_for(key, binary=True).read_bytes()
        except OSError:
            return None
        # Envelope: 8-byte big-endian float expiry, then the raw payload. Keeping
        # the payload raw avoids the 33% a base64 round trip would cost on
        # cover art, which is the only binary thing we cache.
        if len(blob) <= _EXPIRY_BYTES:
            return None
        try:
            expires = struct.unpack(">d", blob[:_EXPIRY_BYTES])[0]
        except struct.error:
            return None
        if expires <= time.time():
            self.delete(key)
            return None
        payload = blob[_EXPIRY_BYTES:]
        self._memo_put(key, payload, expires)
        return payload

    def put_bytes(self, key: str, data: bytes, ttl: float) -> None:
        if not self.enabled or ttl <= 0 or not data:
            return
        expires = time.time() + float(ttl)
        self._memo_put(key, bytes(data), expires)
        blob = struct.pack(">d", expires) + data
        self._write_bytes(self._path_for(key, binary=True), blob)

    # -- maintenance ----------------------------------------------------------

    def delete(self, key: str) -> None:
        self._memo_drop(key)
        for binary in (False, True):
            try:
                self._path_for(key, binary=binary).unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self) -> int:
        """Remove every entry in this namespace. Returns files removed."""
        removed = 0
        with self._lock:
            self._memo.clear()
        if not self.root.is_dir():
            return removed
        for path in self.root.rglob("*"):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    @staticmethod
    def _write(path: Path, payload: str) -> None:
        TTLCache._write_bytes(path, payload.encode("utf-8"))

    @staticmethod
    def _write_bytes(path: Path, payload: bytes) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                os.replace(temporary, path)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except OSError:
            # A cache that cannot be written must not fail the download.
            return


class CacheRegistry:
    """Lazily-created, named caches."""

    def __init__(self) -> None:
        self._caches: dict[str, TTLCache] = {}
        self._lock = threading.Lock()
        self._disabled = False

    def get(self, namespace: str) -> TTLCache:
        with self._lock:
            cache = self._caches.get(namespace)
            if cache is None:
                cache = TTLCache(namespace, enabled=not self._disabled)
                self._caches[namespace] = cache
            return cache

    def disable(self) -> None:
        with self._lock:
            self._disabled = True
            for cache in self._caches.values():
                cache.enabled = False

    def enable(self) -> None:
        with self._lock:
            self._disabled = False
            for cache in self._caches.values():
                cache.enabled = not _env_disabled()


#: Process-wide registry of named caches.
caches = CacheRegistry()


def cached_json(namespace: str, key: str, producer: Callable[[], Any], ttl: float) -> Any:
    """Read-through helper: return the cached value or compute and store it.

    ``None`` from *producer* is stored too when it is a genuine answer, which is
    what keeps a song with no results from being searched again on every retry.
    Use a short ``ttl`` for negatives.
    """
    cache = caches.get(namespace)
    hit = cache.get_json(key)
    if hit is not None:
        return hit
    value = producer()
    cache.put_json(key, value, ttl)
    return value
