"""
Learned channel trust.

The strongest available signal in this catalogue is *who published the video*.
Label channels (``Discos Fuentes Edimusica``), auto-generated ``<Artist> - Topic``
channels and artist-owned channels repeatedly deliver the correct recording,
while personal accounts deliver covers, mashups and reuploads whose titles match
almost perfectly -- which is exactly why title similarity alone picks the wrong
file.

Rather than hardcoding a channel list, this module learns one from the user's
own download state: every previously downloaded entry contributes weight to the
channel it came from (AcoustID-verified downloads count triple), and channels
seen for the *same artist* score higher than channels seen once in passing.

Nothing here reaches the network; it is a pure read of ``.download_state.json``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from .config import Config
from .utils import normalize_title

__all__ = ["ChannelTrust", "channel_url_for", "backfill_channels"]


def _norm(value: Any) -> str:
    return normalize_title(str(value or ""))


def channel_url_for(entry: dict) -> str:
    """Best-effort channel URL for a flat yt-dlp entry."""
    for key in ("channel_url", "uploader_url", "channel", "uploader"):
        value = entry.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    return ""


def _is_youtube(url: str) -> bool:
    """Channel-scoped search only works on YouTube channel/handle URLs."""
    return "youtube.com" in (url or "").lower()


def _probe_channel(url: str, opts: dict) -> tuple[str, str]:
    """Resolve (channel_name, channel_url) for a video URL. Never raises."""
    import yt_dlp

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": True,
    }
    if opts.get("cookies_browser"):
        ydl_opts["cookiesfrombrowser"] = (opts["cookies_browser"],)
    if opts.get("cookies_file"):
        ydl_opts["cookiefile"] = str(opts["cookies_file"])
    if opts.get("proxy"):
        ydl_opts["proxy"] = opts["proxy"]
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception:
        return "", ""
    return channel_url_for(info), str(info.get("channel") or info.get("uploader") or "")


def backfill_channels(state: dict, opts: Optional[dict] = None, workers: int = 4) -> int:
    """
    Populate ``channel``/``channel_url`` on download-state entries that lack them.

    The trust model is only as good as the provenance recorded in the state
    file, and a state file written by an older version has none. Without this,
    the first run after upgrading scores every candidate as coming from an
    unknown channel and the feature only kicks in from the second run onward.

    Returns the number of entries enriched. Callers are responsible for saving
    the state afterwards.
    """
    downloads = (state or {}).get("downloads")
    if not isinstance(downloads, dict):
        return 0

    pending = [
        (key, entry)
        for key, entry in downloads.items()
        if isinstance(entry, dict)
        and entry.get("status") == "downloaded"
        and entry.get("url")
        and not entry.get("channel")
    ]
    if not pending:
        return 0

    opts = opts or {}
    lock = threading.Lock()
    enriched = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_probe_channel, entry["url"], opts): (key, entry)
            for key, entry in pending
        }
        for future in as_completed(futures):
            key, entry = futures[future]
            try:
                channel_url, name = future.result()
            except Exception:
                continue
            if not name:
                continue
            with lock:
                entry["channel"] = name
                if channel_url:
                    entry["channel_url"] = channel_url
                enriched += 1

    return enriched


class ChannelTrust:
    """Read-only trust model built from previously downloaded entries."""

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or Config()
        self.max_bonus = cfg.TRUST_MAX_BONUS
        self.strong_weight = cfg.TRUST_STRONG_WEIGHT
        self.medium_weight = cfg.TRUST_MEDIUM_WEIGHT
        self.verified_multiplier = cfg.TRUST_VERIFIED_MULTIPLIER
        self.base_bonus = cfg.TRUSTED_CHANNEL_BONUS
        self.seen_bonus = cfg.TRUSTED_CHANNEL_BONUS_SEEN
        self.artist_bonus = cfg.TRUSTED_ARTIST_CHANNEL_BONUS
        self._channels: dict[str, dict[str, Any]] = {}

    # -- construction ------------------------------------------------------
    def add(
        self,
        channel: str,
        artist: Optional[str] = None,
        channel_url: str = "",
        verified: bool = False,
    ) -> None:
        """Record one observation of a channel."""
        key = _norm(channel)
        if not key:
            return
        record = self._channels.setdefault(
            key, {"name": channel, "weight": 0, "artists": {}, "url": ""}
        )
        record["weight"] += self.verified_multiplier if verified else 1
        if artist:
            record["artists"][_norm(artist)] = record["artists"].get(_norm(artist), 0) + 1
        if channel_url and not record["url"]:
            record["url"] = channel_url

    @classmethod
    def from_state(cls, state: Optional[dict], config: Optional[Config] = None) -> "ChannelTrust":
        """Build a trust model from a loaded ``.download_state.json`` dict."""
        trust = cls(config)
        downloads = (state or {}).get("downloads")
        if not isinstance(downloads, dict):
            return trust
        for key, entry in downloads.items():
            if not isinstance(entry, dict) or entry.get("status") != "downloaded":
                continue
            channel = entry.get("channel")
            if not channel:
                continue
            artist = str(key).split("::", 1)[0]
            trust.add(
                channel,
                artist=artist,
                channel_url=entry.get("channel_url") or "",
                verified=bool(entry.get("fingerprint_verified")),
            )
        return trust

    # -- queries -----------------------------------------------------------
    def weight(self, channel: str) -> int:
        record = self._channels.get(_norm(channel))
        return int(record["weight"]) if record else 0

    def is_known(self, channel: str) -> bool:
        return _norm(channel) in self._channels

    def bonus_for(self, channel: str, artist: Optional[str] = None) -> int:
        """Score bonus for a candidate published on *channel*.

        A single observation is not a pattern. The state file records whatever
        was downloaded, including the occasional wrong pick, so requiring two
        independent downloads before granting any trust keeps one mistaken
        download (a 7-minute mashup saved as "Payaso", a DRM-protected
        SoundCloud rip) from promoting its uploader to a trusted source.

        Returns 0 for unknown or single-observation channels so the caller can
        omit the signal entirely rather than adding a no-op breakdown entry.
        """
        record = self._channels.get(_norm(channel))
        if not record:
            return 0
        weight = int(record["weight"])
        if weight < self.medium_weight:
            return 0
        if weight >= self.strong_weight:
            bonus = self.base_bonus
        else:
            bonus = self.base_bonus * 2 // 3
        if artist and record["artists"].get(_norm(artist)):
            bonus += self.artist_bonus
        return min(bonus, self.max_bonus)

    def channels_for_artist(self, artist: str, limit: int = 2) -> list[dict[str, Any]]:
        """Known channels for *artist*, best first, as ``{name, url, weight}``.

        Only channels whose URL we actually recorded are returned -- a channel
        name alone is not enough to build a browse URL.
        """
        wanted = _norm(artist)
        matches = [
            {
                "name": record["name"],
                "url": record["url"],
                "weight": int(record["weight"]),
                "artist_hits": int(record["artists"].get(wanted, 0)),
            }
            for record in self._channels.values()
            if _is_youtube(record["url"]) and record["artists"].get(wanted)
        ]
        matches.sort(key=lambda item: (item["artist_hits"], item["weight"]), reverse=True)
        return matches[: max(0, limit)]

    def top_channels(self, limit: int = 15) -> list[tuple[str, int, int]]:
        """Diagnostic view: ``(channel, weight, artist_count)``, heaviest first."""
        rows = [
            (record["name"], int(record["weight"]), len(record["artists"]))
            for record in self._channels.values()
        ]
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:limit]

    def __len__(self) -> int:
        return len(self._channels)
