"""
Post-download validation: duration check, silence check, and metadata embedding.

These checks run after a file has been successfully downloaded and before it is
considered "done".  Each function returns a status tuple so the caller can
decide whether to keep or delete the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import Config
from .events import DownloaderEvents
from .fingerprint import has_excessive_silence, verify_duration
from .metadata import embed_metadata, fetch_musicbrainz


def check_duration(
    downloaded_file: Path,
    expected_duration: int,
    artist: str,
    song: str,
    events: DownloaderEvents,
) -> tuple[bool, int, Optional[str]]:
    """
    Verify the downloaded file's duration matches the expected value.

    Returns (is_ok, actual_seconds, failure_reason_or_None).
    If the discrepancy is > 40 % the file should be deleted.
    """
    dur_ok, actual_dur = verify_duration(downloaded_file, expected_duration)
    events.on_duration_check(artist, song, expected_duration, actual_dur, dur_ok)

    if not dur_ok and expected_duration > 0:
        discrepancy = abs(actual_dur - expected_duration) / max(expected_duration, 1)
        if discrepancy > 0.40:
            downloaded_file.unlink(missing_ok=True)
            return False, actual_dur, f"Duration discrepancy {discrepancy:.0%}"

    return dur_ok, actual_dur, None


def check_silence(
    downloaded_file: Path,
    artist: str,
    song: str,
    config: Config,
    events: DownloaderEvents,
) -> tuple[float, bool, Optional[str]]:
    """
    Run the silence detection check.

    Returns (silence_ratio, is_excessive, failure_reason_or_None).
    If excessive the file should be deleted.
    """
    is_excessive, silence_ratio = has_excessive_silence(downloaded_file, config)
    events.on_silence_check(artist, song, silence_ratio, is_excessive)

    if is_excessive:
        events.on_silence_rejected(artist, song, silence_ratio)
        downloaded_file.unlink(missing_ok=True)
        return silence_ratio, True, f"Excessive silence ({silence_ratio:.1%})"

    return silence_ratio, False, None


def enrich_musicbrainz(
    artist: str,
    song: str,
    musicbrainz_enabled: bool,
    events: DownloaderEvents,
) -> tuple[Optional[dict], bool]:
    """
    Fetch MusicBrainz metadata if enabled.

    Returns (mb_data_or_None, was_enriched).
    """
    if not musicbrainz_enabled:
        return None, False

    mb_data = fetch_musicbrainz(artist, song)
    enriched = mb_data is not None
    events.on_musicbrainz_result(artist, song, enriched, mb_data or {})
    return mb_data, enriched


def embed_and_verify(
    downloaded_file: Path,
    song: str,
    artist: str,
    url: str,
    thumbnail_url: Optional[str],
    fmt: str,
    mb_data: Optional[dict],
    events: DownloaderEvents,
) -> bool:
    """
    Embed metadata and run the integrity check.

    Returns True on success, False if the file should be deleted.
    """
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
        lambda msg: events.on_warn(msg),
    )

    if not embed_ok:
        events.on_metadata_error(artist, song, downloaded_file.name)
        downloaded_file.unlink(missing_ok=True)
        return False

    return True
