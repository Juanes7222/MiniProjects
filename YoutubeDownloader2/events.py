"""
Event/callback system for MusicDownloader.

DownloaderEvents is a base class with no-op implementations of every hook.
Override only the methods you need. The CLI (RichEvents) overrides all of
them to produce the Rich terminal output. Third-party consumers can subclass
and do whatever they want (log to file, update a GUI, send to a webhook, etc.)

Usage:
    class MyEvents(DownloaderEvents):
        def on_result(self, result):
            db.insert(result.to_dict())

    dl = MusicDownloader(events=MyEvents())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from result import DownloadResult


class DownloaderEvents:
    """
    Base event class — all methods are no-ops.
    Subclass and override whichever hooks you need.
    """

    def on_session_start(self, total: int) -> None:
        """Fired once before any downloads begin."""

    def on_session_complete(
        self, results: list["DownloadResult"], elapsed: float
    ) -> None:
        """Fired once after all workers finish."""

    def on_interrupted(self, completed: int, total: int, elapsed: float) -> None:
        """Fired on KeyboardInterrupt."""


    def on_artist_start(self, artist: str, song_count: int) -> None:
        """Fired the first time an artist is encountered in the batch."""


    def on_search_start(self, artist: str, song: str, source: str) -> None:
        """Fired before querying each source."""

    def on_no_results(self, artist: str, song: str, source: str) -> None:
        """Fired when a source returns zero results."""

    def on_candidates_scored(
        self,
        artist: str,
        song: str,
        ranked: list[tuple[dict, int, dict]],
    ) -> None:
        """Fired after scoring, with the full ranked candidate list."""

    def on_search_failed(
        self, artist: str, song: str, sources_tried: list[str]
    ) -> None:
        """Fired when no source produced an acceptable result."""


    def on_verification_status(
        self,
        artist: str,
        song: str,
        score: int,
        score_label: str,
        fp_label: str,
    ) -> None:
        """Fired after the pre-download scoring + fingerprint decision."""


    def on_fingerprint_start(self, artist: str, song: str, seconds: int) -> None:
        """Fired before the partial download for fingerprinting."""

    def on_fingerprint_partial_failed(self, artist: str, song: str) -> None:
        """Fired when the partial download itself fails."""

    def on_fingerprint_result(
        self,
        artist: str,
        song: str,
        verified: bool,
        confidence: float,
        matched_title: str,
    ) -> None:
        """Fired after the AcoustID API call returns."""

    def on_fingerprint_low_confidence(
        self, artist: str, song: str, matched_title: str
    ) -> None:
        """Fired when confidence is > 0.4 but < threshold — trying next candidate."""

    def on_fingerprint_no_match(self, artist: str, song: str) -> None:
        """Fired when AcoustID returns no usable match (confidence <= 0.4)."""

    def on_fingerprint_error(self, artist: str, song: str, error: str) -> None:
        """Fired when the AcoustID call raises an exception."""


    def on_skip_existing(
        self, artist: str, song: str, file_path: Path, md5_ok: bool
    ) -> None:
        """Fired when a song is skipped because it already exists on disk."""

    def on_md5_mismatch(self, artist: str, song: str) -> None:
        """Fired when the stored MD5 does not match the file on disk."""

    def on_download_start(self, artist: str, song: str, url: str) -> None:
        """Fired just before yt-dlp starts downloading."""

    def on_download_progress(
        self,
        artist: str,
        song: str,
        percent: float,
        speed_bps: float,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        """
        Fired repeatedly during download.
        percent: 0.0–100.0
        speed_bps: bytes per second (0 if unknown)
        """

    def on_download_retry(
        self,
        artist: str,
        song: str,
        attempt: int,
        max_attempts: int,
        error: str,
        wait_seconds: float,
    ) -> None:
        """Fired before each retry sleep."""

    def on_download_failed(
        self, artist: str, song: str, error: str
    ) -> None:
        """Fired when all retry attempts are exhausted."""

    def on_disk_full(self) -> None:
        """Fired when an OSError with errno 28 (ENOSPC) is caught."""

    def on_duration_check(
        self,
        artist: str,
        song: str,
        expected_seconds: int,
        actual_seconds: int,
        ok: bool,
    ) -> None:
        """Fired after comparing actual file duration to expected."""

    def on_silence_check(
        self,
        artist: str,
        song: str,
        silence_ratio: float,
        excessive: bool,
    ) -> None:
        """Fired after the silence analysis completes."""

    def on_silence_rejected(
        self, artist: str, song: str, silence_ratio: float
    ) -> None:
        """Fired when a file is deleted due to excessive silence."""

    def on_post_check_summary(
        self,
        artist: str,
        song: str,
        dur_ok: bool,
        actual_dur: int,
        silence_ratio: float,
    ) -> None:
        """Fired once with the combined post-check results."""


    def on_musicbrainz_result(
        self,
        artist: str,
        song: str,
        enriched: bool,
        data: dict,
    ) -> None:
        """Fired after the MusicBrainz lookup."""

    def on_metadata_error(self, artist: str, song: str, file_name: str) -> None:
        """Fired when embed_metadata returns False (integrity check failed)."""

    def on_warn(self, message: str) -> None:
        """Generic warning channel (e.g. opus cover art not supported)."""


    def on_result(self, result: "DownloadResult") -> None:
        """
        Fired once per song after the full pipeline completes.
        This is the primary hook for library consumers that just want results.
        """