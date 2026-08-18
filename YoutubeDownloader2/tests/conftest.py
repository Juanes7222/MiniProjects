"""Shared test fixtures for ytdl_core tests."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from ytdl_core.config import Config
from ytdl_core.events import DownloaderEvents
from ytdl_core.result import DownloadResult


class SpyEvents(DownloaderEvents):
    """DownloaderEvents that records every callback invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def _record(self, name: str, *args) -> None:
        self.calls.append((name, args))

    def on_session_start(self, total, is_verify=False):
        self._record("on_session_start", total, is_verify)

    def on_session_complete(self, results, elapsed):
        self._record("on_session_complete", results, elapsed)

    def on_interrupted(self, completed, total, elapsed):
        self._record("on_interrupted", completed, total, elapsed)

    def on_artist_start(self, artist, song_count):
        self._record("on_artist_start", artist, song_count)

    def on_search_start(self, artist, song, source):
        self._record("on_search_start", artist, song, source)

    def on_no_results(self, artist, song, source):
        self._record("on_no_results", artist, song, source)

    def on_candidates_scored(self, artist, song, ranked):
        self._record("on_candidates_scored", artist, song, ranked)

    def on_search_failed(self, artist, song, sources_tried):
        self._record("on_search_failed", artist, song, sources_tried)

    def on_verification_status(self, artist, song, score, score_label, fp_label):
        self._record("on_verification_status", artist, song, score, score_label, fp_label)

    def on_fingerprint_start(self, artist, song, seconds):
        self._record("on_fingerprint_start", artist, song, seconds)

    def on_fingerprint_partial_failed(self, artist, song):
        self._record("on_fingerprint_partial_failed", artist, song)

    def on_fingerprint_result(self, artist, song, verified, confidence, matched_title):
        self._record("on_fingerprint_result", artist, song, verified, confidence, matched_title)

    def on_fingerprint_low_confidence(self, artist, song, matched_title):
        self._record("on_fingerprint_low_confidence", artist, song, matched_title)

    def on_fingerprint_no_match(self, artist, song):
        self._record("on_fingerprint_no_match", artist, song)

    def on_fingerprint_error(self, artist, song, error):
        self._record("on_fingerprint_error", artist, song, error)

    def on_skip_existing(self, artist, song, file_path, md5_ok):
        self._record("on_skip_existing", artist, song, file_path, md5_ok)

    def on_md5_mismatch(self, artist, song):
        self._record("on_md5_mismatch", artist, song)

    def on_download_start(self, artist, song, url):
        self._record("on_download_start", artist, song, url)

    def on_download_progress(self, artist, song, percent, speed_bps, downloaded_bytes, total_bytes):
        self._record("on_download_progress", artist, song, percent, speed_bps, downloaded_bytes, total_bytes)

    def on_download_retry(self, artist, song, attempt, max_attempts, error, wait_seconds):
        self._record("on_download_retry", artist, song, attempt, max_attempts, error, wait_seconds)

    def on_download_failed(self, artist, song, error):
        self._record("on_download_failed", artist, song, error)

    def on_disk_full(self):
        self._record("on_disk_full")

    def on_duration_check(self, artist, song, expected_seconds, actual_seconds, ok):
        self._record("on_duration_check", artist, song, expected_seconds, actual_seconds, ok)

    def on_silence_check(self, artist, song, silence_ratio, excessive):
        self._record("on_silence_check", artist, song, silence_ratio, excessive)

    def on_silence_rejected(self, artist, song, silence_ratio):
        self._record("on_silence_rejected", artist, song, silence_ratio)

    def on_post_check_summary(self, artist, song, dur_ok, actual_dur, silence_ratio):
        self._record("on_post_check_summary", artist, song, dur_ok, actual_dur, silence_ratio)

    def on_musicbrainz_result(self, artist, song, enriched, data):
        self._record("on_musicbrainz_result", artist, song, enriched, data)

    def on_metadata_error(self, artist, song, file_name):
        self._record("on_metadata_error", artist, song, file_name)

    def on_info(self, message):
        self._record("on_info", message)

    def on_warn(self, message):
        self._record("on_warn", message)

    def on_result(self, result):
        self._record("on_result", result)


@pytest.fixture
def spy_events():
    """A SpyEvents instance that records all callback invocations."""
    return SpyEvents()


@pytest.fixture
def config():
    """Default Config instance for tests."""
    return Config()


@pytest.fixture
def tmp_audio(tmp_path):
    """Create a minimal valid MP3-like file for testing.

    Uses mutagen to create a tiny but valid audio file.
    """
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1

    mp3_path = tmp_path / "test.mp3"
    # Create a minimal MP3 file (silent, 1 second)
    # We'll use a raw approach — create a file with ID3 tags
    tags = ID3()
    tags.add(TIT2(encoding=3, text="Test Song"))
    tags.add(TPE1(encoding=3, text="Test Artist"))
    tags.save(str(mp3_path))

    return mp3_path


@pytest.fixture
def real_mp3(tmp_path):
    """Create a real playable MP3 file using pydub for testing duration checks."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine

        # Generate a 3-second sine wave
        tone = Sine(440).to_audio_segment(duration=3000)
        mp3_path = tmp_path / "real_test.mp3"
        tone.export(str(mp3_path), format="mp3", bitrate="128k")
        return mp3_path
    except Exception:
        pytest.skip("pydub not available for generating test audio")
