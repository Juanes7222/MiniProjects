"""Tests for ytdl_core.post_checks module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ytdl_core.config import Config
from ytdl_core.post_checks import (
    check_duration,
    check_silence,
    embed_and_verify,
    enrich_musicbrainz,
)


# ---------------------------------------------------------------------------
# check_duration
# ---------------------------------------------------------------------------

class TestCheckDuration:
    def test_ok_when_duration_matches(self, real_mp3, spy_events):
        ok, actual, fail = check_duration(real_mp3, 3, "Artist", "Song", spy_events)
        assert ok is True
        assert fail is None
        assert actual > 0
        assert any(c[0] == "on_duration_check" for c in spy_events.calls)

    def test_deletes_file_on_large_discrepancy(self, real_mp3, spy_events):
        ok, actual, fail = check_duration(real_mp3, 100, "Artist", "Song", spy_events)
        assert ok is False
        assert fail is not None
        assert "Duration discrepancy" in fail
        assert not real_mp3.exists()

    def test_returns_none_when_within_tolerance(self, real_mp3, spy_events):
        """3s file vs 3s expected — should pass."""
        ok, actual, fail = check_duration(real_mp3, 3, "Artist", "Song", spy_events)
        assert ok is True
        assert fail is None


# ---------------------------------------------------------------------------
# check_silence
# ---------------------------------------------------------------------------

class TestCheckSilence:
    def test_returns_normal_for_real_audio(self, real_mp3, config, spy_events):
        ratio, excessive, fail = check_silence(real_mp3, "Artist", "Song", config, spy_events)
        assert excessive is False
        assert fail is None
        assert ratio < 0.30
        assert any(c[0] == "on_silence_check" for c in spy_events.calls)

    def test_deletes_file_when_excessive(self, real_mp3, config, spy_events):
        """Force excessive silence by using a very low threshold."""
        config.EXCESSIVE_SILENCE_RATIO = 0.0  # any silence is "excessive"
        ratio, excessive, fail = check_silence(real_mp3, "Artist", "Song", config, spy_events)
        # With EXCESSIVE_SILENCE_RATIO=0, even tiny silence triggers rejection
        if ratio > 0:
            assert excessive is True
            assert fail is not None
            assert not real_mp3.exists()
        else:
            # Pure sine might have zero silence
            assert excessive is False


# ---------------------------------------------------------------------------
# enrich_musicbrainz
# ---------------------------------------------------------------------------

class TestEnrichMusicBrainz:
    def test_returns_none_when_disabled(self, spy_events):
        data, enriched = enrich_musicbrainz("Artist", "Song", False, spy_events)
        assert data is None
        assert enriched is False
        assert not any(c[0] == "on_musicbrainz_result" for c in spy_events.calls)

    def test_calls_fetch_when_enabled(self, spy_events):
        with patch("ytdl_core.post_checks.fetch_musicbrainz") as mock_fetch:
            mock_fetch.return_value = {"album": "Test Album", "year": "2024"}
            data, enriched = enrich_musicbrainz("Artist", "Song", True, spy_events)
            assert enriched is True
            assert data["album"] == "Test Album"
            assert any(c[0] == "on_musicbrainz_result" for c in spy_events.calls)

    def test_handles_fetch_failure(self, spy_events):
        with patch("ytdl_core.post_checks.fetch_musicbrainz", return_value=None):
            data, enriched = enrich_musicbrainz("Artist", "Song", True, spy_events)
            assert enriched is False
            assert data is None


# ---------------------------------------------------------------------------
# embed_and_verify
# ---------------------------------------------------------------------------

class TestEmbedAndVerify:
    def test_returns_true_on_success(self, real_mp3, spy_events):
        with patch("ytdl_core.post_checks.embed_metadata", return_value=True):
            result = embed_and_verify(
                real_mp3, "Song", "Artist", "http://url", None, "mp3", None, spy_events,
            )
            assert result is True

    def test_deletes_file_on_failure(self, real_mp3, spy_events):
        with patch("ytdl_core.post_checks.embed_metadata", return_value=False):
            result = embed_and_verify(
                real_mp3, "Song", "Artist", "http://url", None, "mp3", None, spy_events,
            )
            assert result is False
            assert not real_mp3.exists()
            assert any(c[0] == "on_metadata_error" for c in spy_events.calls)

    def test_passes_mb_data_correctly(self, real_mp3, spy_events):
        mb = {"album": "MB Album", "year": "2023", "genre": "Rock"}
        with patch("ytdl_core.post_checks.embed_metadata", return_value=True) as mock_embed:
            embed_and_verify(real_mp3, "Song", "Artist", "http://url", "http://thumb", "mp3", mb, spy_events)
            args = mock_embed.call_args
            assert args[0][4] == "http://thumb"  # thumbnail_url
            extra = args[0][3]
            assert extra["album"] == "MB Album"
            assert extra["year"] == "2023"
