"""Tests for ytdl_core.fingerprint module."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ytdl_core.config import Config
from ytdl_core.fingerprint import (
    AcoustIDCircuitBreaker,
    _artist_stem,
    has_excessive_silence,
    verify_duration,
    verify_fingerprint,
)


# ---------------------------------------------------------------------------
# AcoustIDCircuitBreaker
# ---------------------------------------------------------------------------

class TestAcoustIDCircuitBreaker:
    def test_starts_closed(self):
        cb = AcoustIDCircuitBreaker(cooldown_seconds=60)
        assert cb.is_open is False

    def test_trip_opens_breaker(self):
        cb = AcoustIDCircuitBreaker(cooldown_seconds=60)
        cb.trip()
        assert cb.is_open is True

    def test_cooldown_expires(self):
        cb = AcoustIDCircuitBreaker(cooldown_seconds=0.1)
        cb.trip()
        assert cb.is_open is True
        time.sleep(0.15)
        assert cb.is_open is False

    def test_trip_is_idempotent(self):
        cb = AcoustIDCircuitBreaker(cooldown_seconds=60)
        cb.trip()
        cb.trip()  # should not extend cooldown
        assert cb.is_open is True


# ---------------------------------------------------------------------------
# verify_duration
# ---------------------------------------------------------------------------

class TestVerifyDuration:
    def test_returns_false_for_nonexistent_file(self, tmp_path):
        ok, actual = verify_duration(tmp_path / "nonexistent.mp3", 100)
        assert ok is False
        assert actual == 0

    def test_returns_true_when_expected_is_zero(self, real_mp3):
        """When expected=0, we just want to know the file is readable."""
        ok, actual = verify_duration(real_mp3, 0)
        assert ok is True
        assert actual > 0

    def test_matches_real_audio_duration(self, real_mp3):
        """The real_mp3 fixture generates a 3-second file."""
        ok, actual = verify_duration(real_mp3, 3)
        assert ok is True
        assert 2 <= actual <= 4

    def test_fails_on_large_discrepancy(self, real_mp3):
        """A 3-second file vs expected 100 seconds should fail."""
        ok, actual = verify_duration(real_mp3, 100)
        assert ok is False
        assert actual > 0

    def test_custom_tolerance(self, real_mp3):
        """Tight tolerance (0.01) should fail for a 3s vs 4s comparison."""
        ok, actual = verify_duration(real_mp3, 4, tolerance=0.01)
        assert ok is False


# ---------------------------------------------------------------------------
# verify_fingerprint (unit-level, mocked acoustid)
# ---------------------------------------------------------------------------

class TestVerifyFingerprint:
    def test_returns_no_key_when_no_key(self, tmp_path, config):
        from ytdl_core.fingerprint import AcoustIDCircuitBreaker
        cb = AcoustIDCircuitBreaker()
        ok, conf, title = verify_fingerprint(
            tmp_path / "x.mp3", "Artist", "Song", "", config, cb
        )
        assert ok is False
        assert conf == 0.0
        assert title == "no_key"

    def test_returns_circuit_breaker_open(self, tmp_path, config):
        cb = AcoustIDCircuitBreaker()
        cb.trip()
        ok, conf, title = verify_fingerprint(
            tmp_path / "x.mp3", "Artist", "Song", "KEY", config, cb
        )
        assert ok is False
        assert conf == 0.0
        assert title == "circuit_breaker_open"

    def test_returns_error_on_exception(self, tmp_path, config):
        cb = AcoustIDCircuitBreaker()
        with patch("ytdl_core.fingerprint.acoustid.match", side_effect=RuntimeError("boom")):
            ok, conf, title = verify_fingerprint(
                tmp_path / "x.mp3", "Artist", "Song", "KEY", config, cb,
                on_fingerprint_error=lambda a, s, e: None,
            )
        assert ok is False
        assert conf == 0.0
        assert title == "fingerprint_error"

    def test_rate_limit_trips_breaker(self, tmp_path, config):
        cb = AcoustIDCircuitBreaker(cooldown_seconds=60)
        with patch("ytdl_core.fingerprint.acoustid.match", side_effect=Exception("error 429 rate limit")):
            with patch("ytdl_core.fingerprint.time.sleep"):
                ok, conf, title = verify_fingerprint(
                    tmp_path / "x.mp3", "Artist", "Song", "KEY", config, cb,
                    on_warn=lambda m: None,
                )
        assert ok is False
        assert title == "rate_limit_exceeded"
        assert cb.is_open is True


# ---------------------------------------------------------------------------
# _artist_stem
# ---------------------------------------------------------------------------

class TestArtistStem:
    def test_plain_artist_unchanged(self):
        assert _artist_stem("Barak") == "Barak"

    def test_strips_feat_clause(self):
        assert _artist_stem("Barak feat. Marcos Yaroide") == "Barak"

    def test_strips_ft_abbreviation(self):
        assert _artist_stem("Wiso Aponte ft. Redimi2") == "Wiso Aponte"

    def test_strips_featuring_and_con(self):
        assert _artist_stem("Generación 12 featuring Coalo Zamorano") == "Generación 12"
        assert _artist_stem("Barak con Marcos Yaroide") == "Barak"

    def test_empty_input(self):
        assert _artist_stem("") == ""
        assert _artist_stem(None) == ""


# ---------------------------------------------------------------------------
# verify_fingerprint with featured artists (mocked acoustid)
# ---------------------------------------------------------------------------

class TestVerifyFingerprintFeaturing:
    def test_matches_when_artist_has_feat_credit(self, tmp_path, config):
        """A recording credited 'Barak feat. Marcos Yaroide' must still match
        the query artist 'Barak' (feature credits are dropped)."""
        cb = AcoustIDCircuitBreaker()
        fake = tmp_path / "fake.mp3"
        fake.write_bytes(b"x")
        with patch("ytdl_core.fingerprint.acoustid.match", return_value=[
            (0.95, "rec1", "Sumérgeme en tu gloria", "Barak feat. Marcos Yaroide"),
        ]):
            ok, conf, title = verify_fingerprint(
                fake, "Barak", "Sumérgeme en Tu Gloria", "KEY", config, cb
            )
        assert ok is True
        assert title == "Sumérgeme en tu gloria"

    def test_rejects_when_artist_truly_differs(self, tmp_path, config):
        """A Marco Barrientos recording must NOT match the query artist Barak."""
        cb = AcoustIDCircuitBreaker()
        fake = tmp_path / "fake.mp3"
        fake.write_bytes(b"x")
        with patch("ytdl_core.fingerprint.acoustid.match", return_value=[
            (0.96, "rec1", "Levántate y resplandece", "Marco Barrientos"),
        ]):
            ok, conf, title = verify_fingerprint(
                fake, "Barak", "Levántate y Resplandece", "KEY", config, cb
            )
        assert ok is False
        assert "Marco Barrientos" in title


# ---------------------------------------------------------------------------
# has_excessive_silence (integration with ffmpeg)
# ---------------------------------------------------------------------------

class TestHasExcessiveSilence:
    def test_returns_false_for_missing_file(self, tmp_path, config):
        is_exc, ratio = has_excessive_silence(tmp_path / "missing.mp3", config)
        assert is_exc is False
        assert ratio == 0.0

    def test_returns_false_for_real_audio(self, real_mp3, config):
        """A 3-second sine wave should have very little silence."""
        is_exc, ratio = has_excessive_silence(real_mp3, config)
        assert is_exc is False
        assert ratio < 0.30
