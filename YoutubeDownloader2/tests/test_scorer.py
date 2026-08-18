"""Tests for ytdl_core.scorer module — score_youtube_result and rank_results."""

from __future__ import annotations

import pytest

from ytdl_core.config import Config
from ytdl_core.scorer import rank_results, score_youtube_result


@pytest.fixture
def config():
    return Config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yt(title, channel="Official Artist", duration=200, **kwargs):
    """Build a minimal yt-dlp-style candidate dict."""
    d = {"title": title, "channel": channel, "uploader": channel, "duration": duration}
    d.update(kwargs)
    return d


def _api(title, artists=None, channel="Artist Topic", duration=200, **kwargs):
    """Build a YouTube Music API candidate dict."""
    d = {
        "title": title,
        "channel": channel,
        "uploader": channel,
        "duration": duration,
        "_source": "ytmusic_api",
        "artists": artists or ["Artist"],
    }
    d.update(kwargs)
    return d


# ===================================================================
# Hard rejection gates (standard yt-dlp path)
# ===================================================================

class TestHardRejection:
    def test_rejects_cover(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Song (Cover)"), "Artist", "Song", None, config,
        )
        assert score == -9999
        assert "hard_reject_cover" in bd

    def test_rejects_karaoke(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Song Karaoke Version"), "Artist", "Song", None, config,
        )
        assert score == -9999

    def test_rejects_remix(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Song (Official Remix)"), "Artist", "Song", None, config,
        )
        assert score == -9999

    def test_rejects_live(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Song Live at Concert"), "Artist", "Song", None, config,
        )
        assert score == -9999

    def test_rejects_nightcore(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Song (Nightcore)"), "Artist", "Song", None, config,
        )
        assert score == -9999

    def test_rejects_sped_up(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Song (Sped Up)"), "Artist", "Song", None, config,
        )
        assert score == -9999

    def test_rejects_full_album(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Full Album 2024"), "Artist", "Song", None, config,
        )
        assert score == -9999
        # The key uses the normalized phrase (space preserved)
        assert any("full album" in k for k in bd)

    def test_rejects_low_song_match(self, config):
        score, bd = score_youtube_result(
            _yt("Completely Different Title Here"), "Artist", "Song", None, config,
        )
        assert score == -9999
        assert "hard_reject_song_absent" in bd

    def test_rejects_low_artist_match(self, config):
        score, bd = score_youtube_result(
            _yt("Some Random Channel - Song", channel="Unknown Channel"), "Artist", "Song", None, config,
        )
        assert score == -9999
        assert "hard_reject_artist_absent" in bd

    def test_does_not_reject_when_forbidden_in_query(self, config):
        """If the query itself contains a forbidden term, don't reject."""
        score, bd = score_youtube_result(
            _yt("Artist - Cover Art Design"), "Artist", "Cover Art", None, config,
        )
        # Should NOT be hard-rejected since "cover" is in the query
        assert score != -9999 or "hard_reject_cover" not in bd

    def test_rejects_tribute(self, config):
        score, bd = score_youtube_result(
            _yt("Tribute to Artist - Song"), "Artist", "Song", None, config,
        )
        assert score == -9999

    def test_rejects_mashup(self, config):
        score, bd = score_youtube_result(
            _yt("Artist - Song Mashup"), "Artist", "Song", None, config,
        )
        assert score == -9999


# ===================================================================
# YouTube Music API fast-path
# ===================================================================

class TestYTMusicAPI:
    def test_exact_match_scores_high(self, config):
        r = _api("Dios Es Amor", artists=["Wiso Aponte"])
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", None, config)
        assert score > 50
        assert "official_ytmusic_api" in bd
        assert "catalog_match" in bd
        assert "artist_match" in bd

    def test_song_mismatch_rejects(self, config):
        """Even with a good artist match, a bad song title should fail the gate."""
        r = _api("Dios Es Asi", artists=["Wiso Aponte"])
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", None, config)
        # song_match < 80 gate should prevent the fast-path from scoring
        assert "official_ytmusic_api" not in bd

    def test_artist_mismatch_rejects(self, config):
        r = _api("Dios Es Amor", artists=["Different Artist"])
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", None, config)
        assert "official_ytmusic_api" not in bd

    def test_duration_perfect(self, config):
        r = _api("Dios Es Amor", artists=["Wiso Aponte"], duration=200)
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", 198, config)
        assert "duration_perfect" in bd
        assert bd["duration_perfect"] == config.DURATION_MATCH_BONUS

    def test_duration_close(self, config):
        r = _api("Dios Es Amor", artists=["Wiso Aponte"], duration=200)
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", 190, config)
        assert "duration_close" in bd
        assert bd["duration_close"] == 10

    def test_duration_mismatch_penalizes(self, config):
        r = _api("Dios Es Amor", artists=["Wiso Aponte"], duration=200)
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", 150, config)
        assert "duration_mismatch" in bd
        assert bd["duration_mismatch"] == -35

    def test_cross_source_consensus(self, config):
        r = _api("Dios Es Amor", artists=["Wiso Aponte"], _source_count=3)
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", None, config)
        assert "cross_source_consensus" in bd
        assert bd["cross_source_consensus"] == 20  # min(20, (3-1)*10)

    def test_single_source_no_consensus(self, config):
        r = _api("Dios Es Amor", artists=["Wiso Aponte"])
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", None, config)
        assert "cross_source_consensus" not in bd

    def test_featuring_in_title_handled(self, config):
        r = _api("Dios Es Amor (feat. Wiso Aponte)", artists=["Wiso Aponte"])
        score, bd = score_youtube_result(r, "Wiso Aponte", "Dios Es Amor", None, config)
        # Should still score well despite featuring credit
        assert score > 30


# ===================================================================
# Standard yt-dlp scoring signals
# ===================================================================

class TestStandardScoring:
    def test_good_match_scores_positive(self, config):
        r = _yt("Artist - Song (Official Audio)", channel="Artist Topic")
        score, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert score > 0
        assert "base_match" in bd

    def test_official_audio_bonus(self, config):
        r1 = _yt("Artist - Song (Official Audio)", channel="Artist Topic")
        s1, _ = score_youtube_result(r1, "Artist", "Song", None, config)

        r2 = _yt("Artist - Song", channel="Artist Topic")
        s2, _ = score_youtube_result(r2, "Artist", "Song", None, config)

        assert s1 > s2

    def test_official_video_bonus(self, config):
        r = _yt("Artist - Song (Official Video)", channel="Artist Topic")
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "official_video" in bd

    def test_topic_channel_bonus(self, config):
        r = _yt("Artist - Song", channel="artist - topic")
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "topic_channel" in bd
        assert bd["topic_channel"] == config.TOPIC_CHANNEL_BONUS

    def test_vevo_channel_bonus(self, config):
        r = _yt("Artist - Song", channel="artistvevo")
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "vevo_channel" in bd
        assert bd["vevo_channel"] == config.VEVO_CHANNEL_BONUS

    def test_artist_in_channel_bonus(self, config):
        r = _yt("Artist - Song", channel="Artist Official Channel")
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "artist_in_channel" in bd
        assert bd["artist_in_channel"] == 25

    def test_high_views_bonus(self, config):
        r = _yt("Artist - Song", view_count=5_000_000)
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "high_views" in bd
        assert bd["high_views"] == 5

    def test_low_views_no_bonus(self, config):
        r = _yt("Artist - Song", view_count=1000)
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "high_views" not in bd

    def test_dj_channel_penalty(self, config):
        r = _yt("Artist - Song", channel="dj remixes channel")
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "dj_channel_penalty" in bd
        assert bd["dj_channel_penalty"] == -25

    def test_lyrics_penalty(self, config):
        r = _yt("Artist - Song (Lyrics)", channel="Artist Topic")
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "lyrics_penalty" in bd
        assert bd["lyrics_penalty"] == -20

    def test_lyric_video_penalty(self, config):
        r = _yt("Artist - Song Lyric Video", channel="Artist Topic")
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "lyrics_penalty" in bd

    def test_cross_source_consensus_standard(self, config):
        r = _yt("Artist - Song", channel="Artist Topic", _source_count=2)
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "cross_source_consensus" in bd
        assert bd["cross_source_consensus"] == 10


# ===================================================================
# Duration matching (standard path)
# ===================================================================

class TestDurationMatching:
    def test_duration_exact(self, config):
        r = _yt("Artist - Song", duration=200)
        _, bd = score_youtube_result(r, "Artist", "Song", 200, config)
        assert "duration_exact" in bd
        assert bd["duration_exact"] == config.DURATION_MATCH_BONUS

    def test_duration_close(self, config):
        r = _yt("Artist - Song", duration=200)
        _, bd = score_youtube_result(r, "Artist", "Song", 210, config)
        assert "duration_close" in bd
        assert bd["duration_close"] == 10

    def test_duration_acceptable(self, config):
        r = _yt("Artist - Song", duration=200)
        _, bd = score_youtube_result(r, "Artist", "Song", 220, config)
        assert "duration_acceptable" in bd
        assert bd["duration_acceptable"] == 0

    def test_duration_mismatch_penalizes(self, config):
        r = _yt("Artist - Song", duration=200)
        _, bd = score_youtube_result(r, "Artist", "Song", 400, config)
        assert "duration_mismatch" in bd
        assert bd["duration_mismatch"] == -35

    def test_no_duration_when_zero(self, config):
        r = _yt("Artist - Song", duration=0)
        _, bd = score_youtube_result(r, "Artist", "Song", 200, config)
        # duration=0 means result_duration is 0, so no duration signal
        assert "duration_exact" not in bd

    def test_no_duration_when_mb_missing(self, config):
        r = _yt("Artist - Song", duration=200)
        _, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert "duration_exact" not in bd


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_title(self, config):
        r = _yt("", channel="Artist Topic")
        score, bd = score_youtube_result(r, "Artist", "Song", None, config)
        # Should not crash, likely rejects due to low song match
        assert isinstance(score, int)

    def test_missing_channel(self, config):
        r = {"title": "Artist - Song", "duration": 200}
        score, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert isinstance(score, int)

    def test_missing_duration(self, config):
        r = {"title": "Artist - Song", "channel": "Artist Topic"}
        score, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert isinstance(score, int)

    def test_zero_duration(self, config):
        r = _yt("Artist - Song", duration=0)
        score, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert isinstance(score, int)

    def test_very_long_title(self, config):
        long_title = "Artist - Song " + "x " * 100
        r = _yt(long_title, channel="Artist Topic")
        score, bd = score_youtube_result(r, "Artist", "Song", None, config)
        assert isinstance(score, int)

    def test_unicode_in_title(self, config):
        r = _yt("Artista - Canción (Audio Oficial)", channel="Artista Topic")
        score, bd = score_youtube_result(r, "Artista", "Canción", None, config)
        assert isinstance(score, int)

    def test_result_dict_not_mutated(self, config):
        r = _yt("Artist - Song", channel="Artist Topic")
        original_keys = set(r.keys())
        score_youtube_result(r, "Artist", "Song", None, config)
        assert set(r.keys()) == original_keys


# ===================================================================
# rank_results
# ===================================================================

class TestRankResults:
    def test_returns_empty_for_no_results(self, config):
        ranked = rank_results([], "Artist", "Song", None, config)
        assert ranked == []

    def test_filters_by_min_duration(self, config):
        results = [
            _yt("Artist - Song 1", duration=30),   # too short
            _yt("Artist - Song 2", duration=200),  # ok
        ]
        ranked = rank_results(results, "Artist", "Song", None, config)
        assert len(ranked) == 1
        assert ranked[0][0]["title"] == "Artist - Song 2"

    def test_filters_by_max_duration(self, config):
        results = [
            _yt("Artist - Song 1", duration=200),   # ok
            _yt("Artist - Song 2", duration=5000),  # too long
        ]
        ranked = rank_results(results, "Artist", "Song", None, config)
        assert len(ranked) == 1

    def test_custom_duration_bounds(self, config):
        results = [
            _yt("Artist - Song 1", duration=100),
            _yt("Artist - Song 2", duration=300),
            _yt("Artist - Song 3", duration=500),
        ]
        ranked = rank_results(results, "Artist", "Song", None, config, min_duration=150, max_duration=400)
        assert len(ranked) == 1
        assert ranked[0][0]["title"] == "Artist - Song 2"

    def test_sorts_descending_by_score(self, config):
        results = [
            _yt("Artist - Song (Official Audio)", channel="artist topic", duration=200),
            _yt("Artist - Song (Lyrics)", channel="random", duration=200),
        ]
        ranked = rank_results(results, "Artist", "Song", None, config)
        if len(ranked) >= 2:
            assert ranked[0][1] >= ranked[1][1]

    def test_attaches_score_fields(self, config):
        results = [_yt("Artist - Song", channel="Artist Topic", duration=200)]
        ranked = rank_results(results, "Artist", "Song", None, config)
        assert len(ranked) == 1
        entry, score, breakdown = ranked[0]
        assert "_composite_score" in entry
        assert "_score_breakdown" in entry
        assert isinstance(score, int)
        assert isinstance(breakdown, dict)

    def test_filters_out_all_if_none_in_range(self, config):
        results = [
            _yt("Artist - Song 1", duration=10),
            _yt("Artist - Song 2", duration=20),
        ]
        ranked = rank_results(results, "Artist", "Song", None, config, min_duration=60)
        assert ranked == []

    def test_handles_results_without_duration(self, config):
        results = [
            {"title": "Artist - Song", "channel": "Artist Topic"},
        ]
        ranked = rank_results(results, "Artist", "Song", None, config)
        # No duration field → filtered out
        assert ranked == []
