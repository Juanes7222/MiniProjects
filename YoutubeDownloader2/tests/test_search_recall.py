"""Tests for the search-recall, catalogue-source and download-fallback work."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ytdl_core.channels import ChannelTrust
from ytdl_core.config import Config
from ytdl_core.core import MusicDownloader
from ytdl_core.downloader import is_fatal_download_error
from ytdl_core.scorer import score_youtube_result
from ytdl_core.search import search_all_sources, search_ytmusic_album


# ===================================================================
# Forbidden / soft terms
# ===================================================================
class TestForbiddenTerms:
    def _score(self, title: str, artist: str = "Artist", song: str = "Song") -> int:
        entry = {"title": title, "channel": "Chan", "duration": 200}
        return score_youtube_result(entry, artist, song, None, Config())[0]

    def test_cover_still_rejected(self) -> None:
        assert self._score("Song (Cover Audio)") == -9999

    def test_spanish_remaster_no_longer_hard_rejected(self) -> None:
        """'Remasterizado' used to be invisible to an English-only blocklist,
        so 'Remastered' was rejected while its Spanish twin sailed through."""
        assert self._score("Artist - Song (Remastered)") > 0
        assert self._score("Artist - Song (Remasterizado)") > 0
        assert self._score("Artist - Cancion (Remasterizacion 2022)", song="Cancion") > 0

    def test_bare_hora_no_longer_rejected(self) -> None:
        assert self._score("Artist - Cancion de la hora", song="Cancion") > 0

    def test_hour_loops_still_rejected(self) -> None:
        assert self._score("Artist - Song 1 Hora") == -9999
        assert self._score("Artist - Song 1 Hour") == -9999

    def test_bare_version_no_longer_rejected(self) -> None:
        assert self._score("Artist - Song (original version, 1993)") > 0

    def test_version_compounds_still_rejected(self) -> None:
        assert self._score("Artist - Song (Version Merengue)") == -9999
        assert self._score("Artist - Song (Version en Vivo)") == -9999

    def test_remaster_penalised_on_unknown_channel(self) -> None:
        plain = self._score("Artist - Song")
        remastered = self._score("Artist - Song (Remastered)")
        assert remastered < plain

    def test_remaster_penalty_waived_on_topic_channel(self) -> None:
        plain_entry = {"title": "Song", "channel": "Artist - Topic", "duration": 200}
        remastered_entry = {"title": "Song (Remastered)", "channel": "Artist - Topic", "duration": 200}
        plain, plain_bd = score_youtube_result(plain_entry, "Artist", "Song", None, Config())
        remastered, remaster_bd = score_youtube_result(
            remastered_entry, "Artist", "Song", None, Config()
        )
        assert "remaster_penalty" not in remaster_bd
        assert remastered >= plain


# ===================================================================
# Catalogue source
# ===================================================================
class TestCatalogSource:
    def test_itunes_candidate_gets_catalog_bonus(self) -> None:
        entry = {
            "title": "Cariñito Sin Mí",
            "channel": "Pastor López",
            "artists": ["Pastor López"],
            "duration": 237,
            "_source": "itunes",
        }
        score, breakdown = score_youtube_result(
            entry, "Pastor López", "Cariñito Sin Mí", None, Config()
        )
        assert breakdown["official_catalog_match"] == Config().CATALOG_SOURCE_BONUS
        assert score > 100

    def test_itunes_mismatch_falls_through_to_generic_scoring(self) -> None:
        entry = {
            "title": "Something Else Entirely",
            "channel": "Wrong Artist",
            "duration": 200,
            "_source": "itunes",
        }
        _score, breakdown = score_youtube_result(
            entry, "Artist", "Song", None, Config()
        )
        assert "official_catalog_match" not in breakdown

    def test_search_ytmusic_album_requires_album(self) -> None:
        assert search_ytmusic_album("A", "S", "", {}) == []

    def test_search_ytmusic_album_picks_the_matching_track(self) -> None:
        class FakeYTMusic:
            def search(self, query, filter, limit):
                return [{"browseId": "MPREb_x"}]

            def get_album(self, browse_id):
                return {
                    "tracks": [
                        {"title": "Otra Cancion", "videoId": "aaa111"},
                        {
                            "title": "Cariñito Sin Mí",
                            "videoId": "bbb222",
                            "duration_seconds": 237,
                            "artists": [{"name": "Pastor López"}],
                        },
                    ]
                }

        import sys as _sys
        module = type(_sys)("ytmusicapi")
        module.YTMusic = FakeYTMusic
        with patch.dict(_sys.modules, {"ytmusicapi": module}):
            results = search_ytmusic_album("Pastor López", "Cariñito Sin Mi", "30 Pegaditas", {})

        assert len(results) == 1
        assert results[0]["id"] == "bbb222"
        assert results[0]["_source"] == "itunes"
        assert results[0]["duration"] == 237

    def test_search_ytmusic_album_ignores_unrelated_tracklists(self) -> None:
        class FakeYTMusic:
            def search(self, query, filter, limit):
                return [{"browseId": "MPREb_x"}]

            def get_album(self, browse_id):
                return {"tracks": [{"title": "Something Else", "videoId": "zzz999"}]}

        import sys as _sys
        module = type(_sys)("ytmusicapi")
        module.YTMusic = FakeYTMusic
        with patch.dict(_sys.modules, {"ytmusicapi": module}):
            assert search_ytmusic_album("A", "Wanted Song", "Album", {}) == []

    def test_search_ytmusic_album_survives_api_failure(self) -> None:
        import sys as _sys

        module = type(_sys)("ytmusicapi")

        def boom(self):
            raise RuntimeError("no network")

        module.YTMusic = boom
        with patch.dict(_sys.modules, {"ytmusicapi": module}):
            assert search_ytmusic_album("A", "S", "Album", {}) == []

    def test_album_source_runs_only_with_an_album(self) -> None:
        with (
            patch("ytdl_core.search.search_ytmusic_official", return_value=[]),
            patch("ytdl_core.search.search_with_variants", return_value=[]),
            patch("ytdl_core.search.search_ytmusic_album", return_value=[]) as album,
        ):
            search_all_sources("A", "S", ["youtube"], {}, mb_data=None)
            assert album.call_count == 0

            search_all_sources("A", "S", ["youtube"], {}, mb_data={"album": "X"})
            assert album.call_count == 1


class TestSourceOrchestration:
    def test_album_source_skipped_without_youtube(self) -> None:
        with (
            patch("ytdl_core.search.search_ytmusic_official", return_value=[]),
            patch("ytdl_core.search.search_with_variants", return_value=[]),
            patch("ytdl_core.search.search_ytmusic_album", return_value=[]) as album,
        ):
            search_all_sources("A", "S", ["soundcloud"], {}, mb_data={"album": "X"})
            assert album.call_count == 0

    def test_channel_search_skipped_without_trust(self) -> None:
        with (
            patch("ytdl_core.search.search_ytmusic_official", return_value=[]),
            patch("ytdl_core.search.search_with_variants", return_value=[]),
            patch("ytdl_core.search.search_channel_tabs", return_value=[]) as channel,
        ):
            search_all_sources("A", "S", ["youtube"], {}, channel_trust=None)
            assert channel.call_count == 0

    def test_canonical_title_reaches_the_youtube_queries(self) -> None:
        captured: dict = {}

        def fake_variants(artist, song, source, opts, queries=None):
            captured["queries"] = queries
            return []

        with (
            patch("ytdl_core.search.search_ytmusic_official", return_value=[]),
            patch("ytdl_core.search.search_with_variants", side_effect=fake_variants),
        ):
            search_all_sources("A", "S", ["youtube"], {}, mb_data={"title": "Canonical"})

        assert "A Canonical" in captured["queries"]


# ===================================================================
# MusicBrainz reference gating
# ===================================================================
class TestMusicBrainzReference:
    @pytest.fixture
    def dl(self):
        events = _SilentEvents()
        return MusicDownloader(config=Config(), events=events), events

    def test_matching_reference_is_kept(self, dl) -> None:
        downloader, _ = dl
        mb = {"title": "Cariñito Sin Mí", "artist": "Pastor López", "duration_seconds": 237}
        reference, warnings = downloader._resolve_reference(mb, "Pastor López", "Cariñito Sin Mi", False)
        assert reference["duration"] == 237
        assert reference["source"] == "MusicBrainz"
        assert warnings == []

    def test_no_reference_at_all(self, dl) -> None:
        downloader, _ = dl
        assert downloader._resolve_reference(None, "A", "S", False) == (None, [])
        assert downloader._resolve_reference({}, "A", "S", False) == (None, [])

    def test_other_artists_duration_is_discarded(self, dl) -> None:
        """'Con el alma en las manos' is a Jesús Manuel recording. Scoring our
        candidates -35 against his duration demotes the correct answer."""
        downloader, events = dl
        mb = {
            "title": "Con el alma en las manos",
            "artist": "Jesús Manuel",
            "duration_seconds": 303,
        }
        reference, warnings = downloader._resolve_reference(
            mb, "Miguel Morales", "Con el alma en las manos", False
        )
        assert reference is None
        assert any("Jesús Manuel" in w for w in warnings)

    def test_unrelated_title_discards_duration(self, dl) -> None:
        downloader, _ = dl
        mb = {"title": "Totalmente Otra Cosa", "artist": "Artist", "duration_seconds": 200}
        assert downloader._resolve_reference(mb, "Artist", "Song", False) == (None, [])

    def test_absent_reference_fields_are_tolerated(self, dl) -> None:
        downloader, _ = dl
        reference, _ = downloader._resolve_reference(
            {"duration_seconds": 200}, "Artist", "Song", False
        )
        assert reference["duration"] == 200

    def test_no_duration_is_not_a_reference(self, dl) -> None:
        downloader, _ = dl
        assert downloader._resolve_reference(
            {"title": "Song", "artist": "Artist"}, "Artist", "Song", False
        ) == (None, [])

    def test_itunes_fills_in_when_musicbrainz_fails(self, dl) -> None:
        downloader, _ = dl
        payload = {
            "title": "Tan bonita",
            "artist": "Pastor López",
            "album": "30 Pegaditas de Oro",
            "duration": 173,
        }
        with patch("ytdl_core.core.fetch_itunes_reference", return_value=payload):
            reference, _ = downloader._resolve_reference(None, "Pastor López", "Tan bonita")
        assert reference["source"] == "iTunes"
        assert reference["duration"] == 173
        assert reference["album"] == "30 Pegaditas de Oro"

    def test_itunes_is_skipped_without_the_flag(self, dl) -> None:
        downloader, _ = dl
        with patch("ytdl_core.core.fetch_itunes_reference") as lookup:
            downloader._resolve_reference(None, "A", "S", allow_itunes=False)
            assert lookup.call_count == 0

    def test_itunes_also_rejects_misattribution(self, dl) -> None:
        downloader, _ = dl
        payload = {"title": "Coqueta", "artist": "Heredero", "duration": 235}
        with patch("ytdl_core.core.fetch_itunes_reference", return_value=payload):
            reference, warnings = downloader._resolve_reference(None, "Jorge Veloza", "Coqueta")
        assert reference is None
        assert any("Heredero" in w for w in warnings)

    def test_itunes_failure_is_not_fatal(self, dl) -> None:
        downloader, _ = dl
        with patch("ytdl_core.core.fetch_itunes_reference", side_effect=RuntimeError("boom")):
            assert downloader._resolve_reference(None, "A", "S") == (None, [])


# ===================================================================
# Download fallback
# ===================================================================
class TestFatalErrors:
    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: [soundcloud] 123: This video is DRM protected",
            "ERROR: Private video. Sign in if you've been granted access",
            "ERROR: [youtube] abc: Video unavailable",
            "ERROR: [youtube] abc: This video is private",
            "ERROR: [youtube] abc: Sign in to confirm your age",
        ],
    )
    def test_fatal(self, message: str) -> None:
        assert is_fatal_download_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP Error 429: Too Many Requests",
            "Connection reset by peer",
            "ERROR: unable to download video data",
            "",
        ],
    )
    def test_retryable(self, message: str) -> None:
        assert is_fatal_download_error(message) is False


class TestDownloadFallback:
    def _ranked(self):
        def entry(video_id, title, score):
            return (
                {
                    "id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": title,
                    "duration": 200,
                    "_composite_score": score,
                    "_score_breakdown": {},
                },
                score,
                {},
            )

        return [entry("bad", "DRM Rip", 180), entry("good", "Real Upload", 150)]

    def test_falls_through_to_next_candidate(self, tmp_path) -> None:
        dl = MusicDownloader(config=Config(), events=_SilentEvents(), score_threshold=25)
        ranked = self._ranked()
        target = tmp_path / "out.mp3"
        target.write_bytes(b"x")

        with patch(
            "ytdl_core.core.execute_download",
            side_effect=[(None, "This video is DRM protected"), (target, "")],
        ) as download:
            used = dl._download_with_fallback(
                ranked, ranked[0][0], "Artist", "Song", tmp_path, "mp3", "320",
                _Event(), {}, _Lock(), ranked[0][0]["url"], 200, _Result(),
            )

        file, err, replacement = used
        assert file == target
        assert err == ""
        assert replacement["id"] == "good"
        assert download.call_count == 2

    def test_gives_up_when_everything_fails(self, tmp_path) -> None:
        dl = MusicDownloader(config=Config(), events=_SilentEvents(), score_threshold=25)
        ranked = self._ranked()
        with patch(
            "ytdl_core.core.execute_download",
            side_effect=[(None, "DRM protected"), (None, "Video unavailable")],
        ):
            file, err, replacement = dl._download_with_fallback(
                ranked, ranked[0][0], "Artist", "Song", tmp_path, "mp3", "320",
                _Event(), {}, _Lock(), ranked[0][0]["url"], 200, _Result(),
            )
        assert file is None
        assert replacement is None
        assert "unavailable" in err

    def test_does_not_fall_below_the_threshold(self, tmp_path) -> None:
        dl = MusicDownloader(config=Config(), events=_SilentEvents(), score_threshold=160)
        ranked = self._ranked()
        with patch("ytdl_core.core.execute_download", return_value=(None, "boom")) as download:
            dl._download_with_fallback(
                ranked, ranked[0][0], "Artist", "Song", tmp_path, "mp3", "320",
                _Event(), {}, _Lock(), ranked[0][0]["url"], 200, _Result(),
            )
        assert download.call_count == 1

    def test_alternate_resets_the_fingerprint_verdict(self, tmp_path) -> None:
        dl = MusicDownloader(config=Config(), events=_SilentEvents(), score_threshold=25)
        ranked = self._ranked()
        result = _Result()
        result.fingerprint_verified = True
        result.fingerprint_confidence = 0.9
        target = tmp_path / "out.mp3"
        target.write_bytes(b"x")

        with patch(
            "ytdl_core.core.execute_download",
            side_effect=[(None, "DRM protected"), (target, "")],
        ):
            dl._download_with_fallback(
                ranked, ranked[0][0], "Artist", "Song", tmp_path, "mp3", "320",
                _Event(), {}, _Lock(), ranked[0][0]["url"], 200, result,
            )

        assert result.fingerprint_verified is False
        assert result.fingerprint_confidence == 0.0
        assert "alternate" in result.fingerprint_label


class TestStateProvenance:
    def test_channel_is_persisted_on_success(self, tmp_path) -> None:
        state: dict = {"downloads": {}}
        MusicDownloader._persist(
            state, _Lock(), "A::S", "downloaded", "url", "path", "md5", tmp_path,
            channel="Discos Fuentes Edimusica",
            channel_url="https://youtube.com/channel/UC1",
        )
        entry = state["downloads"]["A::S"]
        assert entry["channel"] == "Discos Fuentes Edimusica"
        assert entry["channel_url"] == "https://youtube.com/channel/UC1"

    def test_channel_omitted_when_unknown(self, tmp_path) -> None:
        state: dict = {"downloads": {}}
        MusicDownloader._persist(
            state, _Lock(), "A::S", "downloaded", "url", "path", "md5", tmp_path
        )
        assert "channel" not in state["downloads"]["A::S"]

    def test_failure_preserves_known_provenance(self, tmp_path) -> None:
        state: dict = {"downloads": {}}
        MusicDownloader._persist(
            state, _Lock(), "A::S", "downloaded", "url", "path", "md5", tmp_path,
            channel="Label", channel_url="https://youtube.com/channel/UC1",
        )
        MusicDownloader._persist(state, _Lock(), "A::S", "failed", "url2", None, None, tmp_path)
        assert state["downloads"]["A::S"]["channel"] == "Label"


# ===================================================================
# Helpers
# ===================================================================
class _SilentEvents:
    def on_warn(self, message: str) -> None:
        pass

    def on_download_start(self, *args, **kwargs) -> None:
        pass


class _Event:
    def is_set(self) -> bool:
        return False


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Result:
    def __init__(self) -> None:
        self.fingerprint_verified = False
        self.fingerprint_confidence = 0.0
        self.fingerprint_matched_title = None
        self.fingerprint_label = ""
