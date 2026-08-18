"""End-to-end tests for ytdl_core.core.MusicDownloader."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ytdl_core.config import Config
from ytdl_core.core import MusicDownloader
from ytdl_core.result import DownloadResult
from tests.conftest import SpyEvents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spy():
    return SpyEvents()


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def dl(spy, config):
    """MusicDownloader wired to SpyEvents with no delays."""
    return MusicDownloader(
        config=config, events=spy, delay=(0, 0), workers=1,
        no_silence_check=True, skip_fingerprint=True,
    )


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "downloads"
    d.mkdir()
    return d


def _fake_search_result(title="Artist - Song", duration=200, channel="Artist Topic", score=80):
    """Build a fake ranked candidate as select_best_result would return."""
    return {
        "title": title, "channel": channel, "uploader": channel,
        "duration": duration, "url": "http://example.com/video",
        "webpage_url": "http://example.com/video", "thumbnail": None,
        "_source": "youtube", "_composite_score": score,
        "_score_breakdown": {"base_match": score},
    }


def _mock_search_returns_one():
    """Patch search_all_sources to return one result, select_best_result to pick it."""
    result = _fake_search_result()

    def fake_search(artist, song, sources, opts):
        return [result]

    def fake_select(results, artist, song, mb_dur, config, console, lock,
                    min_d, max_d, threshold):
        scored = [(result, result["_composite_score"], result["_score_breakdown"])]
        return result, scored

    return fake_search, fake_select


# ===================================================================
# Initialization
# ===================================================================

class TestInitialization:
    def test_defaults(self):
        dl = MusicDownloader()
        assert dl.config is not None
        assert dl.events is not None
        assert dl.acoustid_key is None
        assert dl.workers >= 1
        assert dl.delay == (2.0, 5.0)
        assert dl.sources == ["youtube", "soundcloud"]

    def test_custom_params(self, config, spy):
        dl = MusicDownloader(
            config=config, events=spy, acoustid_key="KEY",
            force_fingerprint=True, workers=4, delay=(1, 2),
            musicbrainz=True, proxy="http://proxy:8080",
        )
        assert dl.acoustid_key == "KEY"
        assert dl.force_fingerprint is True
        assert dl.workers == 4
        assert dl.delay == (1, 2)
        assert dl.musicbrainz is True
        assert dl.proxy == "http://proxy:8080"

    def test_workers_capped(self):
        dl = MusicDownloader(workers=999)
        assert dl.workers <= Config().MAX_WORKERS

    def test_workers_minimum_one(self):
        dl = MusicDownloader(workers=0)
        assert dl.workers == 1

    def test_threshold_default(self):
        dl = MusicDownloader()
        assert dl.score_threshold == Config().SCORE_THRESHOLD_REJECT

    def test_threshold_custom(self):
        dl = MusicDownloader(score_threshold=50)
        assert dl.score_threshold == 50


# ===================================================================
# _iter_entries (static, no mocks needed)
# ===================================================================

class TestIterEntries:
    def test_flat_entry(self):
        data = {"title": "Song", "entries": None}
        entries = list(MusicDownloader._iter_entries(data))
        assert len(entries) == 1
        assert entries[0]["title"] == "Song"

    def test_nested_entries(self):
        data = {
            "entries": [
                {"title": "Song1", "entries": None},
                {"entries": [
                    {"title": "Song2", "entries": None},
                    {"title": "Song3", "entries": None},
                ]},
            ]
        }
        entries = list(MusicDownloader._iter_entries(data))
        assert len(entries) == 3
        titles = [e["title"] for e in entries]
        assert titles == ["Song1", "Song2", "Song3"]

    def test_empty_entries(self):
        data = {"entries": []}
        entries = list(MusicDownloader._iter_entries(data))
        # Empty list is falsy → falls through to yield data itself
        assert len(entries) == 1

    def test_none_entries_filtered(self):
        data = {"entries": [None, {"title": "Song", "entries": None}, None]}
        entries = list(MusicDownloader._iter_entries(data))
        assert len(entries) == 1


# ===================================================================
# _persist (static)
# ===================================================================

class TestPersist:
    def test_creates_state_entry(self, tmp_path):
        state = {"downloads": {}}
        lock = threading.Lock()
        MusicDownloader._persist(
            state, lock, "Artist::Song", "downloaded",
            "http://url", "/path/file.mp3", "abc123", tmp_path,
        )
        entry = state["downloads"]["Artist::Song"]
        assert entry["status"] == "downloaded"
        assert entry["url"] == "http://url"
        assert entry["md5"] == "abc123"
        assert "timestamp" in entry

    def test_preserves_timestamp(self, tmp_path):
        state = {"downloads": {
            "Artist::Song": {"timestamp": "2024-01-01T00:00:00", "status": "downloaded"},
        }}
        lock = threading.Lock()
        MusicDownloader._persist(
            state, lock, "Artist::Song", "verified",
            None, None, None, tmp_path, preserve_timestamp=True,
        )
        assert state["downloads"]["Artist::Song"]["timestamp"] == "2024-01-01T00:00:00"

    def test_writes_state_file(self, tmp_path):
        state = {"downloads": {}}
        lock = threading.Lock()
        MusicDownloader._persist(state, lock, "A::S", "downloaded", None, None, None, tmp_path)
        state_file = tmp_path / ".download_state.json"
        assert state_file.exists()
        loaded = json.loads(state_file.read_text())
        assert loaded["downloads"]["A::S"]["status"] == "downloaded"


# ===================================================================
# download() — single song
# ===================================================================

class TestDownload:
    def test_successful_download(self, dl, output_dir, spy):
        fake_file = output_dir / "Artist" / "Song.mp3"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"\x00" * 60000)

        fake_search, fake_select = _mock_search_returns_one()

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(fake_file, "")), \
             patch("ytdl_core.core.check_duration", return_value=(True, 200, None)), \
             patch("ytdl_core.core.check_silence", return_value=(0.0, False, None)), \
             patch("ytdl_core.core.enrich_musicbrainz", return_value=(None, False)), \
             patch("ytdl_core.core.embed_and_verify", return_value=True), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "downloaded"
        assert result.file_path == fake_file
        assert result.artist == "Artist"
        assert result.song == "Song"
        assert result.duration_seconds == 200
        assert any(c[0] == "on_download_start" for c in spy.calls)

    def test_search_failure(self, dl, output_dir, spy):
        def fake_search(artist, song, sources, opts):
            return []

        def fake_select(results, *args, **kwargs):
            return None, []

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "failed"
        assert "No valid result" in result.reason
        assert any(c[0] == "on_search_failed" for c in spy.calls)

    def test_skip_existing_with_matching_md5(self, dl, output_dir, spy):
        fake_file = output_dir / "Artist" / "Song.mp3"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"\x00" * 60000)

        from ytdl_core.utils import compute_md5
        md5 = compute_md5(fake_file)

        state = {"downloads": {
            "Artist::Song": {"status": "downloaded", "md5": md5},
        }}

        with patch("ytdl_core.core.load_state", return_value=state), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir, skip_existing=True)

        assert result.status == "skipped"
        assert result.md5 == md5
        assert any(c[0] == "on_skip_existing" for c in spy.calls)

    def test_skip_existing_md5_mismatch_triggers_redownload(self, dl, output_dir, spy):
        fake_file = output_dir / "Artist" / "Song.mp3"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"\x00" * 60000)

        state = {"downloads": {
            "Artist::Song": {"status": "downloaded", "md5": "wrong_md5"},
        }}

        new_file = output_dir / "Artist" / "Song.mp3"
        fake_search, fake_select = _mock_search_returns_one()

        with patch("ytdl_core.core.load_state", return_value=state), \
             patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(new_file, "")), \
             patch("ytdl_core.core.check_duration", return_value=(True, 200, None)), \
             patch("ytdl_core.core.check_silence", return_value=(0.0, False, None)), \
             patch("ytdl_core.core.enrich_musicbrainz", return_value=(None, False)), \
             patch("ytdl_core.core.embed_and_verify", return_value=True), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir, skip_existing=True)

        assert result.status == "downloaded"
        assert any(c[0] == "on_md5_mismatch" for c in spy.calls)

    def test_download_failure_propagates_error(self, dl, output_dir, spy):
        fake_search, fake_select = _mock_search_returns_one()

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(None, "network error")), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "failed"
        assert "network error" in result.reason
        assert any(c[0] == "on_download_failed" for c in spy.calls)

    def test_user_skip_via_selector(self, dl, output_dir, spy):
        dl.events.selector_fn = lambda a, s, ranked: None

        fake_search, fake_select = _mock_search_returns_one()

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "skipped"
        assert "User skipped" in result.reason

    def test_user_confirm_rejects(self, dl, output_dir, spy):
        dl.events.confirm_fn = lambda a, s, best: False

        fake_search, fake_select = _mock_search_returns_one()

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "skipped"
        assert "User skipped" in result.reason

    def test_user_confirm_accepts(self, dl, output_dir, spy):
        dl.events.confirm_fn = lambda a, s, best: True

        fake_file = output_dir / "Artist" / "Song.mp3"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"\x00" * 60000)

        fake_search, fake_select = _mock_search_returns_one()

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(fake_file, "")), \
             patch("ytdl_core.core.check_duration", return_value=(True, 200, None)), \
             patch("ytdl_core.core.check_silence", return_value=(0.0, False, None)), \
             patch("ytdl_core.core.enrich_musicbrainz", return_value=(None, False)), \
             patch("ytdl_core.core.embed_and_verify", return_value=True), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "downloaded"

    def test_duration_check_failure(self, dl, output_dir, spy):
        fake_search, fake_select = _mock_search_returns_one()
        fake_dl_file = output_dir / "Artist" / "Song.mp3"
        fake_dl_file.parent.mkdir(parents=True, exist_ok=True)
        fake_dl_file.write_bytes(b"\x00" * 60000)

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(fake_dl_file, "")), \
             patch("ytdl_core.core.check_duration", return_value=(False, 50, "Duration discrepancy 80%")), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "failed"
        assert "Duration discrepancy" in result.reason

    def test_silence_check_failure(self, tmp_path, spy, config):
        """Silence check triggers when no_silence_check=False."""
        dl_no_silence = MusicDownloader(
            config=config, events=spy, delay=(0, 0), workers=1,
            no_silence_check=False, skip_fingerprint=True,
        )
        out = tmp_path / "dl"
        out.mkdir()

        fake_search, fake_select = _mock_search_returns_one()
        fake_dl_file = out / "Artist" / "Song.mp3"
        fake_dl_file.parent.mkdir(parents=True, exist_ok=True)
        fake_dl_file.write_bytes(b"\x00" * 60000)

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(fake_dl_file, "")), \
             patch("ytdl_core.core.check_duration", return_value=(True, 200, None)), \
             patch("ytdl_core.core.check_silence", return_value=(0.5, True, "Excessive silence (50.0%)")), \
             patch("ytdl_core.core.apply_delay"):

            result = dl_no_silence.download("Artist", "Song", out)

        assert result.status == "failed"
        assert "silence" in result.reason.lower()

    def test_silence_check_skipped_when_disabled(self, dl, output_dir, spy):
        """When no_silence_check=True, check_silence is never called."""
        fake_search, fake_select = _mock_search_returns_one()
        fake_dl_file = output_dir / "Artist" / "Song.mp3"
        fake_dl_file.parent.mkdir(parents=True, exist_ok=True)
        fake_dl_file.write_bytes(b"\x00" * 60000)

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(fake_dl_file, "")), \
             patch("ytdl_core.core.check_duration", return_value=(True, 200, None)), \
             patch("ytdl_core.core.check_silence") as mock_silence, \
             patch("ytdl_core.core.enrich_musicbrainz", return_value=(None, False)), \
             patch("ytdl_core.core.embed_and_verify", return_value=True), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        mock_silence.assert_not_called()
        assert result.status == "downloaded"

    def test_metadata_embed_failure(self, dl, output_dir, spy):
        fake_search, fake_select = _mock_search_returns_one()
        fake_dl_file = output_dir / "Artist" / "Song.mp3"
        fake_dl_file.parent.mkdir(parents=True, exist_ok=True)
        fake_dl_file.write_bytes(b"\x00" * 60000)

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(fake_dl_file, "")), \
             patch("ytdl_core.core.check_duration", return_value=(True, 200, None)), \
             patch("ytdl_core.core.check_silence", return_value=(0.0, False, None)), \
             patch("ytdl_core.core.enrich_musicbrainz", return_value=(None, False)), \
             patch("ytdl_core.core.embed_and_verify", return_value=False), \
             patch("ytdl_core.core.apply_delay"):

            result = dl.download("Artist", "Song", output_dir)

        assert result.status == "failed"
        assert "Metadata integrity" in result.reason

    def test_stop_event_skips(self, dl, output_dir, spy):
        # Pre-set a stop event by using _process_song directly
        state = {"downloads": {}}
        stop = threading.Event()
        stop.set()

        result = dl._process_song(
            "Artist", "Song", output_dir, "mp3", "192", False,
            state, threading.Lock(), stop, set(), threading.Lock(), [("Artist", "Song")],
        )
        assert result.status == "skipped"
        assert result.reason == "Interrupted"


# ===================================================================
# download_batch()
# ===================================================================

class TestDownloadBatch:
    def test_empty_songs(self, dl, output_dir, spy):
        results = dl.download_batch({}, output_dir)
        assert results == []
        assert any(c[0] == "on_session_start" for c in spy.calls)
        assert any(c[0] == "on_session_complete" for c in spy.calls)

    def test_multiple_songs(self, dl, output_dir, spy):
        songs = {"Artist": ["Song1", "Song2"]}

        def fake_search(artist, song, sources, opts):
            return [_fake_search_result(title=f"{artist} - {song}")]

        def fake_select(results, artist, song, mb_dur, config, console, lock,
                        min_d, max_d, threshold):
            r = results[0]
            return r, [(r, r["_composite_score"], r["_score_breakdown"])]

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", side_effect=lambda *a, **k: (
                 lambda p: (p, ""))(output_dir / "tmp.mp3")), \
             patch("ytdl_core.core.check_duration", return_value=(True, 200, None)), \
             patch("ytdl_core.core.check_silence", return_value=(0.0, False, None)), \
             patch("ytdl_core.core.enrich_musicbrainz", return_value=(None, False)), \
             patch("ytdl_core.core.embed_and_verify", return_value=True), \
             patch("ytdl_core.core.apply_delay"):

            # Create the fake files that execute_download would return
            for s in ["Song1", "Song2"]:
                f = output_dir / "Artist" / f"{s}.mp3"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"\x00" * 60000)

            def fake_exec(url, output_dir, fmt, quality, artist, song, events, config,
                         stop, state, state_lock, cb, cf, proxy):
                f = output_dir / "Artist" / f"{song}.mp3"
                return f, ""

            with patch("ytdl_core.core.execute_download", side_effect=fake_exec):
                results = dl.download_batch(songs, output_dir)

        assert len(results) == 2
        statuses = [r.status for r in results]
        assert all(s == "downloaded" for s in statuses)

    def test_batch_with_report_formats(self, dl, output_dir, spy):
        with patch("ytdl_core.core.export_report") as mock_export:
            dl.download_batch({}, output_dir, report_formats=["json", "csv"])
            # export_report is called even with empty results list
            mock_export.assert_called_once()

    def test_batch_state_persisted(self, dl, output_dir, spy):
        songs = {"Artist": ["Song1"]}

        with patch("ytdl_core.core.search_all_sources", return_value=[]), \
             patch("ytdl_core.core.select_best_result", return_value=(None, [])), \
             patch("ytdl_core.core.apply_delay"):

            results = dl.download_batch(songs, output_dir)

        assert results[0].status == "failed"
        # State should have been persisted with failed status
        state = json.loads((output_dir / ".download_state.json").read_text())
        assert state["downloads"]["Artist::Song1"]["status"] == "failed"


# ===================================================================
# download_url()
# ===================================================================

class TestDownloadUrl:
    def test_scan_failure(self, dl, output_dir, spy):
        with patch("ytdl_core.core.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.side_effect = RuntimeError("network error")

            dl.download_url("http://example.com/playlist", output_dir)

        assert any(c[0] == "on_download_failed" for c in spy.calls)

    def test_no_entries(self, dl, output_dir, spy):
        with patch("ytdl_core.core.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = None

            dl.download_url("http://example.com/video", output_dir)

        # Should not crash, no session start
        assert not any(c[0] == "on_session_start" for c in spy.calls)

    def test_filters_live_streams(self, dl, output_dir, spy):
        with patch("ytdl_core.core.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = {
                "entries": [
                    {"title": "Live Stream", "uploader": "Channel",
                     "is_live": True, "duration": 200,
                     "webpage_url": "http://example.com/live"},
                ]
            }

            dl.download_url("http://example.com/playlist", output_dir)

        assert any("live" in str(c) for c in spy.calls)

    def test_match_title_filter(self, dl, output_dir, spy):
        with patch("ytdl_core.core.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = {
                "entries": [
                    {"title": "Artist - Song", "uploader": "Channel",
                     "duration": 200, "webpage_url": "http://example.com/1"},
                    {"title": "Artist - Other", "uploader": "Channel",
                     "duration": 200, "webpage_url": "http://example.com/2"},
                ]
            }

            dl.download_url("http://example.com/playlist", output_dir, match_title="Song")

        warns = [c for c in spy.calls if c[0] == "on_warn"]
        skipped = [c for c in warns if "no match" in str(c)]
        assert len(skipped) >= 1

    def test_reject_title_filter(self, dl, output_dir, spy):
        with patch("ytdl_core.core.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = {
                "entries": [
                    {"title": "Artist - Song", "uploader": "Channel",
                     "duration": 200, "webpage_url": "http://example.com/1"},
                    {"title": "Artist - Remix", "uploader": "Channel",
                     "duration": 200, "webpage_url": "http://example.com/2"},
                ]
            }

            dl.download_url("http://example.com/playlist", output_dir, reject_title="Remix")

        warns = [c for c in spy.calls if c[0] == "on_warn"]
        rejected = [c for c in warns if "rejected" in str(c)]
        assert len(rejected) >= 1

    def test_duration_too_short_skipped(self, dl, output_dir, spy):
        dl.min_duration = 100
        with patch("ytdl_core.core.yt_dlp.YoutubeDL") as MockYDL:
            instance = MockYDL.return_value.__enter__.return_value
            instance.extract_info.return_value = {
                "entries": [
                    {"title": "Short Song", "uploader": "Channel",
                     "duration": 10, "webpage_url": "http://example.com/short"},
                ]
            }

            dl.download_url("http://example.com/playlist", output_dir)

        warns = [c for c in spy.calls if c[0] == "on_warn"]
        short_skipped = [c for c in warns if "too short" in str(c)]
        assert len(short_skipped) >= 1


# ===================================================================
# verify_library()
# ===================================================================

class TestVerifyLibrary:
    def test_delegates_to_verifier(self, dl, output_dir, spy):
        with patch("ytdl_core.core._verify_library") as mock_verify:
            mock_verify.return_value = [
                DownloadResult(artist="A", song="S", status="verified"),
            ]

            results = dl.verify_library({"A": ["S"]}, output_dir)

        assert len(results) == 1
        assert results[0].status == "verified"
        mock_verify.assert_called_once()

    def test_empty_songs(self, dl, output_dir, spy):
        results = dl.verify_library({}, output_dir)
        assert results == []


# ===================================================================
# Event callbacks coverage
# ===================================================================

class TestEventCallbacks:
    def test_on_session_start_fired(self, dl, output_dir, spy):
        dl.download_batch({}, output_dir)
        start_calls = [c for c in spy.calls if c[0] == "on_session_start"]
        assert len(start_calls) >= 1

    def test_on_session_complete_fired(self, dl, output_dir, spy):
        dl.download_batch({}, output_dir)
        complete_calls = [c for c in spy.calls if c[0] == "on_session_complete"]
        assert len(complete_calls) >= 1

    def test_on_result_fired_per_song(self, dl, output_dir, spy):
        def fake_search(artist, song, sources, opts):
            return [_fake_search_result(title=f"{artist} - {song}")]

        def fake_select(results, artist, song, mb_dur, config, console, lock,
                        min_d, max_d, threshold):
            r = results[0]
            return r, [(r, r["_composite_score"], r["_score_breakdown"])]

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.execute_download", return_value=(None, "fail")), \
             patch("ytdl_core.core.apply_delay"):

            dl.download_batch({"Artist": ["Song1", "Song2"]}, output_dir)

        result_calls = [c for c in spy.calls if c[0] == "on_result"]
        assert len(result_calls) == 2

    def test_on_search_start_fired(self, dl, output_dir, spy):
        def fake_search(artist, song, sources, opts):
            return []

        def fake_select(results, *a, **kw):
            return None, []

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.apply_delay"):

            dl.download("Artist", "Song", output_dir)

        search_calls = [c for c in spy.calls if c[0] == "on_search_start"]
        assert len(search_calls) >= 1

    def test_on_artist_start_fired(self, dl, output_dir, spy):
        def fake_search(artist, song, sources, opts):
            return []

        def fake_select(results, *a, **kw):
            return None, []

        with patch("ytdl_core.core.search_all_sources", fake_search), \
             patch("ytdl_core.core.select_best_result", fake_select), \
             patch("ytdl_core.core.apply_delay"):

            dl.download("Artist", "Song", output_dir)

        artist_calls = [c for c in spy.calls if c[0] == "on_artist_start"]
        assert len(artist_calls) >= 1
        assert artist_calls[0][1][0] == "Artist"
