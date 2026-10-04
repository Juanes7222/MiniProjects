"""The declared duration is what makes a truncated fingerprint find anything.

AcoustID narrows its candidate set by the duration it is told. ``acoustid.match``
reads that duration off the file it is handed, which for the pipeline's 90-second
excerpt is 90 seconds -- and the excerpt's audio is not from a 90-second
recording, so nothing matches.

Measured on one track that AcoustID recognises at 0.95 from the whole file, using
the same 90 seconds of audio throughout:

    declared  90s / 120s / 180s / 240s / 300s  -> no match
    declared 340s (the real length)             -> match, 0.951
    declared 400s / 600s                        -> no match

So the window is tight around the true figure, which is why these tests pin the
plumbing rather than a duration constant.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ytdl_core import fingerprint as fp
from ytdl_core.fingerprint import _lookup, verify_fingerprint


class _Breaker:
    def allow(self):
        return True

    def record_success(self):
        pass

    def record_failure(self):
        pass

    def release_probe(self):
        pass

    def is_open(self):
        return False


@pytest.fixture
def breaker():
    return _Breaker()


def test_the_tracks_own_duration_is_declared_not_the_clips(tmp_path: Path):
    clip = tmp_path / "partial.mp3"
    clip.write_bytes(b"\x00" * 1024)

    with (
        patch.object(fp.acoustid, "fingerprint_file", return_value=(90.0, ["a", "b"])) as fing,
        patch.object(fp.acoustid, "lookup", return_value={"results": []}) as look,
        patch.object(fp.acoustid, "parse_lookup_result", return_value=iter(())),
    ):
        list(_lookup("key", clip, 340, force_fpcalc=True))

    fing.assert_called_once()
    assert fing.call_args.args[0] == str(clip)
    assert look.call_args.args == ("key", ["a", "b"], 340)


def test_fingerprinting_is_forced_when_a_binary_was_found(tmp_path: Path):
    clip = tmp_path / "p.mp3"
    clip.write_bytes(b"\x00")
    with (
        patch.object(fp.acoustid, "fingerprint_file", return_value=(90.0, [])) as fing,
        patch.object(fp.acoustid, "lookup", return_value={}),
        patch.object(fp.acoustid, "parse_lookup_result", return_value=iter(())),
    ):
        list(_lookup("key", clip, 200, force_fpcalc=True))
    assert fing.call_args.kwargs["force_fpcalc"] is True


def test_without_a_known_duration_it_falls_back_to_match(tmp_path: Path):
    """A wrong duration is worse than the file's own, so an unknown one is not guessed."""
    clip = tmp_path / "p.mp3"
    clip.write_bytes(b"\x00")
    sentinel = object()
    for unknown in (None, 0, -5):
        with patch.object(fp.acoustid, "match", return_value=iter([sentinel])) as match:
            assert list(_lookup("key", clip, unknown)) == [sentinel]
        assert match.call_args.kwargs["meta"] == "recordings"


def test_verify_fingerprint_passes_the_duration_through(tmp_path: Path, breaker):
    clip = tmp_path / "p.mp3"
    clip.write_bytes(b"\x00" * 64)
    with (
        patch.object(fp, "acoustid_bucket") as bucket,
        patch.object(fp, "_lookup", return_value=iter([])) as lookup,
    ):
        bucket.return_value.acquire.return_value = 0.0
        ok, conf, title = verify_fingerprint(
            clip, "A", "S", "key", fp.Config(), breaker, expected_duration=412
        )
    assert (ok, conf, title) == (False, 0.0, "")
    assert lookup.call_args.args[2] == 412, "the duration must reach the lookup"


def test_a_second_alternate_declares_its_own_duration(tmp_path: Path):
    """Each candidate is a different recording, so each needs its own length.

    Reusing the winner's duration would search for the wrong track -- which is
    the exact failure this whole change exists to fix.
    """
    from ytdl_core.core import MusicDownloader

    dl = MusicDownloader(acoustid_key="key")
    seen: list[object] = []

    def fake_one(url, artist, song, output_dir, expected_duration=None):
        seen.append(expected_duration)
        return False, 0.0, None, "no match"

    dl._fingerprint_one = fake_one  # type: ignore[method-assign]
    dl._fingerprint_cache.get = lambda *a, **k: None  # type: ignore[method-assign]

# A title is required: _diversified_alternates drops untitled candidates, and
    # it needs distinct titles to choose between.
    ranked = [
        ({"title": "Song (Official)", "webpage_url": "u0", "duration": 100}, 100, {}),
        ({"title": "Song Live", "webpage_url": "u1", "duration": 250}, 90, {}),
        ({"title": "Song Acoustic", "webpage_url": "u2", "duration": 375}, 80, {}),
        ({"title": "Song Karaoke", "webpage_url": "u3", "duration": 500}, 70, {}),
    ]
    dl._try_next_fp(ranked, "A", "S", tmp_path, type("R", (), {"duration_seconds": 100})())

    # Fan-out is 3 by default, so the first three alternates are tried, each with
    # its own duration; a candidate with no duration falls back rather than
    # declaring a wrong one.
    assert seen == [250, 375, 500], seen


def test_the_downloader_threads_the_winners_duration(tmp_path: Path):
    """The candidate chosen by the scorer carries its own length into the check."""
    from ytdl_core.core import MusicDownloader

    dl = MusicDownloader(acoustid_key="key")
    captured: dict = {}

    def fake_one(url, artist, song, output_dir, expected_duration=None):
        captured["url"] = url
        captured["duration"] = expected_duration
        return False, 0.0, None, "no match"

    dl._fingerprint_one = fake_one  # type: ignore[method-assign]
    dl._fingerprint_cache.get = lambda *a, **k: None  # type: ignore[method-assign]
    dl._fingerprint_cache.coalesce = lambda *a, **k: False  # type: ignore[method-assign]

    result = type("R", (), {"duration_seconds": 317, "heuristic_score": 50, "fingerprint_verified": False,
                            "fingerprint_confidence": 0.0, "fingerprint_matched_title": None,
                            "fingerprint_label": None})()
    dl._fingerprint_check("A", "S", "https://x/y", tmp_path, None, [], result)

    assert captured["duration"] == 317


def test_an_unknown_duration_reaches_the_lookup_as_none(tmp_path: Path):
    """Zero must not become a lookup with duration 0."""
    from ytdl_core.core import MusicDownloader

    dl = MusicDownloader(acoustid_key="key")
    captured: dict = {}

    def fake_one(url, artist, song, output_dir, expected_duration=None):
        captured["duration"] = expected_duration
        return False, 0.0, None, "no match"

    dl._fingerprint_one = fake_one  # type: ignore[method-assign]
    dl._fingerprint_cache.get = lambda *a, **k: None  # type: ignore[method-assign]
    dl._fingerprint_cache.coalesce = lambda *a, **k: False  # type: ignore[method-assign]

    result = type("R", (), {"duration_seconds": 0, "heuristic_score": 50, "fingerprint_verified": False,
                            "fingerprint_confidence": 0.0, "fingerprint_matched_title": None,
                            "fingerprint_label": None})()
    dl._fingerprint_check("A", "S", "https://x/y", tmp_path, None, [], result)

    assert captured["duration"] is None