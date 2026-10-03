"""Tests for album-level catalogue resolution (ytdl_core.metadata)."""

from __future__ import annotations

from unittest.mock import patch

from ytdl_core.config import Config
from ytdl_core.metadata import CatalogContext

RELEASE = {
    "release_id": "rel-1",
    "album": "Album Uno",
    "year": "1998",
    "artist": "Artista",
    "cover_url": "https://coverartarchive.org/release/rel-1/front",
    "tracks": [
        {
            "mb_id": "mb-1",
            "title": "Primera Cancion",
            "artist": "Artista",
            "duration_seconds": 210,
            "track_num": "1",
            "genre": "Cumbia",
        },
        {
            "mb_id": "mb-2",
            "title": "Segunda Cancion",
            "artist": "Artista",
            "duration_seconds": 185,
            "track_num": "2",
            "genre": "Cumbia",
        },
    ],
}


def _context(musicbrainz: bool = True) -> CatalogContext:
    return CatalogContext(Config(), musicbrainz=musicbrainz)


class TestTracklistMatching:
    def test_no_release_loaded_returns_none(self):
        assert _context().from_tracklist("Artista", "Primera Cancion") is None

    def test_disabled_context_returns_none(self):
        context = _context(musicbrainz=False)
        context._by_artist["artista"] = RELEASE
        assert context.from_tracklist("Artista", "Primera Cancion") is None

    def test_exact_title_matches(self):
        context = _context()
        context._by_artist["artista"] = RELEASE
        entry = context.from_tracklist("Artista", "Primera Cancion")
        assert entry is not None
        assert entry["mb_id"] == "mb-1"
        assert entry["duration_seconds"] == 210
        assert entry["album"] == "Album Uno"
        assert entry["year"] == "1998"
        assert entry["track_num"] == "1"

    def test_tolerates_accents_and_case(self):
        context = _context()
        context._by_artist["artista"] = RELEASE
        assert context.from_tracklist("Artista", "primera cancion") is not None

    def test_one_release_answers_many_songs(self):
        """The point: a single lookup covers every track of the album."""
        context = _context()
        context._by_artist["artista"] = RELEASE
        first = context.from_tracklist("Artista", "Primera Cancion")
        second = context.from_tracklist("Artista", "Segunda Cancion")
        assert first["mb_id"] != second["mb_id"]
        assert first["cover_url"] == second["cover_url"]

    def test_unrelated_title_is_not_forced(self):
        context = _context()
        context._by_artist["artista"] = RELEASE
        assert context.from_tracklist("Artista", "Cancion Que No Existe") is None

    def test_track_by_a_different_artist_is_rejected(self):
        """Its duration is someone else's recording, not a valid reference."""
        release = {
            **RELEASE,
            "tracks": [{**RELEASE["tracks"][0], "artist": "Los Raros"}],
        }
        context = _context()
        context._by_artist["artista"] = release
        assert context.from_tracklist("Artista", "Primera Cancion") is None

    def test_artist_key_is_case_insensitive(self):
        context = _context()
        context._by_artist["artista"] = RELEASE
        assert context.from_tracklist("ARTISTA", "Primera Cancion") is not None


class TestResolve:
    def test_uses_the_search_when_no_release_is_loaded(self):
        context = _context()
        payload = {"release_id": "rel-1", "duration_seconds": 200, "title": "Primera Cancion"}
        with (
            patch("ytdl_core.metadata.fetch_release_tracklist", return_value=RELEASE),
            patch("ytdl_core.metadata.fetch_musicbrainz", return_value=payload) as search,
        ):
            assert context.resolve("Artista", "Primera Cancion") == payload
        search.assert_called_once_with("Artista", "Primera Cancion")

    def test_a_resolved_release_is_remembered(self):
        context = _context()
        with (
            patch("ytdl_core.metadata.fetch_release_tracklist", return_value=RELEASE),
            patch("ytdl_core.metadata.fetch_musicbrainz", return_value={"release_id": "rel-1"}),
        ):
            context.resolve("Artista", "Primera Cancion")
        assert context.from_tracklist("Artista", "Segunda Cancion") is not None

    def test_later_songs_need_no_further_search(self):
        """N-1 catalogue calls saved for an album."""
        context = _context()
        with (
            patch("ytdl_core.metadata.fetch_release_tracklist", return_value=RELEASE),
            patch("ytdl_core.metadata.fetch_musicbrainz", return_value={"release_id": "rel-1"}),
        ):
            context.resolve("Artista", "Primera Cancion")
            with patch("ytdl_core.metadata.fetch_musicbrainz") as second:
                entry = context.resolve("Artista", "Segunda Cancion")
        second.assert_not_called()
        assert entry["mb_id"] == "mb-2"

    def test_disabled_context_makes_no_calls(self):
        context = _context(musicbrainz=False)
        with patch("ytdl_core.metadata.fetch_musicbrainz") as search:
            assert context.resolve("Artista", "Primera Cancion") is None
        search.assert_not_called()

    def test_a_search_without_a_release_id_is_harmless(self):
        context = _context()
        with (
            patch("ytdl_core.metadata.fetch_release_tracklist") as tracklist,
            patch("ytdl_core.metadata.fetch_musicbrainz", return_value={"title": "x"}),
        ):
            assert context.resolve("Artista", "Primera Cancion") == {"title": "x"}
        tracklist.assert_not_called()

    def test_tracklist_failure_falls_back_to_search(self):
        context = _context()
        payload = {"release_id": "rel-1"}
        with (
            patch("ytdl_core.metadata.fetch_release_tracklist", return_value=None),
            patch("ytdl_core.metadata.fetch_musicbrainz", return_value=payload),
        ):
            assert context.resolve("Artista", "Primera Cancion") == payload
        # Second call has no tracklist, so it must search again rather than lie.
        with patch("ytdl_core.metadata.fetch_musicbrainz", return_value=payload) as search:
            context.resolve("Artista", "Primera Cancion")
        search.assert_called_once()


class TestPrime:
    def test_prime_loads_the_album(self):
        context = _context()
        with (
            patch("ytdl_core.metadata.fetch_release_tracklist", return_value=RELEASE),
            patch("ytdl_core.metadata.fetch_musicbrainz", return_value={"release_id": "rel-1"}),
        ):
            context.prime("Artista", "Primera Cancion")
        assert context.from_tracklist("Artista", "Segunda Cancion") is not None

    def test_priming_twice_does_not_reload(self):
        context = _context()
        context._by_artist["artista"] = RELEASE
        with patch("ytdl_core.metadata.fetch_musicbrainz") as search:
            context.prime("Artista", "Primera Cancion")
        search.assert_not_called()

    def test_prime_is_a_no_op_when_disabled(self):
        context = _context(musicbrainz=False)
        with patch("ytdl_core.metadata.fetch_musicbrainz") as search:
            context.prime("Artista", "Primera Cancion")
        search.assert_not_called()


class TestReleaseTracklistParsing:
    def test_builds_entries_from_musicbrainz_shapes(self):
        from ytdl_core.metadata import fetch_release_tracklist

        raw = {
            "id": "rel-9",
            "title": "Disco",
            "date": "2001-05-04",
            "artist-credit": [{"name": "Banda"}, {"name": "invitado"}],
            "medium-list": [
                {
                    "track-list": [
                        {
                            "id": "m1",
                            "title": "Uno",
                            "length": 200500,
                            "position": 1,
                            "artist-credit": [{"name": "Banda"}],
                            "tag-list": [{"name": "Salsa"}],
                        },
                        {"id": "m2", "title": "Dos", "length": 100000, "position": 2},
                    ]
                }
            ],
        }
        with patch("ytdl_core.metadata._throttled_release", return_value=raw):
            release = fetch_release_tracklist("rel-9")

        assert release["album"] == "Disco"
        assert release["year"] == "2001"
        assert release["artist"] == "Banda invitado"
        assert [t["title"] for t in release["tracks"]] == ["Uno", "Dos"]
        assert release["tracks"][0]["duration_seconds"] == 200
        assert release["tracks"][0]["genre"] == "Salsa"
        # A track with no artist of its own inherits the release's.
        assert release["tracks"][1]["artist"] == "Banda invitado"

    def test_returns_none_without_tracks(self):
        from ytdl_core.metadata import fetch_release_tracklist

        with patch("ytdl_core.metadata._throttled_release", return_value={"id": "x", "title": "T"}):
            assert fetch_release_tracklist("x") is None

    def test_request_failure_is_none_not_an_exception(self):
        from ytdl_core.metadata import fetch_release_tracklist

        with patch("ytdl_core.metadata._throttled_release", side_effect=RuntimeError("503")):
            assert fetch_release_tracklist("x") is None