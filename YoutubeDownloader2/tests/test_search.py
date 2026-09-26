from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

from ytdl_core.search import (
    build_query_variants,
    build_search_query,
    search_source,
    search_with_variants,
)


def test_library_import_does_not_load_rich() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ytdl_core; raise SystemExit('rich' in sys.modules)",
        ],
        check=False,
    )

    assert completed.returncode == 0


def test_search_source_uses_flat_extraction() -> None:
    entries = [{"id": "one", "title": "One", "duration": 200}]
    with patch("ytdl_core.search.yt_dlp.YoutubeDL") as ytdl:
        ytdl.return_value.__enter__.return_value.extract_info.return_value = {"entries": entries}

        results = search_source("Artist Song", "youtube", {"max_results": 1})

    options = ytdl.call_args.args[0]
    assert results == entries
    assert options["extract_flat"] is True
    assert options["skip_download"] is True


def test_query_variants_cover_common_search_shapes() -> None:
    assert build_query_variants("Artist", "Song", "youtube") == [
        "Artist Song official audio",
        "Artist - Song",
        "Artist Song",
    ]
    assert build_query_variants("Artist", "Song", "soundcloud") == ["Song Artist"]


def test_search_query_uses_source_specific_order() -> None:
    assert build_search_query("Artist", "Song", "youtube") == ("Artist Song official audio")
    assert build_search_query("Artist", "Song", "soundcloud") == "Song Artist"


def test_search_with_variants_fetches_every_query_before_capping() -> None:
    """Every query is exhausted before the fetch budget is applied.

    max_results is a presentation budget; truncating mid-fan-out let the first
    variant consume the whole allowance, which is what hid the real recording
    of obscure catalogue tracks.
    """
    source_results = [
        [{"id": f"{query}-{index}", "title": str(index), "duration": 200} for index in range(2)]
        for query in range(3)
    ]
    with patch("ytdl_core.search.search_source", side_effect=source_results) as search:
        results = search_with_variants(
            "Artist",
            "Song",
            "youtube",
            {"max_results": 5, "min_fetch_per_query": 1, "fetch_multiplier": 4},
        )

    assert [result["id"] for result in results] == [
        "0-0",
        "0-1",
        "1-0",
        "1-1",
        "2-0",
        "2-1",
    ]
    assert search.call_count == 3


def test_search_with_variants_raises_per_query_floor() -> None:
    """A small max_results must not shrink the per-query fetch depth."""
    source_results = [[{"id": "a", "title": "a", "duration": 200}]]
    with patch("ytdl_core.search.search_source", side_effect=source_results * 3) as search:
        search_with_variants(
            "Artist",
            "Song",
            "youtube",
            {"max_results": 10, "min_fetch_per_query": 10, "fetch_multiplier": 4},
        )

    assert search.call_count == 3
    assert all(call.args[2]["max_results"] == 10 for call in search.call_args_list)


def test_search_with_variants_caps_at_fetch_multiplier() -> None:
    entries = [{"id": f"id-{index}", "title": str(index), "duration": 200} for index in range(50)]
    with patch("ytdl_core.search.search_source", return_value=entries):
        results = search_with_variants(
            "Artist",
            "Song",
            "youtube",
            {"max_results": 5, "min_fetch_per_query": 5, "fetch_multiplier": 2},
        )

    assert len(results) == 10


def test_canonical_title_adds_queries() -> None:
    plain = build_query_variants("Artist", "Song", "youtube")
    assert plain == [
        "Artist Song official audio",
        "Artist - Song",
        "Artist Song",
    ]

    with_canonical = build_query_variants(
        "Artist", "Song", "youtube", canonical_song="Canonical Song"
    )
    assert with_canonical[:3] == plain
    assert "Artist Canonical Song" in with_canonical
    assert "Artist - Canonical Song" in with_canonical
    assert len(with_canonical) == 5


def test_canonical_title_skipped_when_identical() -> None:
    variants = build_query_variants("Artist", "Song", "youtube", canonical_song="Song")
    assert len(variants) == 3


def test_extra_queries_are_appended_once() -> None:
    variants = build_query_variants(
        "Artist", "Song", "youtube", extra_queries=["Custom Query", "Artist Song"]
    )
    assert variants.count("Custom Query") == 1
    assert variants.count("Artist Song") == 1
    assert variants[-1] == "Custom Query"
