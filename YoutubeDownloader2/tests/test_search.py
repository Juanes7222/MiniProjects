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


def test_search_with_variants_enforces_total_limit() -> None:
    source_results = [
        [{"id": f"{query}-{index}", "title": str(index), "duration": 200} for index in range(2)]
        for query in range(3)
    ]
    with patch("ytdl_core.search.search_source", side_effect=source_results) as search:
        results = search_with_variants(
            "Artist",
            "Song",
            "youtube",
            {"max_results": 5},
        )

    assert [result["id"] for result in results] == [
        "0-0",
        "0-1",
        "1-0",
        "1-1",
        "2-0",
    ]
    assert search.call_count == 3
    assert all(call.args[2]["max_results"] == 2 for call in search.call_args_list)
