from __future__ import annotations

from pathlib import Path

from ytdl_core.ytdlp_options import build_ytdlp_base_opts


def _postprocessor_keys(options: dict) -> list[str]:
    return [postprocessor["key"] for postprocessor in options["postprocessors"]]


def test_thumbnail_embedding_can_be_disabled(tmp_path: Path) -> None:
    options = build_ytdlp_base_opts(
        tmp_path,
        "mp3",
        "192",
        quiet=True,
        no_warnings=True,
        embed_thumbnail=False,
    )

    assert "writethumbnail" not in options
    assert "EmbedThumbnail" not in _postprocessor_keys(options)


def test_thumbnail_embedding_is_enabled_for_url_downloads(tmp_path: Path) -> None:
    options = build_ytdlp_base_opts(
        tmp_path,
        "mp3",
        "192",
        quiet=True,
        no_warnings=True,
        embed_thumbnail=True,
    )

    assert options["writethumbnail"] is True
    assert "EmbedThumbnail" in _postprocessor_keys(options)
