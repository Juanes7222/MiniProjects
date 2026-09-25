from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from ytdl_core.cli import action_summary
from ytdl_core.cli.action_summary import _quote, print_action_summary
from ytdl_core.result import DownloadResult


def _args(**overrides):
    values = {
        "output": Path("downloads"),
        "file": Path("songs.json"),
        "data": None,
        "profile": None,
        "verify": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_posix_quoting_handles_spaces(monkeypatch) -> None:
    monkeypatch.setattr(action_summary.os, "name", "posix")

    assert _quote("Radiohead's Creep") == "'Radiohead'\"'\"'s Creep'"


def test_action_summary_suggests_retry_and_review() -> None:
    console = Console(record=True, width=200, color_system=None)
    results = [
        DownloadResult(artist="Artist", song="Song", reason="Network error"),
        DownloadResult(artist="Artist", song="Other", status="downloaded"),
    ]

    print_action_summary(
        console,
        results,
        _args(),
        {"pending": 1},
    )

    text = console.export_text()
    assert "1 failed" in text
    assert "1 unverified" in text
    assert "1 queued" in text
    assert f"--output {_quote(Path('downloads'))} --retry" in text
    assert f"--file {_quote(Path('songs.json'))} --review" in text


def test_action_summary_reports_completed_library() -> None:
    console = Console(record=True, width=160, color_system=None)
    results = [
        DownloadResult(
            artist="Artist",
            song="Song",
            status="downloaded",
            fingerprint_verified=True,
        )
    ]

    print_action_summary(console, results, _args(), {"pending": 0})

    text = console.export_text()
    assert "1 completed" in text
    assert "Library complete" in text
    assert "--retry" not in text


def test_action_summary_preserves_relevant_profile_settings() -> None:
    console = Console(record=True, width=240, color_system=None)
    results = [DownloadResult(artist="Artist", song="Song", reason="Timeout")]

    print_action_summary(
        console,
        results,
        _args(
            profile="high-quality",
            profiles_path=Path("config/profiles.toml"),
            format="m4a",
            quality="320",
        ),
        {"pending": 1},
        [
            {
                "artist": "Artist",
                "song": "Song",
                "reason": "Timeout",
                "attempts": 3,
            }
        ],
    )

    text = console.export_text()
    assert f"--profile {_quote('high-quality')}" in text
    assert f"--profiles-file {_quote(Path('config/profiles.toml'))}" in text
    assert f"--format {_quote('m4a')}" in text
    assert f"--quality {_quote('320')}" in text
    assert "retry 3x" in text


def test_verify_summary_suggests_repair() -> None:
    console = Console(record=True, width=200, color_system=None)
    results = [DownloadResult(artist="Artist", song="Song", reason="File does not exist")]

    print_action_summary(
        console,
        results,
        _args(verify=True),
        {"pending": 0},
    )

    assert f"--file {_quote(Path('songs.json'))} --repair" in console.export_text()
