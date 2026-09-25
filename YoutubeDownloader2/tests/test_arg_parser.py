from __future__ import annotations

from pathlib import Path

import pytest

from ytdl_core.cli.arg_parser import parse_args


def test_profile_can_supply_the_source(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        """
[profiles.high-quality]
file = "songs.json"
quality = "320"
sources = ["youtube", "soundcloud"]
""",
        encoding="utf-8",
    )

    args = parse_args(["--profile", "high-quality", "--profiles-file", str(path)])

    assert args.file == Path("songs.json")
    assert args.quality == "320"
    assert args.sources == ["youtube", "soundcloud"]


def test_explicit_arguments_override_profile(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        '[profiles.high-quality]\nfile = "songs.json"\nquality = "320"\n',
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--profile",
            "high-quality",
            "--profiles-file",
            str(path),
            "--quality",
            "192",
        ]
    )

    assert args.quality == "192"


def test_explicit_report_replaces_profile_report(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        '[profiles.reports]\nfile = "songs.json"\nreport = ["json"]\n',
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--profile",
            "reports",
            "--profiles-file",
            str(path),
            "--report",
            "csv",
        ]
    )

    assert args.report == ["csv"]


def test_retry_is_a_valid_source() -> None:
    args = parse_args(["--retry", "--output", "downloads"])

    assert args.retry is True


def test_profile_choices_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        '[profiles.broken]\nfile = "songs.json"\nformat = "wav"\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        parse_args(["--profile", "broken", "--profiles-file", str(path)])


def test_explicit_retry_overrides_profile_source(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text('[profiles.queue]\nfile = "songs.json"\n', encoding="utf-8")

    args = parse_args(
        [
            "--profile",
            "queue",
            "--profiles-file",
            str(path),
            "--retry",
        ]
    )

    assert args.retry is True
    assert args.file is None


def test_profile_cannot_define_multiple_sources(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        '[profiles.broken]\nfile = "songs.json"\nurl = "http://example.com"\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        parse_args(["--profile", "broken", "--profiles-file", str(path)])


def test_profiles_file_requires_profile() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--file", "songs.json", "--profiles-file", "profiles.toml"])


def test_retry_conflicts_with_verify() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--retry", "--verify"])


def test_abbreviated_source_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--ret"])


def test_missing_source_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
