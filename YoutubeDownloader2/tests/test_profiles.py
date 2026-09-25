from __future__ import annotations

from pathlib import Path

import pytest

from ytdl_core.profiles import ProfileError, load_profile


def test_load_profile_normalizes_paths_and_sequences(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        """
[profiles.high-quality]
output = "music"
sources = ["youtube", "soundcloud"]
report = ["json"]
quality = "320"
""",
        encoding="utf-8",
    )

    resolved, settings = load_profile(
        "high-quality",
        path,
        {"output", "sources", "report", "quality"},
    )

    assert resolved == path
    assert settings == {
        "output": Path("music"),
        "sources": ["youtube", "soundcloud"],
        "report": ["json"],
        "quality": "320",
    }


def test_load_profile_rejects_unknown_option(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        "[profiles.broken]\nnot-a-real-option = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="Unknown option"):
        load_profile("broken", path, {"output"})


def test_load_profile_rejects_invalid_delay_shape(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text("[profiles.broken]\ndelay = 2.5\n", encoding="utf-8")

    with pytest.raises(ProfileError, match="two values"):
        load_profile("broken", path, {"delay"})


def test_load_profile_suggests_similar_option(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text("[profiles.broken]\nworker = 2\n", encoding="utf-8")

    with pytest.raises(ProfileError, match="Did you mean 'workers'"):
        load_profile("broken", path, {"workers"})


def test_load_profile_reports_unknown_name(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text('[profiles.existing]\nquality = "320"\n', encoding="utf-8")

    with pytest.raises(ProfileError, match="Available profiles: existing"):
        load_profile("missing", path, {"quality"})
