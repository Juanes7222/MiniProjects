"""Guards against configuration that looks like a knob and does nothing.

Six fields in :class:`~ytdl_core.config.Config` were declared, documented and
never read: two superseded scoring penalties, one scoring weight that was
dropped in a rewrite, a stage size for a stage that does not exist, a
MusicBrainz app name that had four hardcoded copies instead, and a probe budget
duplicated as a literal. Five of them looked like working settings, so anyone
tuning them would have seen no effect and lost trust in the rest of the config.

This module makes that class of rot a test failure rather than something to
notice by accident.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from ytdl_core.config import Config

PACKAGE = Path(__file__).resolve().parents[1] / "ytdl_core"
CONFIG_FILE = PACKAGE / "config.py"

#: Fields that are legitimately not referenced by name elsewhere.
#:
#: Only for values that are consumed some other way -- currently none. Anything
#: added here should come with the reason it cannot be referenced.
ALLOWED_UNREFERENCED: frozenset[str] = frozenset()

_FIELD = re.compile(r"^    ([A-Z][A-Z0-9_]+)\s*:", re.MULTILINE)


def _declared_fields() -> list[str]:
    return _FIELD.findall(CONFIG_FILE.read_text(encoding="utf-8"))


def _package_body() -> str:
    """Every module except config.py, concatenated."""
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != CONFIG_FILE
    )


def test_the_field_pattern_actually_finds_fields():
    """If the regex stops matching, the guard below passes vacuously."""
    fields = _declared_fields()
    assert len(fields) > 50, f"only matched {len(fields)} fields; the pattern is wrong"
    assert "SCORE_THRESHOLD_REJECT" in fields
    assert "MUSICBRAINZ_APP" in fields


def test_every_config_field_is_actually_read():
    body = _package_body()
    dead = sorted(
        name
        for name in _declared_fields()
        if name not in body and name not in ALLOWED_UNREFERENCED
    )
    assert not dead, (
        f"Config fields declared but never read: {dead}. Either wire them up or "
        "delete them -- a setting that looks live and is not is worse than no "
        "setting, because tuning it appears to do nothing."
    )


def test_no_field_is_declared_twice():
    """A duplicate silently overrides the first, and only the last is reachable."""
    names = _declared_fields()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"declared more than once: {duplicates}"


def test_the_removed_scoring_weights_stay_removed():
    """They are gone on purpose; this records why, in a place that runs.

    ``COVER_KARAOKE_PENALTY`` and ``REACTION_REMIX_PENALTY`` were superseded by
    the forbidden-term hard reject, which discards those titles outright at
    -9999 rather than penalising them. ``HIGH_FUZZY_BONUS`` was a discriminator
    the scorer stopped using; re-adding it would change which song gets
    downloaded for a whole library, which is not a call to make in passing.
    """
    names = set(_declared_fields())
    for gone in ("COVER_KARAOKE_PENALTY", "REACTION_REMIX_PENALTY", "HIGH_FUZZY_BONUS"):
        assert gone not in names, f"{gone} came back; did it get wired, or just restored?"
    assert Config.FORBIDDEN_TERMS >= {"cover", "karaoke", "reaction", "remix"}


def test_musicbrainz_app_is_one_field_not_four_literals():
    """Two call shapes need the value; neither needs its own copy."""
    from ytdl_core.metadata import _app_name_version, _http_user_agent

    assert _app_name_version() == ("YTMusicDownloader", "2.0")
    assert _http_user_agent() == "YTMusicDownloader/2.0"

    package = _package_body()
    assert '"YTMusicDownloader/2.0"' not in package.replace(
        'MUSICBRAINZ_APP: str = "YTMusicDownloader/2.0"', ""
    ), "a User-Agent literal is still hardcoded outside Config"
    assert package.count('musicbrainzngs.set_useragent(') == 1, (
        "set_useragent should be called from one place, not from the CLI as well"
    )


def test_app_name_version_tolerates_a_malformed_value():
    """A half-configured value must not produce an empty User-Agent."""
    from ytdl_core.metadata import _app_name_version

    class Fake:
        MUSICBRAINZ_APP = "NoVersion"

    assert _app_name_version(Fake()) == ("YTMusicDownloader", "2.0")

    class Empty:
        MUSICBRAINZ_APP = ""

    assert _app_name_version(Empty()) == ("YTMusicDownloader", "2.0")

    class Custom:
        MUSICBRAINZ_APP = "MiApp/9.9"

    assert _app_name_version(Custom()) == ("MiApp", "9.9")


def test_config_is_still_a_dataclass_with_defaults():
    """A guard this structural depends on the shape of the class."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    assert "SCORE_THRESHOLD_REJECT" in fields
    for name, field in fields.items():
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            pytest.fail(f"Config.{name} has no default")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("DECISION_MAX_IN_FLIGHT", 6),
        ("DECISION_PROBE_BUDGET_SECONDS", 20.0),
        ("FRAGMENT_CONCURRENCY", 4),
        ("SILENCE_DETECT_SAMPLE_RATE", 8000),
        ("DESCRIPTION_LIMIT", 500),
    ],
)
def test_values_changed_by_the_audit(name, expected):
    """The tuned numbers, asserted so a stray edit cannot quietly undo them."""
    from ytdl_core import decision_questions

    if name == "DESCRIPTION_LIMIT":
        assert getattr(decision_questions, name) == expected
    else:
        assert getattr(Config(), name) == expected