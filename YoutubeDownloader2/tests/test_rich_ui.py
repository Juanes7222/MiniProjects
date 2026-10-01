from __future__ import annotations

from rich.console import Console

from ytdl_core.cli.rich_ui import RichEvents
from ytdl_core.config import Config


def _events(**kwargs):
    console = Console(record=True, width=240, color_system=None)
    events = RichEvents(
        console,
        score_threshold=25,
        config=Config(),
        decision_threshold=kwargs.pop("decision_threshold", 0.75),
        **kwargs,
    )
    return events, console


def _entry(**overrides) -> dict:
    entry = {
        "title": "Artist - Song",
        "channel": "Official Artist",
        "duration": 200,
        "_decision_provider": "Jev",
        "_decision_probability": 0.74,
        "_decision_runs": 3,
        "_decision_spread": 0.05,
        "_decision_stable": True,
        "_decision_eligible": True,
        "_decision_dimensions": {"identity": 0.97, "origin": 0.92, "studio": 0.81},
        "_heuristic_score": 140,
        "_score_breakdown": {"base_match": 140, "identity": 97, "origin": 92, "studio": 81},
    }
    entry.update(overrides)
    return entry


def test_candidate_table_separates_jev_and_heuristic_scores():
    events, console = _events()
    entry = _entry()

    events.on_candidates_scored("Artist", "Song", [(entry, 74, entry["_score_breakdown"])])
    text = console.export_text()

    assert "Jev" in text
    assert "Heur." in text
    assert "74% x3" in text
    assert "140" in text
    assert ">1" not in text


def test_candidate_table_shows_the_atomic_dimensions():
    """The whole point of the split: you can see *which* judgment was made."""
    events, console = _events()
    entry = _entry()

    events.on_candidates_scored("Artist", "Song", [(entry, 74, entry["_score_breakdown"])])
    text = console.export_text()

    assert "identity 97%" in text
    assert "origin 92%" in text
    assert "studio 81%" in text
    assert "base_match" in text


def test_candidate_table_shows_the_form_gate():
    """`form` is what catches an instrumental the origin gate waves through."""
    events, console = _events()
    entry = _entry(_decision_dimensions={"identity": 0.97, "origin": 0.90, "form": 0.04})

    events.on_candidates_scored("Artist", "Song", [(entry, 40, entry["_score_breakdown"])])
    text = console.export_text()

    assert "form 4%" in text


def test_vetoed_candidate_is_marked_and_not_selected():
    events, console = _events()
    vetoed = _entry(
        title="Some Guy - Song (Cover)",
        _decision_eligible=False,
        _decision_failed_gates=["origin"],
        _decision_probability=0.0,
        _composite_score=0,
    )
    kept = _entry()

    events.on_candidates_scored(
        "Artist",
        "Song",
        [
            (vetoed, 0, vetoed["_score_breakdown"]),
            (kept, 74, kept["_score_breakdown"]),
        ],
    )
    text = console.export_text()

    assert "veto origin" in text
    assert ">2" not in text


def test_vetoed_top_candidate_is_not_marked_as_best():
    events, console = _events()
    vetoed = _entry(
        _decision_eligible=False,
        _decision_failed_gates=["origin", "identity"],
        _decision_probability=0.0,
    )

    events.on_candidates_scored("Artist", "Song", [(vetoed, 0, vetoed["_score_breakdown"])])
    text = console.export_text()

    assert "veto origin/identity" in text
    assert ">1" not in text


def test_unstable_candidate_is_flagged():
    events, console = _events()
    entry = _entry(_decision_stable=False, _decision_spread=0.34)

    events.on_candidates_scored("Artist", "Song", [(entry, 74, entry["_score_breakdown"])])
    text = console.export_text()

    assert "74%?" in text
    assert "Jev spread 34%" in text


def test_choice_confidence_is_surfaced():
    events, console = _events()
    entry = _entry(_decision_confidence=0.81)

    events.on_candidates_scored("Artist", "Song", [(entry, 74, entry["_score_breakdown"])])
    text = console.export_text()

    assert "conf 81%" in text


def test_summary_marks_vetoed_result():
    events, console = _events()
    from ytdl_core.result import DownloadResult

    result = DownloadResult(
        artist="Artist",
        song="Song",
        status="downloaded",
        file_path="song.mp3",
        decision_provider="Jev",
        decision_probability=0.0,
        decision_runs=1,
        decision_threshold=0.60,
        decision_failed_gates=["origin"],
    )

    events.on_session_complete([result], 1.0)
    text = console.export_text()

    assert "veto origin" in text


def test_summary_marks_unstable_result():
    events, console = _events()
    from ytdl_core.result import DownloadResult

    result = DownloadResult(
        artist="Artist",
        song="Song",
        status="downloaded",
        file_path="song.mp3",
        decision_provider="Jev",
        decision_probability=0.81,
        decision_runs=3,
        decision_threshold=0.60,
        decision_spread=0.22,
        decision_stable=False,
    )

    events.on_session_complete([result], 1.0)
    text = console.export_text()

    assert "81%" in text
    assert "spread 22%" in text


def test_summary_marks_result_that_needs_review():
    events, console = _events()
    from ytdl_core.result import DownloadResult

    result = DownloadResult(
        artist="Artist",
        song="Song",
        status="downloaded",
        file_path="song.mp3",
        decision_provider="Jev",
        decision_probability=0.63,
        decision_runs=1,
        decision_threshold=0.60,
        decision_needs_review=True,
    )

    events.on_session_complete([result], 1.0)
    text = console.export_text()

    assert "review" in text
