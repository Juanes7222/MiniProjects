from __future__ import annotations

from rich.console import Console

from ytdl_core.cli.rich_ui import RichEvents
from ytdl_core.config import Config


def test_candidate_table_separates_jev_and_heuristic_scores():
    console = Console(record=True, width=220, color_system=None)
    events = RichEvents(console, score_threshold=25, config=Config(), jev_threshold=0.75)
    entry = {
        "title": "Artist - Song",
        "channel": "Official Artist",
        "duration": 200,
        "_jev_probability": 0.74,
        "_heuristic_score": 140,
        "_score_breakdown": {"base_match": 140, "jev_probability": 74},
    }

    events.on_candidates_scored("Artist", "Song", [(entry, 74, entry["_score_breakdown"])])
    text = console.export_text()

    assert "Jev" in text
    assert "Heur." in text
    assert "74%" in text
    assert "140" in text
    assert ">1" not in text
