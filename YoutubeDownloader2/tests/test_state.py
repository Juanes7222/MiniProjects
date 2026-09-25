from __future__ import annotations

from ytdl_core.state import load_state, save_state


def test_custom_state_filename_round_trip(tmp_path) -> None:
    state = {"downloads": {"Artist::Song": {"status": "downloaded"}}}

    save_state(state, tmp_path, "custom.json")

    assert load_state(tmp_path, "custom.json") == state
    assert not (tmp_path / ".download_state.json").exists()


def test_invalid_state_returns_empty_downloads(tmp_path) -> None:
    state_path = tmp_path / ".download_state.json"
    for content in ("not-json", '{"downloads": null}', '{"downloads": []}'):
        state_path.write_text(content, encoding="utf-8")
        assert load_state(tmp_path) == {"downloads": {}}


def test_malformed_entries_are_discarded(tmp_path) -> None:
    state_path = tmp_path / ".download_state.json"
    state_path.write_text(
        '{"downloads": {"Artist::Song": {"status": "downloaded"}, "Broken::Song": null}}',
        encoding="utf-8",
    )

    assert load_state(tmp_path) == {"downloads": {"Artist::Song": {"status": "downloaded"}}}
