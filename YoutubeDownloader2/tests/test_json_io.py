from __future__ import annotations

import json

from ytdl_core.json_io import write_json_atomic


def test_replaces_existing_json_atomically(tmp_path) -> None:
    destination = tmp_path / "nested" / "state.json"
    write_json_atomic(destination, {"version": 1})

    write_json_atomic(destination, {"version": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 2}
    assert list(destination.parent.glob("*.tmp")) == []


def test_preserves_unicode(tmp_path) -> None:
    destination = tmp_path / "state.json"

    write_json_atomic(destination, {"artist": "Artista"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"artist": "Artista"}
