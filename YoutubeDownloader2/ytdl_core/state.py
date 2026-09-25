from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config
from .json_io import write_json_atomic

_DEFAULT_STATE_FILENAME = Config().STATE_FILE


def load_state(
    output_dir: Path,
    state_filename: str | None = None,
) -> dict[str, Any]:
    path = Path(output_dir) / (state_filename or _DEFAULT_STATE_FILENAME)
    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"downloads": {}}
    if not isinstance(state, dict) or not isinstance(state.get("downloads"), dict):
        return {"downloads": {}}
    state["downloads"] = {
        key: entry for key, entry in state["downloads"].items() if isinstance(entry, dict)
    }
    return state


def save_state(
    state: dict[str, Any],
    output_dir: Path,
    state_filename: str | None = None,
) -> None:
    path = Path(output_dir) / (state_filename or _DEFAULT_STATE_FILENAME)
    write_json_atomic(path, state)
