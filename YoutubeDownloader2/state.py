"""
Persistent state cache: load, save (atomically), and update download records.

State file structure:
{
  "downloads": {
    "{artist}::{song}": {
      "status": "downloaded" | "failed" | "skipped",
      "url": "https://...",
      "file_path": "...",
      "md5": "...",
      "timestamp": "ISO8601"
    }
  }
}
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from config import Config

_CFG = Config()


def load_state(output_dir: Path) -> dict:
    """Load the state JSON from *output_dir*, or return an empty skeleton."""
    path = output_dir / _CFG.STATE_FILE
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {"downloads": {}}
    return {"downloads": {}}


def save_state(state: dict, output_dir: Path) -> None:
    """Write *state* to disk atomically (temp-file + rename)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / _CFG.STATE_FILE
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
        shutil.move(tmp, str(dest))
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise