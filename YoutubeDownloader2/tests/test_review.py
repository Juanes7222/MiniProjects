"""Tests for ytdl_core.cli.review."""

from __future__ import annotations

import json
from pathlib import Path

from ytdl_core.cli.review import (
    _build_candidates,
    _fingerprint_text,
    _regenerate_not_verified,
)
from ytdl_core.state import save_state


def _write(path: Path, size: int = 1000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _state(*entries) -> dict:
    downloads = {}
    for artist, song, status, fp_verified, conf, file_path in entries:
        key = f"{artist}::{song}"
        downloads[key] = {
            "status": status,
            "file_path": str(file_path) if file_path else None,
            "fingerprint_verified": fp_verified,
            "fingerprint_confidence": conf,
            "fingerprint_label": "no match" if conf == 0 else "no match (other song)",
        }
    return {"downloads": downloads}


class TestBuildCandidates:
    def test_scopes_to_songs_and_excludes_verified(self, tmp_path):
        good = _write(tmp_path / "Barak" / "Dios Habla.mp3")
        bad = _write(tmp_path / "Barak" / "Adoracion.mp3")
        other = _write(tmp_path / "Other" / "Old Song.mp3")
        state = _state(
            ("Barak", "Dios Habla", "downloaded", True, 0.9, good),
            ("Barak", "Adoracion", "downloaded", False, 0.0, bad),
            ("Other", "Old Song", "downloaded", False, 0.0, other),
        )
        songs = {"Barak": ["Dios Habla", "Adoracion"]}
        cands = _build_candidates(state, tmp_path, "mp3", songs, False)
        keys = [c["key"] for c in cands]
        assert keys == ["Barak::Adoracion"]
        assert cands[0]["suspect"] is False

    def test_suspects_first(self, tmp_path):
        a = _write(tmp_path / "A" / "a.mp3")
        b = _write(tmp_path / "B" / "b.mp3")
        state = _state(
            ("A", "a", "downloaded", False, 0.0, a),
            ("B", "b", "downloaded", False, 0.97, b),
        )
        songs = {"A": ["a"], "B": ["b"]}
        cands = _build_candidates(state, tmp_path, "mp3", songs, False)
        assert cands[0]["key"] == "B::b"
        assert cands[0]["suspect"] is True

    def test_only_suspects_filters(self, tmp_path):
        a = _write(tmp_path / "A" / "a.mp3")
        b = _write(tmp_path / "B" / "b.mp3")
        state = _state(
            ("A", "a", "downloaded", False, 0.0, a),
            ("B", "b", "downloaded", False, 0.9, b),
        )
        songs = {"A": ["a"], "B": ["b"]}
        cands = _build_candidates(state, tmp_path, "mp3", songs, True)
        assert [c["key"] for c in cands] == ["B::b"]

    def test_skips_missing_files_and_deleted(self, tmp_path):
        missing = tmp_path / "Barak" / "gone.mp3"
        a = _write(tmp_path / "Barak" / "a.mp3")
        state = _state(
            ("Barak", "gone", "downloaded", False, 0.0, missing),
            ("Barak", "a", "downloaded", False, 0.0, a),
        )
        songs = {"Barak": ["gone", "a"]}
        cands = _build_candidates(state, tmp_path, "mp3", songs, False)
        assert [c["key"] for c in cands] == ["Barak::a"]

    def test_failed_with_file_is_candidate(self, tmp_path):
        f = _write(tmp_path / "G12" / "s.mp3")
        state = _state(
            ("G12", "s", "failed", False, 0.95, f),
        )
        songs = {"G12": ["s"]}
        cands = _build_candidates(state, tmp_path, "mp3", songs, False)
        assert len(cands) == 1
        assert cands[0]["suspect"] is True


class TestFingerprintText:
    def test_plain_label(self):
        assert _fingerprint_text({"fingerprint_label": "no match"}) == "no match"

    def test_label_with_confidence(self):
        assert _fingerprint_text(
            {"fingerprint_confidence": 0.94, "fingerprint_label": "verified 94%"}
        ) == "verified 94% (94%)"

    def test_no_conf_does_not_append(self):
        assert _fingerprint_text(
            {"fingerprint_confidence": 0, "fingerprint_label": "verified (stored)"}
        ) == "verified (stored)"

    def test_missing_label(self):
        assert _fingerprint_text({}) == "not attempted"


class TestRegenerateNotVerified:
    def test_writes_only_unverified_in_scope(self, tmp_path):
        good = _write(tmp_path / "Barak" / "a.mp3")
        bad = _write(tmp_path / "Barak" / "b.mp3")
        old = _write(tmp_path / "Other" / "c.mp3")
        state = _state(
            ("Barak", "a", "verified", True, 0.9, good),
            ("Barak", "b", "verified", False, 0.0, bad),
            ("Other", "c", "downloaded", False, 0.0, old),
        )
        out_dir = tmp_path / "downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_state(state, out_dir)
        songs = {"Barak": ["a", "b"]}
        _regenerate_not_verified(out_dir, songs)
        data = json.loads((out_dir / "not_verified.json").read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["song"] == "b"
        assert data[0]["suspect"] is False

    def test_marks_suspect(self, tmp_path):
        f = _write(tmp_path / "G12" / "s.mp3")
        state = _state(("G12", "s", "verified", False, 0.88, f))
        out_dir = tmp_path / "downloads"
        out_dir.mkdir()
        save_state(state, out_dir)
        _regenerate_not_verified(out_dir, {"G12": ["s"]})
        data = json.loads((out_dir / "not_verified.json").read_text(encoding="utf-8"))
        assert data[0]["suspect"] is True