"""Change detection: the hash algorithm switch and its backwards compatibility."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ytdl_core.utils import (
    HASH_ALGORITHM,
    compute_file_hash,
    compute_md5,
    file_matches_hash,
    parse_stored_hash,
)


def _write(path: Path, size: int = 200_000) -> Path:
    path.write_bytes(b"\xa5" * size)
    return path


def test_the_digest_names_its_algorithm(tmp_path: Path) -> None:
    stored = compute_file_hash(_write(tmp_path / "a.mp3"))
    assert stored.startswith(f"{HASH_ALGORITHM}:")
    assert parse_stored_hash(stored)[0] == HASH_ALGORITHM


def test_an_untagged_digest_is_read_as_md5(tmp_path: Path) -> None:
    """State files written before the switch hold bare MD5 hex."""
    algorithm, value = parse_stored_hash("d41d8cd98f00b204e9800998ecf8427e")
    assert algorithm == "md5"
    assert value == "d41d8cd98f00b204e9800998ecf8427e"


def test_a_library_written_by_an_older_version_still_verifies(tmp_path: Path) -> None:
    """The whole point of the tag.

    Comparing the recomputed digest to the stored string would report every
    already-downloaded file as changed and re-download the library, because the
    new digest carries a prefix the old one does not.
    """
    path = _write(tmp_path / "old.mp3")
    legacy = compute_md5(path)
    assert ":" not in legacy
    assert file_matches_hash(path, legacy) is True


def test_a_digest_written_by_this_version_verifies(tmp_path: Path) -> None:
    path = _write(tmp_path / "new.mp3")
    assert file_matches_hash(path, compute_file_hash(path)) is True


def test_modified_content_does_not_verify(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.mp3")
    stored = compute_file_hash(path)
    path.write_bytes(b"\x5a" * 200_000)
    assert file_matches_hash(path, stored) is False


def test_a_missing_digest_never_verifies(tmp_path: Path) -> None:
    """No recorded digest means "unverified", never "verified"."""
    assert file_matches_hash(_write(tmp_path / "d.mp3"), None) is False
    assert file_matches_hash(_write(tmp_path / "e.mp3"), "") is False


def test_the_digest_is_the_real_one(tmp_path: Path) -> None:
    path = _write(tmp_path / "f.bin", 1000)
    expected = hashlib.blake2b(path.read_bytes()).hexdigest()
    assert compute_file_hash(path).endswith(expected)


def test_blake2b_is_not_md5(tmp_path: Path) -> None:
    """Otherwise the switch bought nothing."""
    path = _write(tmp_path / "g.bin")
    assert compute_file_hash(path) != f"md5:{compute_md5(path)}"