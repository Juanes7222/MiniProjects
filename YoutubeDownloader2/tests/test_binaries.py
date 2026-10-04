"""Locating the bundled helper binaries.

Both of these were invisible to the code that used them: ``fpcalc.exe`` and
``aria2c.exe`` ship in the project root, and ``shutil.which`` only sees PATH.
"""
from __future__ import annotations

import os
from pathlib import Path

from ytdl_core import fingerprint, ytdlp_options
from ytdl_core.fingerprint import configure_fpcalc, find_fpcalc


def _clear_cache() -> None:
    fingerprint._fpcalc_command = None


def test_find_fpcalc_returns_an_absolute_path(monkeypatch, tmp_path: Path) -> None:
    """pyacoustid hands the string straight to subprocess.

    On Windows ``shutil.which`` answers with a *relative* name when the binary
    happens to sit in the current directory, which works from one directory and
    silently breaks from another.
    """
    monkeypatch.setattr(fingerprint, "shutil_which", lambda name: None)
    binary = tmp_path / ("fpcalc.exe" if os.name == "nt" else "fpcalc")
    binary.write_bytes(b"MZ" if os.name == "nt" else b"#!/bin/sh\n")
    monkeypatch.setattr(fingerprint, "Path", _RootedAt(tmp_path))

    found = find_fpcalc()
    assert found is not None
    assert os.path.isabs(found)


def test_find_fpcalc_is_none_when_absent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fingerprint, "shutil_which", lambda name: None)
    monkeypatch.setattr(fingerprint, "Path", _RootedAt(tmp_path))
    assert find_fpcalc() is None


def test_configure_fpcalc_points_pyacoustid_at_the_binary(monkeypatch, tmp_path: Path) -> None:
    """Finding the binary is only half the job.

    pyacoustid chooses its backend at import time from whether the ``chromaprint``
    C extension loaded. When it did not -- which is the case here -- it falls back
    to ``audioread``, a pure-Python FFT, unless the caller both names the command
    and asks for it.
    """
    import acoustid

    _clear_cache()
    monkeypatch.setattr(fingerprint, "shutil_which", lambda name: None)
    binary = tmp_path / ("fpcalc.exe" if os.name == "nt" else "fpcalc")
    binary.write_bytes(b"MZ" if os.name == "nt" else b"#!/bin/sh\n")
    monkeypatch.setattr(fingerprint, "Path", _RootedAt(tmp_path))

    original = acoustid.FPCALC_COMMAND
    try:
        found = configure_fpcalc()
        assert found
        assert acoustid.FPCALC_COMMAND == found
    finally:
        acoustid.FPCALC_COMMAND = original
        _clear_cache()


def test_configure_fpcalc_caches(monkeypatch) -> None:
    _clear_cache()
    calls = {"n": 0}

    def fake_find():
        calls["n"] += 1
        return None

    monkeypatch.setattr(fingerprint, "find_fpcalc", fake_find)
    configure_fpcalc()
    configure_fpcalc()
    configure_fpcalc()
    assert calls["n"] == 1
    _clear_cache()


def test_find_aria2c_is_absolute_or_none(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ytdlp_options.shutil, "which", lambda name: None)
    monkeypatch.setattr(ytdlp_options, "Path", _RootedAt(tmp_path))
    assert ytdlp_options.find_aria2c() is None

    binary = tmp_path / ("aria2c.exe" if os.name == "nt" else "aria2c")
    binary.write_bytes(b"MZ" if os.name == "nt" else b"#!/bin/sh\n")
    found = ytdlp_options.find_aria2c()
    assert found is not None and os.path.isabs(found)


class _RootedAt:
    """Makes ``Path(__file__).resolve().parent`` land inside *root*.

    The finders walk up from the module's own location, so redirecting ``Path``
    is what lets a test place a binary where the lookup will actually look.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._real = Path

    def __call__(self, value=os.curdir):
        path = self._real(value)
        if isinstance(value, str) and value.endswith(".py"):
            return self._root / "pkg" / "module.py"
        return path

    def __getattr__(self, name):
        return getattr(self._real, name)