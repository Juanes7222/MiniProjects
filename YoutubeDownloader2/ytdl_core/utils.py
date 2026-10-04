"""
Shared utility helpers: filename sanitisation, formatting, MD5, phrase matching.
"""

from __future__ import annotations

import functools
import hashlib
import random
import re
import shutil
import sys
import time
import unicodedata
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


def sanitize_filename(name: str) -> str:
    """Strip illegal filesystem characters, collapse whitespace, cap at 200 chars."""
    name = re.sub(r'[/\\:*?"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:200]


def migrate_legacy_audio_path(path: Path) -> Path:
    if not path.suffix:
        return path
    legacy_path = path.with_name(f"{path.name}{path.suffix}")
    if not legacy_path.is_file() or path.exists():
        return path
    try:
        legacy_path.replace(path)
    except OSError:
        return legacy_path
    return path


def format_duration(seconds: int) -> str:
    """Return MM:SS string for a given number of seconds."""
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def format_size(nbytes: int) -> str:
    """Return a human-readable file-size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


#: Digest used to detect that a file on disk is not the file we wrote.
#:
#: blake2b rather than md5 because this is change detection, not security: no
#: current x86 has MD5 hardware acceleration, while blake2b is consistently
#: faster in software, and the whole file is re-read on the skip-existing path and
#: twice more during a verify pass.
HASH_ALGORITHM = "blake2b"

#: Read size for hashing. The old 64 KB meant ~250 syscalls for a 16 MB track,
#: which is call overhead rather than I/O.
HASH_CHUNK_BYTES = 1 << 20


def compute_file_hash(path: Path, algorithm: str = HASH_ALGORITHM) -> str:
    """Digest *path*, tagged with the algorithm that produced it.

    The tag is what makes the switch safe. Deltas recorded by older versions are
    bare MD5 hex with nothing to distinguish them from a digest of the same
    length, so an untagged value is read as MD5 and re-checked with MD5; a tagged
    value is re-checked with the algorithm it names. Without that, switching the
    algorithm would silently mark every already-downloaded file as changed and
    re-download the whole library.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"


def parse_stored_hash(stored: str | None) -> tuple[str, str]:
    """Split a stored digest into ``(algorithm, hex)``. Untagged means MD5."""
    if not stored:
        return HASH_ALGORITHM, ""
    algorithm, separator, value = stored.partition(":")
    if separator and algorithm in hashlib.algorithms_available and value:
        return algorithm, value
    return "md5", stored


def compute_md5(path: Path) -> str:
    """Compute the plain MD5 hex-digest of a file. Legacy only.

    Kept for re-checking digests recorded before the algorithm switch; new
    records go through :func:`compute_file_hash`.
    """
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches_hash(path: Path, stored: str | None) -> bool:
    """Whether *path* still hashes to *stored*, using the algorithm *stored* names.

    Compares the hex rather than the stored string: a legacy entry is bare MD5
    while :func:`compute_file_hash` always returns a tagged value, so comparing
    the two whole strings would report every pre-existing library as changed.
    """
    algorithm, expected = parse_stored_hash(stored)
    if not expected:
        return False
    try:
        _tagged, actual = compute_file_hash(path, algorithm).split(":", 1)
    except (OSError, ValueError):
        return False
    return actual == expected


def apply_delay(min_s: float, max_s: float) -> None:
    """Sleep for a uniformly-random duration in [min_s, max_s]."""
    time.sleep(random.uniform(min_s, max_s))


def check_ffmpeg(console: "Console") -> None:
    """Exit with an informative Rich panel if ffmpeg is not on PATH."""
    from rich.panel import Panel  # local import avoids circular deps at module level

    if shutil.which("ffmpeg") is None:
        console.print(
            Panel(
                "[red]ffmpeg executable not found on PATH.[/red]\n\n"
                "Install instructions:\n"
                "  [bold]Ubuntu/Debian:[/bold]  sudo apt install ffmpeg\n"
                "  [bold]macOS (brew):[/bold]   brew install ffmpeg\n"
                "  [bold]Windows:[/bold]        https://ffmpeg.org/download.html\n"
                "  [bold]Arch Linux:[/bold]     sudo pacman -S ffmpeg",
                title="[bold red] Missing Dependency: ffmpeg[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)


@functools.lru_cache(maxsize=8192)
def normalize_title(title: str) -> str:
    if not title:
        return ""
    nfkd = unicodedata.normalize("NFKD", title)
    ascii_title = nfkd.encode("ascii", "ignore").decode("ascii")

    cleaned = re.sub(r"[\(\)\[\]\{\}\-\|\\\/,.:;!?_~*+^=]", " ", ascii_title)

    return " ".join(cleaned.lower().split()).strip()


_MATCHING_NOISE_PATTERN = re.compile(
    r"\b(official\s*(audio|video|music\s*video|lyric\s*video)?|hd|hq|4k|remastered|visualizer)\b",
    re.IGNORECASE,
)


def remove_matching_noise(text: str) -> str:
    return _MATCHING_NOISE_PATTERN.sub("", text).strip()


_FEAT_PATTERN = re.compile(
    r"\s*(feat\.?|ft\.?|with|&|\+)\s+.+$",
    re.IGNORECASE,
)


def strip_featuring(text: str) -> str:
    return _FEAT_PATTERN.sub("", text).strip()


class PhraseMatcher:
    """Matches a fixed phrase set against arbitrary text.

    The phrase sets here are module-level constants -- 55 forbidden terms, 14 soft
    terms, 5 live terms -- but they were being normalised *per candidate, per
    call*: ranking forty candidates re-normalised every constant five times over.
    ``normalize_title`` is not cheap (NFKD decomposition, a regex pass, a split),
    so that was several thousand redundant normalisations per song.

    Normalising the constants once and collapsing the whole set into a single
    alternation turns the check into one C-level scan of the text.

    Matching is word-boundary based, and deliberately so: bare "hora" or
    "version" would reject legitimate Spanish titles.
    """

    __slots__ = ("_mapping", "_pattern")

    def __init__(self, phrases: Collection[str]) -> None:
        mapping: dict[str, str] = {}
        for phrase in phrases:
            normalized = normalize_title(phrase)
            if normalized:
                mapping.setdefault(normalized, phrase)
        self._mapping = mapping
        if mapping:
            # " a " and " b c " as one alternation over a space-padded haystack is
            # exactly the old substring-per-phrase behaviour, in a single pass.
            alternatives = "|".join(re.escape(f" {normalized} ") for normalized in mapping)
            self._pattern: re.Pattern[str] | None = re.compile(alternatives)
        else:
            self._pattern = None

    def find(self, text: str) -> set[str]:
        """Return the original phrases of this set that *text* contains."""
        if self._pattern is None:
            return set()
        padded = f" {normalize_title(text)} "
        found: set[str] = set()
        for match in self._pattern.finditer(padded):
            original = self._mapping.get(match.group(0).strip())
            if original is not None:
                found.add(original)
        return found

    def __bool__(self) -> bool:
        return self._pattern is not None


@functools.lru_cache(maxsize=64)
def phrase_matcher(phrases: frozenset[str]) -> PhraseMatcher:
    """A :class:`PhraseMatcher` for *phrases*, built once per distinct set.

    The arguments are the frozen term sets from ``config``, so the cache has one
    entry per set rather than one per candidate.
    """
    return PhraseMatcher(phrases)


def find_forbidden_phrases(text: str, forbidden: Collection[str]) -> set[str]:
    """Which of *forbidden* appear in *text*, as whole words.

    Prefer :func:`phrase_matcher` on a frozen set when matching repeatedly; this
    form rebuilds the matcher, which is fine for a one-off call.
    """
    return phrase_matcher(frozenset(forbidden)).find(text)
