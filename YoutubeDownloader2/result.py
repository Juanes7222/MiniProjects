"""
DownloadResult dataclass — the single return type of the entire pipeline.
All consumers (CLI, library users, tests) work with this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional


@dataclass
class DownloadResult:
    artist: str
    song: str

    status: Literal["downloaded", "failed", "skipped"] = "failed"
    reason: Optional[str] = None

    source: Optional[str] = None
    url: Optional[str] = None
    matched_title: Optional[str] = None
    fuzzy_score: int = 0

    composite_score: int = 0
    score_breakdown: dict[str, int] = field(default_factory=dict)

    duration_seconds: int = 0
    file_path: Optional[Path] = None
    file_size_bytes: int = 0
    md5: Optional[str] = None

    musicbrainz_enriched: bool = False
    album: Optional[str] = None
    year: Optional[str] = None
    genre: Optional[str] = None

    fingerprint_verified: bool = False
    fingerprint_confidence: float = 0.0
    fingerprint_matched_title: Optional[str] = None
    silence_ratio: float = 0.0
    duration_verified: bool = True

    def to_dict(self) -> dict:
        """Serialise to a plain dict (suitable for JSON / CSV export)."""
        d = asdict(self)
        if self.file_path is not None:
            d["file_path"] = str(self.file_path)
        return d

    @property
    def ok(self) -> bool:
        return self.status == "downloaded"