"""
Configuration dataclass for YT Music Downloader v2.0.
All tunable defaults live here so they are easy to find and override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    MAX_DURATION_SECONDS: int = 1080
    MIN_DURATION_SECONDS: int = 60
    DEFAULT_FORMAT: str = "mp3"
    DEFAULT_QUALITY: str = "192"
    DEFAULT_OUTPUT_DIR: str = "./downloads"
    DEFAULT_MAX_RESULTS: int = 5
    DEFAULT_WORKERS: int = 1
    MAX_WORKERS: int = os.cpu_count() or 4
    DEFAULT_DELAY_MIN: float = 2.0
    DEFAULT_DELAY_MAX: float = 5.0
    DEFAULT_FUZZY_THRESHOLD: int = 65
    DEFAULT_SOURCES: List[str] = field(default_factory=lambda: ["youtube", "soundcloud"])
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: ["mp3", "m4a", "opus"])
    MUSICBRAINZ_APP: str = "YTMusicDownloader/2.0"
    STATE_FILE: str = ".download_state.json"
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE: float = 2.0

    PARTIAL_DOWNLOAD_SECONDS: int = 90
    FINGERPRINT_MIN_CONFIDENCE: float = 0.60
    SCORE_THRESHOLD_SKIP_FINGERPRINT: int = 70
    SCORE_THRESHOLD_REJECT: int = 25
    SILENCE_THRESHOLD_DB: int = -50
    SILENCE_MIN_DURATION_MS: int = 3000
    EXCESSIVE_SILENCE_RATIO: float = 0.30
    TOPIC_CHANNEL_BONUS: int = 50
    VEVO_CHANNEL_BONUS: int = 30
    OFFICIAL_AUDIO_BONUS: int = 20
    HIGH_FUZZY_BONUS: int = 20
    DURATION_MATCH_BONUS: int = 25
    LIVE_PENALTY: int = -40
    COVER_KARAOKE_PENALTY: int = -50
    REACTION_REMIX_PENALTY: int = -50