from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_LIVE_TERMS = frozenset({"live", "en vivo", "concert", "concierto", "tour"})

# Hard rejects: a title containing any of these is never the recording we want.
# Matching is word-boundary based (see utils.find_forbidden_phrases), so bare
# "hora" / "version" were removed -- both are ordinary Spanish words that reject
# legitimate titles ("...durante las horas", "(original version, ...)"). The
# specific long-form and compound spellings are listed instead.
DEFAULT_FORBIDDEN_TERMS = frozenset(
    {
        "cover",
        "karaoke",
        "tribute",
        "reaction",
        "reacts",
        "reaccion",
        "remix",
        "mashup",
        "slowed",
        "reverb",
        "nightcore",
        "bootleg",
        "edit",
        "edits",
        "8d",
        "bass boosted",
        "extended",
        "loop",
        "looping",
        "10 hours",
        "1 hour",
        "2 hours",
        "24 hours",
        "1 hora",
        "2 horas",
        "hora completa",
        "horas continuas",
        "compilation",
        "instrumental",
        "acoustic",
        "acustica",
        "acústica",
        "version en vivo",
        "version live",
        "version remix",
        "version extended",
        "version extendida",
        "version acustica",
        "version acústica",
        "version instrumental",
        "version merengue",
        "radio edit",
        "fan made",
        "fanmade",
        "sped up",
        "speed up",
        "ultra slowed",
        "dj mix",
        "megamix",
        "mix",
        "full album",
        "album completo",
        "parodia",
        "parody",
        "reupload",
    }
)

# Soft terms: penalised, not rejected. A remaster from the artist's own Topic
# channel is a legitimate source; a remaster from a random reupload is not, so
# the penalty is waived for trusted/official channels (see scorer).
DEFAULT_SOFT_TERMS = frozenset(
    {
        "remaster",
        "remastered",
        "remasterizado",
        "remasterizacion",
        "remasterización",
        "reedicion",
        "reedición",
        "edicion remasterizada",
        "edición remasterizada",
        "reissue",
        "reedition",
        "version original",
        "original version",
    }
)


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
    DEFAULT_SOURCES: list[str] = field(default_factory=lambda: ["youtube", "soundcloud"])
    YOUTUBE_PLAYER_CLIENTS: list[str] = field(
        default_factory=lambda: ["android", "mweb", "web_embedded"]
    )
    SUPPORTED_FORMATS: list[str] = field(default_factory=lambda: ["mp3", "m4a", "opus", "mp4"])
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
    LIVE_PENALTY: int = -25
    COVER_KARAOKE_PENALTY: int = -50
    REACTION_REMIX_PENALTY: int = -50
    IDENTITY_OVERRIDE_THRESHOLD = 40
    JEV_DEFAULT_THRESHOLD: float = 0.60
    JEV_DEFAULT_RUNS: int = 1
    JEV_MAX_RUNS: int = 20

    # --- Search recall -----------------------------------------------------
    # The presentation budget (max_results) and the fetch budget are decoupled:
    # a small --max-results must not shrink how many candidates we pull from
    # each provider, or obscure catalogue tracks never surface at all.
    FETCH_MULTIPLIER: int = 4
    MIN_FETCH_PER_QUERY: int = 10

    # --- Learned channel trust --------------------------------------------
    TRUSTED_CHANNEL_BONUS: int = 40
    TRUSTED_CHANNEL_BONUS_SEEN: int = 12
    TRUSTED_ARTIST_CHANNEL_BONUS: int = 25
    TRUST_MAX_BONUS: int = 65
    TRUST_VERIFIED_MULTIPLIER: int = 3
    TRUST_STRONG_WEIGHT: int = 4
    TRUST_MEDIUM_WEIGHT: int = 2
    MAX_CHANNEL_SEARCHES: int = 2

    # --- Catalog / exact-match sources ------------------------------------
    CATALOG_SOURCE_BONUS: int = 60
    SOFT_TERM_PENALTY: int = -8

    # MusicBrainz reference is only trusted for scoring when the recording it
    # matched actually looks like the song we asked for.
    MB_REFERENCE_MIN_TITLE_MATCH: int = 70
    MB_REFERENCE_MIN_ARTIST_MATCH: int = 60

    LIVE_TERMS = DEFAULT_LIVE_TERMS
    FORBIDDEN_TERMS = DEFAULT_FORBIDDEN_TERMS
    SOFT_TERMS = DEFAULT_SOFT_TERMS
