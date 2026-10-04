from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_LIVE_TERMS = frozenset({"live", "en vivo", "concert", "concierto", "tour"})


def _logical_cpus() -> int:
    return os.cpu_count() or 4

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
    # The workload is overwhelmingly I/O-bound (yt-dlp extractions, partial
    # downloads, HTTP metadata calls), so the worker ceiling tracks logical
    # CPUs with headroom rather than the CPU count itself. A previous value of
    # ``os.cpu_count()`` capped an I/O-bound pool at the number of cores and
    # silently throttled a 12-thread machine to 12 in-flight songs.
    DEFAULT_WORKERS: int = min(8, max(2, _logical_cpus()))
    MAX_WORKERS: int = max(16, _logical_cpus() * 4)
    DEFAULT_DELAY_MIN: float = 0.0
    DEFAULT_DELAY_MAX: float = 0.0
    DEFAULT_FUZZY_THRESHOLD: int = 65
    DEFAULT_SOURCES: list[str] = field(default_factory=lambda: ["youtube", "soundcloud"])
    YOUTUBE_PLAYER_CLIENTS: list[str] = field(
        default_factory=lambda: ["android", "mweb", "web_embedded"]
    )
    SUPPORTED_FORMATS: list[str] = field(default_factory=lambda: ["mp3", "m4a", "opus", "mp4"])
    # MusicBrainz requires a User-Agent identifying the app and its version, and
    # it is used in two shapes: the library's own (name, version) pair and the
    # HTTP header for the iTunes/cover-art requests. One field, both call sites --
    # previously four hardcoded copies of the same two strings.
    MUSICBRAINZ_APP: str = "YTMusicDownloader/2.0"
    STATE_FILE: str = ".download_state.json"
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE: float = 2.0
    RETRY_BACKOFF_CAP: float = 30.0

    # --- Remote service budgets ---------------------------------------------
    # These are per-process ceilings published by the services themselves. They
    # are enforced by token buckets that block only when the caller would
    # actually exceed the budget, replacing the unconditional sleeps that used
    # to run once per song regardless of how much of the budget was left.
    ACOUSTID_RATE_PER_SECOND: float = 3.0
    # Burst of 1 keeps the "no more than 3 requests per second" rule true even
    # when measured over a sub-second window.
    ACOUSTID_BURST: float = 1.0
    MUSICBRAINZ_RATE_PER_SECOND: float = 1.0
    MUSICBRAINZ_BURST: float = 1.0
    ITUNES_RATE_PER_SECOND: float = 5.0
    ITUNES_BURST: float = 5.0
    YOUTUBE_RATE_PER_SECOND: float = 8.0
    YOUTUBE_BURST: float = 8.0

    # --- Stage concurrency ---------------------------------------------------
    # Each pipeline stage is sized against the resource it actually contends
    # for. Search, download and metadata are network-bound and want more threads
    # than cores; the post-download checks decode audio and want about one
    # worker per core; the decision model is latency-bound and needs few.
    #
    # There is deliberately no separate decide pool. The decision model is asked
    # inside the search stage, and what bounds it is DECISION_MAX_IN_FLIGHT (a
    # gate on the shared single-device server), not a stage size: the model has
    # one GPU whether eight threads call it or one. A fifth stage would add a
    # queue and a set of workers without changing that.
    PIPELINE_ENABLED: bool = True
    PIPELINE_QUEUE_DEPTH: int = 64
    SEARCH_WORKERS: int = 0  # 0 = auto
    VERIFY_WORKERS: int = 0
    DOWNLOAD_WORKERS: int = 0
    POST_WORKERS: int = 0
    # Concurrency for the 90-second partial downloads. Sized so that
    # FP_CONCURRENCY / partial_latency comfortably exceeds ACOUSTID_RATE_PER_SECOND:
    # otherwise the AcoustID budget is the ceiling and the semaphore is what
    # actually throttles the run.
    FP_CONCURRENCY: int = 8
    # How many alternate candidates to fingerprint speculatively, in parallel,
    # when the winner does not match. Serialised this was N round trips.
    FP_ALTERNATE_FANOUT: int = 3

    # --- Decision model ------------------------------------------------------
    # A decision server is a shared single-device resource, so these bound how
    # hard it is leaned on.
    #
    # How many evaluations may be in flight at once. This used to be 1, on the
    # reasoning that one at a time makes a slow server predictable. That is true
    # of a server that does not batch, and it is the wrong assumption for one
    # that does: kev.serve drains up to MAX_BATCH=64 queued requests into a
    # single model pass, so N requests in flight finish in roughly one pass
    # instead of N. Holding it at 1 threw that away and capped a whole batch at
    # one evaluation per round trip.
    #
    # Not free, though: each queued request holds its state in the server's cache
    # and its rows in the batching buffers, so this is bounded by VRAM as much as
    # by throughput. Six is a measured-middle setting for a 16 GB card running a
    # 4B checkpoint (~14 GB resident); raise it on a bigger card, lower it to 1 to
    # make a slow server strictly serial.
    DECISION_MAX_IN_FLIGHT: int = 6
    DECISION_TIMEOUT_SECONDS: int = 60
    # Consecutive failures after which the provider is abandoned for the run.
    # Low on purpose: the heuristic is a complete ranker, so the model is a
    # second opinion, and there is no reason to keep paying for one.
    DECISION_FAILURE_THRESHOLD: int = 2

    # --- Search fan-out and cache -------------------------------------------
    SEARCH_VARIANT_FANOUT: int = 3
    SEARCH_CACHE_TTL: float = 7 * 24 * 3600.0
    SEARCH_CACHE_NEGATIVE_TTL: float = 6 * 3600.0
    CATALOG_CACHE_TTL: float = 30 * 24 * 3600.0
    COVER_CACHE_TTL: float = 30 * 24 * 3600.0
    FINGERPRINT_CACHE_TTL: float = 30 * 24 * 3600.0
    CACHE_ENABLED: bool = True

    # --- yt-dlp request shaping ---------------------------------------------
    # yt-dlp already sleeps between extraction requests, with jitter, and only
    # when it needs to. Letting it do that is strictly better than a fixed
    # per-song sleep, and it applies per request rather than per song.
    SLEEP_INTERVAL_REQUESTS: float = 0.0
    MAX_SLEEP_INTERVAL: float = 5.0
    # Two retry layers used to stack: yt-dlp's internal 10 and the app's 3, for
    # up to 30 attempts on one URL. The app loop now owns retries, so yt-dlp's
    # internal count is kept short and its sleeps bounded and jittered.
    YTDLP_SCAN_RETRIES: int = 2
    YTDLP_DOWNLOAD_RETRIES: int = 4
    YTDLP_RETRY_SLEEP: str = "http:exp=1:8"
    SOCKET_TIMEOUT: int = 30
    # DASH audio arrives as many small fragments and yt-dlp fetches them one at a
    # time by default, so a single song used exactly one connection however much
    # bandwidth was idle. Four is enough to saturate a normal connection without
    # tripping per-host rate limits by opening a pile of sockets.
    FRAGMENT_CONCURRENCY: int = 4
    # Hand the transfer to aria2c when the binary is available. It is a native
    # multi-connection downloader and the executable already ships in the repo.
    # Opt-in, because an external downloader changes how ranges and retries
    # behave and that is not something to change silently.
    USE_ARIA2C: bool = False

    # --- State persistence ---------------------------------------------------
    # The state file was rewritten in full, with an fsync, on every persist --
    # including every failure path -- while holding the lock that all other
    # workers need in order to make progress. Writes are now coalesced onto a
    # single background writer.
    STATE_FLUSH_INTERVAL: float = 1.0
    STATE_FLUSH_BATCH: int = 25

    PARTIAL_DOWNLOAD_SECONDS: int = 90
    FINGERPRINT_MIN_CONFIDENCE: float = 0.60
    SCORE_THRESHOLD_SKIP_FINGERPRINT: int = 70
    SCORE_THRESHOLD_REJECT: int = 25
    SILENCE_THRESHOLD_DB: int = -50
    SILENCE_MIN_DURATION_MS: int = 3000
    # Rate the silence check decimates to before running the filter. A -50 dB
    # threshold is decided by a coarse envelope; 8 kHz mono keeps every relevant
    # band and cuts the filter's work by roughly an order of magnitude on 44.1 kHz
    # stereo. Raise it if quiet-but-not-silent passages start being counted.
    SILENCE_DETECT_SAMPLE_RATE: int = 8000
    EXCESSIVE_SILENCE_RATIO: float = 0.30
    TOPIC_CHANNEL_BONUS: int = 50
    VEVO_CHANNEL_BONUS: int = 30
    OFFICIAL_AUDIO_BONUS: int = 20
    DURATION_MATCH_BONUS: int = 25
    LIVE_PENALTY: int = -25
    # Three scoring weights that used to live here were removed rather than
    # wired, and the reasons are worth keeping:
    #
    # * COVER_KARAOKE_PENALTY and REACTION_REMIX_PENALTY (-50 each) predate the
    #   forbidden-term **hard reject** by about a month. "cover", "karaoke",
    #   "reaction", "remix" and "mashup" are all in FORBIDDEN_TERMS, and a title
    #   containing one is now discarded outright at -9999. The hard reject
    #   replaced them with something strictly stronger, so wiring the penalties
    #   back would have changed nothing except making the config look honest.
    #
    # * HIGH_FUZZY_BONUS (+20 for a near-exact fuzzy match) is a *discriminator*,
    #   not a gate: the scorer already ranks on `song_match`, so a bonus for a
    #   high one only separates candidates that are otherwise close. Re-adding it
    #   would change which song gets downloaded across a whole library, and that
    #   is a decision to make against real data rather than in passing. If it is
    #   wanted back, add it to the generic path next to `base_score` and measure
    #   the selections before and after.
    #
    # None of the three was reachable, so a field that looks like a knob and does
    # nothing is worse than no field: someone tunes it, sees no effect, and loses
    # trust in the rest of the config.
    IDENTITY_OVERRIDE_THRESHOLD = 40
    JEV_DEFAULT_THRESHOLD: float = 0.60
    JEV_DEFAULT_RUNS: int = 1
    JEV_MAX_RUNS: int = 20

    # --- Decision model ------------------------------------------------------
    # Gates veto a candidate outright when it falls under the floor; the
    # remaining dimensions are weighted into a ranking score.
    DECISION_GATE_FLOOR: float = 0.50
    DECISION_STABLE_SPREAD: float = 0.15
    # How many candidates go to the model per song. Search returns far more than
    # this, and every extra candidate multiplies the request size across all
    # dimensions; the ones past the cap cannot change which candidate wins.
    DECISION_MAX_CANDIDATES: int = 4
    # The real budget. Cost is the *question count*, which grows as
    # candidates x dimensions, so bounding candidates alone does not actually
    # bound anything: at 12 candidates plus 4 of headroom this reached 97
    # questions per song, and at --kev-runs 2 that is 194 generations before a
    # single download starts. Trimming to a question budget keeps the cost
    # predictable no matter how the dimension list grows later.
    DECISION_MAX_QUESTIONS: int = 32
    DECISION_HEADROOM: int = 1
    # Candidates whose per-dimension answers moved more than this across runs are
    # unstable, and are surfaced for review instead of downloaded silently.
    DECISION_REVIEW_MARGIN: float = 0.10
    # Wall-clock budget for the startup latency probe: above this, one real
    # evaluation is reported as too slow for a batch, with the remedy.
    DECISION_PROBE_BUDGET_SECONDS: float = 20.0

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
