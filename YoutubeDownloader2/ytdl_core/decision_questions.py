"""Question design for System One decision models (Jev / Kev).

Single source of truth for how a candidate is judged by a decision model.

Design rules follow the System One contract:

* The ``state`` holds only the **target** -- what we are looking for. Candidates
  never go into the state, so no candidate is nudged by its competitors and the
  same state is reusable across every question.
* Each **candidate** is passed inside its own structured ``instructions`` object,
  so a question reads "is *this record* the requested song" instead of "is
  candidate N, out of the ones you can see, the requested song".
* One **Noul per dimension**. A Noul answers exactly one yes/no proposition, so
  a judgment with several conditions is decomposed into several questions and
  the combination happens in Python where the weights are inspectable.
* Every Noul carries ``criteria`` describing what a yes and a no mean. The
  boundary between them is where these models lose accuracy; ``criteria`` is the
  typed mechanism for pinning it down.

Providers only implement transport: they ship the payload this module builds and
read the typed answers back. They must not restate any of this in their own
prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import normalize_title, remove_matching_noise, strip_featuring

# Bumped whenever the question set, the state shape or the scoring math changes,
# so stored decisions can be invalidated instead of silently compared to old ones.
#
# 3 added the derived `provenance` field and the deterministic shield. Both change
# what a question can see and what a gate is allowed to veto, so a score gathered
# under v2 is not comparable to one gathered under v3.
QUESTION_CONTRACT_VERSION = 3

# Short description cap. Long descriptions burn tokens and dilute the fields the
# model actually reasons about; the head of a YouTube description carries the
# "official audio" / "lyrics" signals we care about.
#
# 500 characters is roughly the first three lines, which is where those signals
# live ("Provided to YouTube by...", "Official Audio", "Lyrics", the label's own
# copyright line). It was 1200, which cost 1200 characters per candidate on the
# only dimension that reads it -- about a fifth of the whole request -- to
# describe the middle of boilerplate no criterion can turn on.
DESCRIPTION_LIMIT = 500

# How many candidates to send for evaluation. The heuristic has already
# hard-rejected the obvious rejects by this point; the model exists to arbitrate
# the plausible remainder, not to re-read the whole search pool.
MAX_EVALUATED_CANDIDATES = 12

# The heuristic's hard-reject floor (see scorer.score_youtube_result). A candidate
# sitting here was rejected by a deterministic word match on its title.
HARD_REJECT_SCORE = -9999

# Extra candidates evaluated beyond the cap, so near-ties at the cut are not
# decided by list order alone.
EVALUATION_HEADROOM = 4

REFERENCE_KEYS = ("album", "year", "genre", "track_num", "mb_id", "release_id")


@dataclass(frozen=True)
class Dimension:
    """One atomic judgment about a candidate.

    ``key`` is the dimension name reported in the score breakdown.
    ``gate`` dimensions veto a candidate outright; non-gate dimensions only rank.
    ``weight`` is the share of the ranking score; ignored for gates.
    ``fields`` is the candidate state this question actually reads.

    ``fields`` exists because of how the model is served. Every question is
    prefilled as its own sequence -- the shared state plus that question -- so a
    field repeated into six questions is paid for six times. Measured with five
    candidates, sending the whole candidate dict to every dimension produced 31
    questions and 31,389 characters of prefill; trimming each question to the
    fields it can reason about roughly halves that. The state itself is ~96
    characters, so essentially all of the GPU cost is the questions.

    The rule applied when assigning fields: include a field if the question's own
    ``criteria`` text could plausibly mention it. ``key``, ``title`` and
    ``channel`` are in every set -- every dimension here is a judgment about what
    a particular upload is. ``view_count`` is in none: no criterion refers to
    popularity, and it is one of the longest fields.
    """

    key: str
    gate: bool
    weight: float
    instructions: dict[str, str]
    criteria: dict[str, str]
    fields: tuple[str, ...] = ()

    def question(self, candidate: dict[str, Any], noul_type: str) -> dict[str, Any]:
        return {
            "type": noul_type,
            "instructions": {
                "candidate": candidate,
                **self.instructions,
            },
            "criteria": self.criteria,
        }


@dataclass(frozen=True)
class Dialect:
    """How a provider's API names the same three primitives.

    The judgments are identical; only the wire vocabulary differs. TypeSafe's own
    endpoint calls the yes/no type ``noul``; Vercel AI Gateway calls the same
    thing ``boolean`` and returns the value under ``probability`` instead of
    ``noul``. Providers declare their dialect and :mod:`ytdl_core.jev` accepts
    either answer key, so one contract serves both.
    """

    noul: str
    choice: str = "choice"
    score: str = "score"


TYPESAFE_DIALECT = Dialect(noul="noul")
GATEWAY_DIALECT = Dialect(noul="boolean")


def _noul(
    key: str,
    instructions: dict[str, str],
    criteria: dict[str, str],
    fields: tuple[str, ...],
) -> Dimension:
    return Dimension(
        key=key,
        gate=False,
        weight=0.0,
        instructions=instructions,
        criteria=criteria,
        fields=fields,
    )


# Candidate fields, and which questions may read them.
#
# ``IDENTITY_FIELDS`` is the floor every question gets: these three identify the
# upload being judged. The rest are attached only where the question's own
# criteria could plausibly turn on them -- so a question about whether the audio
# is clean does not carry the view count, and a question about whether the album
# matches the reference does not carry the description.
IDENTITY_FIELDS = ("key", "title", "channel")
# What `song` gets instead. `key` is not optional -- it is how an answer is
# attributed back to its candidate -- and the rest is the title, the credited
# artists and the length.
#
# No `channel`, deliberately. The channel is where the *artist* lives: an upload on
# "Marcos Witt Oficial" is strong evidence of who is playing and none at all of
# which song. Shipping it to the song question is how the old compound `identity`
# gate ended up answering 56% for a different song -- the artist half was visibly
# true and it propped the song half up.
_SONG_FIELDS = ("key", "title", "artists", "duration_seconds")
_WHO_FIELDS = IDENTITY_FIELDS + ("artists",)
_UPLOAD_FIELDS = IDENTITY_FIELDS + ("artists", "source")
_RECORDING_FIELDS = IDENTITY_FIELDS + ("duration_seconds", "upload_date")
_CATALOGUE_FIELDS = IDENTITY_FIELDS + ("album", "year", "genre", "upload_date")


# --- Gates ------------------------------------------------------------------
# These answer "is this even the right recording?". Below GATE_FLOOR the
# candidate is discarded no matter how good the other dimensions look.

# These answer "is this even the right recording?". Below the gate floor the
# candidate is discarded no matter how good the other dimensions look.
#
# One Noul answers exactly one proposition. `song` and `artist` used to be a single
# `identity` gate asking whether this is "the requested song performed by the
# requested artist", and that compound question was the weakest link in the rubric:
# asked about "Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger" for the song
# "Heme Aqui" it answered 56%, against 62-64% for the correct upload. The artist
# half was plainly true -- Marcos Witt, on Marcos Witt's own channel -- and it
# carried a compound answer for a song that is not the one requested.
#
# Two questions cannot do that. `song` cannot see the artist at all, and a shared
# word in the title ("Aqui" in both "Heme Aqui" and "Al Estar Aqui") is a fact about
# the string, not about the song. This is the decomposition the module's own design
# note calls for; it is late, and it is here because a measurement demanded it.
GATE_SONG = Dimension(
    key="song",
    gate=True,
    weight=0.0,
    instructions={
        "question": (
            "Is `candidate` a recording of the song requested in `target`?"
        ),
    },
    criteria={
        "true": (
            "The song performed is the one `target` names. A different spelling, "
            "punctuation or capitalisation, a featured-artist credit on the same "
            "song, or a title that adds the artist around it still counts."
        ),
        "false": (
            "A different song, even by the same artist on the same channel and even "
            "when the titles share words. A medley, a compilation of several songs, "
            "an instrumental setting of another piece, or a version whose title names "
            "a different song."
        ),
    },
    # Only what identifies *which song* this is. Notably not `channel`: it names the
    # performer, and letting it into this question is how the artist half of the
    # old compound question kept dragging the song half up with it.
    fields=_SONG_FIELDS,
)

GATE_ARTIST = Dimension(
    key="artist",
    gate=True,
    weight=0.0,
    instructions={
        "question": "Is `candidate` a recording performed by the artist requested in `target`?"},
    criteria={
        "true": (
            "The performing artist is the one `target` names, established by the "
            "title, the artists or the channel. A differently spelled name, or a "
            "channel belonging to the artist or to their label, still counts."
        ),
        "false": (
            "A different performer: another artist who happens to have recorded a "
            "song with the same name, an orchestra or a choir in place of the "
            "artist, or audio whose performer cannot be tied to the requested one."
        ),
    },
    # "artists ... or the channel" is named in its own criteria, so both travel
    # with it; so does duration, which catches a medley listed under one title.
    fields=_UPLOAD_FIELDS + ("duration_seconds",),
)

GATE_ORIGIN = Dimension(
    key="origin",
    gate=True,
    weight=0.0,
    instructions={
        "question": (
            "Is `candidate` published by the artist or their record label as a "
            "version of this song that a listener would accept as the song itself?"
        ),
    },
    criteria={
        "true": (
            "Published by the artist's own channel, an auto-generated 'Topic' channel, "
            "the record label, or a licensed distributor. Includes official releases "
            "and official live recordings."
        ),
        "false": (
            "A cover, tribute, karaoke, instrumental or sing-along rendition, a lyric "
            "or scrolling-text video, a reaction, commentary or compilation, an "
            "unofficial re-upload, or a 'DJ edit'/'sped up'/'reverb' variant. An "
            "upload by a fan or aggregator channel is false even when the audio "
            "itself is the original recording."
        ),
    },
    # Provenance is the whole question here: who published it, and whether the
    # upload presents itself as official. So the description and the source travel
    # with it; duration and view count do not.
    fields=_UPLOAD_FIELDS,
)

# A title saying "(Instrumental)" is still the artist's own upload, so it clears
# `origin`. Whether the *performance* is the song we asked for is a separate
# question, and one the previous gate set never asked -- which is how an
# instrumental ended up ranked third with origin 90%.
GATE_FORM = Dimension(
    key="form",
    gate=True,
    weight=0.0,
    instructions={
        "question": (
            "Is `candidate` the full song as a listener would expect to hear it, "
            "with the artist's vocal performance?"
        ),
    },
    criteria={
        "true": (
            "The complete song with the artist's vocals, from the first sung line to "
            "the end. Lyrics on screen, album art and video footage are fine; only "
            "the audio content matters."
        ),
        "false": (
            "An instrumental, backing track or karaoke version with no lead vocal, a "
            "cover sung by someone else, a spoken-word or acapella treatment, or an "
            "upload whose audio is missing, silent or replaced by commentary."
        ),
    },
    # "upload whose audio is missing, silent" is a judgement about what plays, so
    # length and upload date inform it; the description carries the vocal/instrumental
    # signal the title may omit.
    fields=_UPLOAD_FIELDS + ("duration_seconds", "upload_date", "description"),
)

# --- Ranking dimensions -----------------------------------------------------
# These never veto. They separate candidates that already cleared the gates.

DIM_STUDIO = _noul(
    "studio",
    {"question": "Is `candidate` a studio recording of the song?"},
    {
        "true": (
            "A studio recording, including an official remaster, extended mix or "
            "alternate take from the same studio session."
        ),
        "false": (
            "A concert, festival, tour, unplugged, acoustic or session performance, "
            "or a video whose audio comes from a live show."
        ),
    },
    # `provenance` is why this dimension can answer at all. Without it the question
    # arrived with a title, a channel and a duration and nothing that distinguishes
    # a studio master from a live cut, so it returned its prior -- measured at 8-21%
    # for every candidate in a real run, on the official channel and the fan upload
    # alike. Carrying the full description to fix that cost more than the signal was
    # worth; a token derived from the channel name costs about twenty characters.
    _RECORDING_FIELDS + ("artists", "provenance"),
)

DIM_AUDIO = _noul(
    "audio",
    {"question": "Is `candidate` audio of the song itself, with no commentary over it?"},
    {
        "true": (
            "The performance runs start to finish. Fades, brief silence and a short "
            "intro or outro are part of the track."
        ),
        "false": (
            "A spoken intro, outro, host commentary or reaction talking over the "
            "music, an announcement between songs, or a long stretch where the "
            "performance does not play."
        ),
    },
    # "runs start to finish" is about duration, and commentary is announced in the
    # description. No album, year, genre or view count.
    _RECORDING_FIELDS + ("artists",),
)

DIM_REFERENCE = _noul(
    "reference",
    {"question": ("Does `candidate` agree with the reference release details in `target`?")},
    {
        "true": (
            "The album, year or genre either match the reference, or `candidate` "
            "simply does not state them, which is not a disagreement."
        ),
        "false": (
            "`candidate` states an album, year or genre that contradicts the "
            "reference, which points at a different release of the same song."
        ),
    },
    # Exactly the fields the reference is made of. Notably no description: the
    # longest field by far, and nothing in these criteria can turn on it.
    _CATALOGUE_FIELDS,
)

GATE_DIMENSIONS: tuple[Dimension, ...] = (
    GATE_SONG,
    GATE_ARTIST,
    GATE_ORIGIN,
    GATE_FORM,
)
RANK_DIMENSIONS: tuple[Dimension, ...] = (DIM_STUDIO, DIM_AUDIO, DIM_REFERENCE)
DIMENSIONS: tuple[Dimension, ...] = GATE_DIMENSIONS + RANK_DIMENSIONS

# Only `form` turns on the upload description: it is the one question about what
# the video *is* (lyric video? official audio? commentary?) rather than about the
# recording. Every other dimension reads title and channel, so shipping the
# description again to each of them cost tokens for nothing -- it was 60% of the
# request. Dimensions that need it declare it here.
DESCRIPTION_DIMENSIONS: frozenset[str] = frozenset({GATE_FORM.key})

# Ranking weights. Identity and origin already veto, so these only decide
# between recordings of the right song.
#
# The order is the reverse of what it was, and the reason is measured rather than
# argued. It used to be studio 0.40 / audio 0.35 / reference 0.25, on the reasoning
# that reference "is the weakest signal (most uploads omit it)". Reference is in
# fact the strongest discriminator in the rubric:
#
#     right song          reference 81-96%
#     wrong song          reference 20-30%
#
# while `studio` ranged 5-72% and was the noisiest of the three -- and its 72% was
# an outlier on "Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger", a music video
# where the model appears to have answered "is this a professionally produced
# video" rather than "was this audio recorded in a studio". At weight 0.40 that
# one bad answer was worth +18.4 points, enough to carry a *different song* past
# the right one (66% to 63%) even after the publisher signal was added.
#
# So reference leads, because it is what actually tells the two apart, and studio
# and audio share the rest. `studio` keeps a real share: live against studio is a
# difference worth making, and it is made on the evidence rather than on the
# publisher, which cannot tell two uploads from one channel apart at all.
#
# Replayed over ten songs from two real runs, this ordering picks the right upload
# on all ten; the old one picks the wrong one on one of them.
RANK_WEIGHTS: dict[str, float] = {
    DIM_STUDIO.key: 0.30,
    DIM_AUDIO.key: 0.30,
    DIM_REFERENCE.key: 0.40,
}

# The dimension names the ranking is defined over, so a weight table that named
# something else would be a typo rather than a silent no-op.
RANK_DIMENSIONS_KEYS: tuple[str, ...] = tuple(RANK_WEIGHTS)

# --- Gate floors -------------------------------------------------------------
# A gate must clear this to be considered at all.
#
# One floor for all three gates was 0.50, and it vetoed every correct candidate in
# a real 5-song run: the official `- Topic` upload of "Alto precio" answered
# identity 32% / origin 41% / form 33%, and "Solo Una Palabra Tuya" 33/33/30. The
# model's absolute scale is compressed into roughly 0.30-0.45 for a match and
# 0.03-0.22 for a different song, so 0.50 sat *above* the whole correct-answer
# distribution. A veto that fires on everything discriminates nothing; it is noise
# wearing the shape of a rule.
#
# The floors are therefore per-gate, because the gates are not equally noisy:
# `origin` answered 52% for a different song on the artist's own channel, so it
# cannot carry the tightest floor. Measured separation, correct vs. different song:
#
#     identity  32-53%  vs   4-22%
#     origin    33-43%  vs   3-52%
#     form      30-66%  vs   6-23%
#
# `song` and `artist` have no measured distribution at all -- the model has never
# been asked them separately, which is the whole point of asking. So they start at
# the floor `identity` was measured at rather than at a number invented for them,
# and the calibration log is how they get real ones. Guessing a tighter floor for
# an unmeasured gate is how the 0.50 got there in the first place.
#
# These are five songs, not a calibration set -- the gap between a correct answer
# at 30% and a wrong one at 23% is seven points, which is narrower than anyone
# should trust. So the floors are set conservatively *below* the observed wrong
# maximum rather than optimally between the distributions, and every gate can be
# overridden by :func:`deterministic_shield` for the cases where the evidence does
# not depend on the model at all. `core.py` writes a JSONL calibration log for the
# same reason: retune these against real data, not against this paragraph.
GATE_FLOORS: dict[str, float] = {
    "song": 0.22,
    "artist": 0.22,
    "origin": 0.18,
    "form": 0.22,
}

# Kept for callers that pass one number (and for the provider default arguments).
# It is the loosest floor, so a scalar override never becomes stricter than the
# per-gate defaults -- which is the safe direction for a floor that guards against
# admitting a bad candidate.
GATE_FLOOR = min(GATE_FLOORS.values())


def gate_floor_for(
    dimension_key: str,
    floors: dict[str, float] | None = None,
    default: float | None = None,
) -> float:
    """The floor for one gate.

    *floors* -- the configured mapping -- is consulted first, then
    :data:`GATE_FLOORS`, then *default*, then the loosest known floor.

    The configured mapping has to win, or ``Config.DECISION_GATE_FLOORS`` would be
    a field that reads as a knob and does nothing: this function originally went
    straight to ``GATE_FLOORS`` and treated the caller's value as a fallback, so a
    tuned floor was silently discarded on every evaluation.
    """
    for mapping in (floors, GATE_FLOORS):
        if mapping and dimension_key in mapping:
            return float(mapping[dimension_key])
    if default is not None:
        return float(default)
    return GATE_FLOOR


def deterministic_shield(candidate: dict, artist: str, song: str) -> bool:
    """Whether this candidate is provably the right one without asking a model.

    All three must hold:

    * the normalised title is **exactly** the normalised requested title, so the
      upload is not a medley, an alternate take, a live cut or a re-title;
    * the publisher is proven: a ``- Topic`` channel belonging to this artist, or a
      VEVO-style official channel (:func:`provenance_signal`). An artist's own
      channel counts too -- it is the same fact about a different spelling of it;
    * the candidate was not hard-rejected by the heuristic, so no title word
      ("cover", "instrumental", "lyrics") contradicts the match.

    This exists because a miscalibrated probability must not be able to veto a fact.
    A gate floor tuned on a handful of songs is a guess; "a channel this artist
    controls, carrying the exact requested title" is not. It is deliberately narrow
    -- no fuzzy title match, no duration tolerance, no "close enough" channel -- so
    it cannot rescue the bad-search-result case the floors are there to catch.
    """
    if int(candidate.get("_composite_score") or 0) <= HARD_REJECT_SCORE:
        return False

    if provenance_signal(candidate, artist) == "unknown":
        return False

    artist_clean = normalize_title(strip_featuring(str(artist or "").lower()))
    if not artist_clean:
        return False

    title = normalize_title(strip_featuring(str(candidate.get("title") or "").lower()))
    if not title:
        return False
    # remove_matching_noise drops the "(Official Audio)" / "HD" / "Remastered"
    # decorations that sit alongside an otherwise exact title.
    return remove_matching_noise(title) == normalize_title(strip_featuring(str(song or "").lower()))


# Spread across runs above this is treated as unstable rather than decisive.
STABLE_SPREAD = 0.15


def question_id(candidate_key: str, dimension_key: str) -> str:
    return f"{candidate_key}::{dimension_key}"


def choice_id() -> str:
    return "best_match::choice"


def target_state(artist: str, song: str, metadata: dict | None) -> dict[str, Any]:
    """The state: only what we are looking for, never the candidates."""
    target: dict[str, Any] = {"artist": artist, "song": song}
    reference = reference_fields(metadata)
    if reference:
        target["reference"] = reference
    return {"target": target}


def reference_fields(metadata: dict | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {key: metadata[key] for key in REFERENCE_KEYS if metadata.get(key) not in (None, "")}


def _first_value(candidate: dict, *keys: str) -> Any:
    for key in keys:
        value = candidate.get(key)
        if value not in (None, ""):
            return value
    return None


def provenance_signal(candidate: dict, artist: str = "") -> str:
    """What the publisher alone already proves about where this recording came from.

    Four values, because these are the four answers that matter and all four are
    facts rather than judgments:

    * ``topic_channel`` -- an auto-generated channel *belonging to this artist*.
      YouTube creates these per release from the distributor's metadata, so the
      audio is the label's own master. The strongest provenance there is.
    * ``artist_channel`` -- a channel the artist controls. A live upload, an
      official video, a lyric video: their own material, which is still theirs.
    * ``official_audio`` -- a VEVO-style or catalogue-sourced release.
    * ``unknown`` -- everything else, including fan re-uploads with tidy names.

    The artist name has to be a whole run of words in the channel name. A plain
    substring test is not enough: the artist "Alto" matches the channel
    "Alto Precio" inside it, which would hand the strongest provenance to
    whatever channel happens to contain the name as a fragment.

    This is a conclusion drawn in Python on purpose. Asking a model "was this
    published by the artist or their label?" is what ``GATE_ORIGIN`` already does,
    and it answered 37% for the official Topic upload and 52% for a *different*
    song by the same artist -- so it cannot be the thing that decides.
    """
    channel = str(candidate.get("channel") or candidate.get("uploader") or "").strip()
    if not channel:
        return "unknown"

    channel_clean = normalize_title(channel)
    artist_clean = normalize_title(strip_featuring(str(artist or "").lower()))
    owned_by_artist = (
        bool(artist_clean)
        and f" {artist_clean} " in f" {channel_clean} "
    )

    if channel_clean.endswith(" topic"):
        return "topic_channel" if owned_by_artist else "unknown"
    if channel_clean.endswith("vevo"):
        return "official_audio"
    if owned_by_artist:
        return "artist_channel"
    if candidate.get("_source") in ("ytmusic_api", "itunes"):
        return "official_audio"
    return "unknown"


# How much provenance is worth in the ranking score.
#
# This is the one signal the ranking was missing, and the omission is visible in a
# real run: with the ranking as it stood, "IC0048-Alto Precio" on a church channel
# beat the official `- Topic` upload 45% to 41%, and a LYRIC VIDEO on a fan channel
# beat the official audio 39% to 37%. Not because those candidates were judged
# better -- `studio` and `audio` had no way to tell them apart and answered 5-24%
# and 16-81% respectively for official and fan uploads alike -- but because the
# composite was three noisy numbers and the tie went to whichever row came first.
#
# So provenance is added here as what it is: a fact about the publisher, already
# computed, carrying more weight than any of the three dimensions it outranks. It
# is deliberately *not* a gate. The gates still decide whether this is the right
# song, the right artist and a real performance; this only decides, between
# recordings that already cleared those, which publisher to trust.
#
# Sized against the measured composites: the official uploads scored 0.37-0.43
# and the ones that wrongly beat them 0.39-0.45, so any bonus above ~0.06 decides
# those cases. 0.15 is comfortably clear of that line without being decisive over
# everything else: because `studio` carries weight 0.40, it takes a 0.375 swing in
# that dimension -- a live take against a studio one -- to overturn the publisher,
# which is the order of difference a real live/studio contrast produces.
PROVENANCE_RANK_BONUS: dict[str, float] = {
    "topic_channel": 0.15,
    "artist_channel": 0.13,
    "official_audio": 0.10,
    "unknown": 0.0,
}

# Any unknown key scores as no bonus rather than raising, so a future provenance
# value cannot silently outrank the official one by omission.
DEFAULT_PROVENANCE_BONUS: float = 0.0


def provenance_bonus(provenance: str) -> float:
    return PROVENANCE_RANK_BONUS.get(provenance, DEFAULT_PROVENANCE_BONUS)


def candidate_state(
    candidate: dict,
    index: int,
    with_description: bool = False,
    artist: str = "",
) -> dict[str, Any]:
    """The candidate fields the model reasons about.

    Deliberately excludes the heuristic score: a decision model that can see its
    own prior anchors to it, which defeats the point of asking it separately.

    ``with_description`` is opt-in because a candidate is serialized once per
    dimension. Only ``form`` asks what the upload is rather than what the
    recording is, so only ``form`` pays for the description.

    ``artist`` is needed by :func:`provenance_signal` to tell the artist's own
    ``- Topic`` channel from someone else's. Without it every candidate reads as
    ``unknown``, which is the safe direction but a wasted field.
    """
    state: dict[str, Any] = {
        "key": f"candidate_{index}",
        "title": str(candidate.get("title") or ""),
        "channel": str(candidate.get("channel") or candidate.get("uploader") or ""),
        "artists": candidate.get("artists") or [],
        "duration_seconds": candidate.get("duration"),
        "source": str(candidate.get("_source") or "unknown"),
        "provenance": provenance_signal(candidate, artist),
        "upload_date": _first_value(candidate, "upload_date", "release_date"),
        "album": candidate.get("album"),
        "year": _first_value(candidate, "year", "release_year"),
        "genre": candidate.get("genre"),
        "view_count": candidate.get("view_count"),
        "is_live": bool(candidate.get("is_live")),
    }
    if with_description:
        state["description"] = str(candidate.get("description") or "")[:DESCRIPTION_LIMIT]
    return state


def candidate_label(candidate: dict[str, Any]) -> str:
    title = str(candidate.get("title") or "untitled")
    channel = str(candidate.get("channel") or candidate.get("uploader") or "unknown channel")
    return f'candidate "{title}" uploaded by "{channel}"'


def build_questions(
    candidate_states: list[dict[str, Any]],
    dialect: Dialect = TYPESAFE_DIALECT,
) -> dict[str, dict[str, Any]]:
    """Every dimension for every candidate, plus one tie-break Choice.

    Each question carries only the candidate fields its own ``fields`` tuple
    declares, so nothing is serialised once per dimension for the sake of a
    question that cannot use it. The state is shared and prefilled once; the
    questions are not, and their size is what the GPU actually pays for.

    Adding questions barely changes the response time -- they are evaluated in
    parallel -- so the whole rubric still ships in a single request.
    """
    questions: dict[str, dict[str, Any]] = {}
    for state in candidate_states:
        for dimension in DIMENSIONS:
            questions[question_id(state["key"], dimension.key)] = dimension.question(
                _project(state, dimension.fields), dialect.noul
            )
    if len(candidate_states) > 1:
        questions[choice_id()] = {
            "type": dialect.choice,
            "instructions": (
                "Which candidate below is the recording we want to download, based on "
                "the `target` in `state`?"
            ),
            "criteria": {state["key"]: candidate_label(state) for state in candidate_states},
        }
    return questions


def _project(state: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """The subset of *state* a dimension reads.

    ``key`` is always kept: it is how the answer is attributed back to the
    candidate, so dropping it would break the contract rather than just the
    prompt. A dimension with no declared fields gets the full state, which keeps
    any future dimension correct by default instead of silently starved.
    """
    if not fields:
        return dict(state)
    return {name: state[name] for name in fields if name in state}


def build_state_and_questions(
    artist: str,
    song: str,
    candidates: list[dict],
    reference_metadata: dict | None = None,
    dialect: Dialect = TYPESAFE_DIALECT,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build the whole System One request: (state, candidate_states, questions).

    ``candidate_states`` comes back because the caller needs it to attribute each
    answer back to the original candidate.
    """
    keep_description = DESCRIPTION_DIMENSIONS
    states = [
        # The description is kept here and stripped per dimension in
        # build_questions, so one candidate is read once instead of once per
        # dimension.
        candidate_state(
            candidate,
            index,
            with_description=bool(keep_description),
            artist=artist,
        )
        for index, candidate in enumerate(candidates)
    ]
    return (
        target_state(artist, song, reference_metadata),
        states,
        build_questions(states, dialect),
    )


def select_for_evaluation(
    candidates: list[dict],
    limit: int = MAX_EVALUATED_CANDIDATES,
    headroom: int = EVALUATION_HEADROOM,
) -> list[int]:
    """Indices of the candidates worth spending a model call on.

    Three cuts, all of them cheap and all of them ones the search should have
    made before asking a model:

    * Anything the heuristic hard-rejected (a score at the floor, e.g. a title
      containing "cover") is dropped. Those are deterministic word matches on the
      title; re-asking a model to overturn them is both slower and, in a real
      run, let a score of 87 overrule a correct "instrumental" rejection.
    * Exact duplicates of an already-kept candidate are dropped. Search returns
      the same upload from several sources; asking about it twice buys nothing
      and spends a full set of questions.
    * Whatever is still past ``limit`` is dropped, taking the best ``headroom``
      extra so that near-ties at the cut are not decided by list order alone.
    """
    keep: list[int] = []
    seen: set[tuple[Any, ...]] = set()
    for index, candidate in enumerate(candidates):
        if int(candidate.get("_composite_score") or 0) <= HARD_REJECT_SCORE:
            continue
        fingerprint = _duplicate_fingerprint(candidate)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        keep.append(index)

    if limit and len(keep) > limit:
        budget = limit + max(0, headroom)
        # Anything the heuristic scored as well as its best candidate is always
        # worth a second opinion -- the cap exists to trim the tail, not to
        # decide a tie between equals by list position.
        best = max(int(candidates[index].get("_composite_score") or 0) for index in keep)
        protected = [
            index for index in keep if int(candidates[index].get("_composite_score") or 0) >= best
        ]
        rest = [index for index in keep if index not in set(protected)]
        return sorted(protected + rest[: max(0, budget - len(protected))])
    return keep


def _duplicate_fingerprint(candidate: dict) -> tuple[str | int, ...]:
    """Two rows are the same upload when artist, song and length all agree."""
    duration = int(candidate.get("duration") or 0)
    return (
        str(candidate.get("title") or "").strip().lower(),
        str(candidate.get("channel") or candidate.get("uploader") or "").strip().lower(),
        duration // 5,
    )
