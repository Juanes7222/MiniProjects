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

# Bumped whenever the question set, the state shape or the scoring math changes,
# so stored decisions can be invalidated instead of silently compared to old ones.
QUESTION_CONTRACT_VERSION = 2

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
_WHO_FIELDS = IDENTITY_FIELDS + ("artists",)
_UPLOAD_FIELDS = IDENTITY_FIELDS + ("artists", "source")
_RECORDING_FIELDS = IDENTITY_FIELDS + ("duration_seconds", "upload_date")
_CATALOGUE_FIELDS = IDENTITY_FIELDS + ("album", "year", "genre", "upload_date")


# --- Gates ------------------------------------------------------------------
# These answer "is this even the right recording?". Below GATE_FLOOR the
# candidate is discarded no matter how good the other dimensions look.

GATE_IDENTITY = Dimension(
    key="identity",
    gate=True,
    weight=0.0,
    instructions={
        "question": ("Is `candidate` the requested song performed by the requested artist?"),
    },
    criteria={
        "true": (
            "The title, artists, channel or description identify the requested song "
            "and the requested artist or their official channel. A featured-artist "
            "credit or a differently spelled title still counts."
        ),
        "false": (
            "A different song, a different artist performing the song, a mashup of "
            "several songs, or audio whose artist cannot be tied to the requested one."
        ),
    },
    # "artists ... or description identify the requested song" -- both are named in
    # its own criteria, so both travel with it.
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
    # Live-show evidence lives in the upload date and the description ("live at...",
    # venue names, crowd noise); the catalogue fields are not consulted here.
    _RECORDING_FIELDS + ("artists",),
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

GATE_DIMENSIONS: tuple[Dimension, ...] = (GATE_IDENTITY, GATE_ORIGIN, GATE_FORM)
RANK_DIMENSIONS: tuple[Dimension, ...] = (DIM_STUDIO, DIM_AUDIO, DIM_REFERENCE)
DIMENSIONS: tuple[Dimension, ...] = GATE_DIMENSIONS + RANK_DIMENSIONS

# Only `form` turns on the upload description: it is the one question about what
# the video *is* (lyric video? official audio? commentary?) rather than about the
# recording. Every other dimension reads title and channel, so shipping the
# description again to each of them cost tokens for nothing -- it was 60% of the
# request. Dimensions that need it declare it here.
DESCRIPTION_DIMENSIONS: frozenset[str] = frozenset({GATE_FORM.key})

# Ranking weights. Identity and origin already vetoed, so these only decide
# between recordings of the right song: quality of the master first, reference
# agreement last, because it is the weakest signal (most uploads omit it).
RANK_WEIGHTS: dict[str, float] = {
    DIM_STUDIO.key: 0.40,
    DIM_AUDIO.key: 0.35,
    DIM_REFERENCE.key: 0.25,
}

# A gate must clear this to be considered at all.
GATE_FLOOR = 0.50

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


def candidate_state(
    candidate: dict,
    index: int,
    with_description: bool = False,
) -> dict[str, Any]:
    """The candidate fields the model reasons about.

    Deliberately excludes the heuristic score: a decision model that can see its
    own prior anchors to it, which defeats the point of asking it separately.

    ``with_description`` is opt-in because a candidate is serialized once per
    dimension. Only ``form`` asks what the upload is rather than what the
    recording is, so only ``form`` pays for the description.
    """
    state: dict[str, Any] = {
        "key": f"candidate_{index}",
        "title": str(candidate.get("title") or ""),
        "channel": str(candidate.get("channel") or candidate.get("uploader") or ""),
        "artists": candidate.get("artists") or [],
        "duration_seconds": candidate.get("duration"),
        "source": str(candidate.get("_source") or "unknown"),
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
        candidate_state(candidate, index, with_description=bool(keep_description))
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
