from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ytdl_core import decision_questions as dq
from ytdl_core.jev import JevClassifier, JevEvaluationError


def _candidate(title: str, score: int) -> dict:
    return {
        "title": title,
        "channel": "Official Artist",
        "duration": 200,
        "_composite_score": score,
        "_score_breakdown": {"base_match": score},
    }


DEFAULTS = {
    dq.GATE_IDENTITY.key: 0.95,
    dq.GATE_ORIGIN.key: 0.90,
    dq.GATE_FORM.key: 0.92,
    dq.DIM_STUDIO.key: 0.80,
    dq.DIM_AUDIO.key: 0.85,
    dq.DIM_REFERENCE.key: 0.70,
}


def _answers(
    keys: list[str],
    *,
    identity: float | None = None,
    origin: float | None = None,
    form: float | None = None,
    studio: float | None = None,
    audio: float | None = None,
    reference: float | None = None,
    per_key: dict[str, dict[str, float]] | None = None,
    choice: dict[str, float] | None = None,
) -> dict:
    """Build a typed answer set: every dimension, every candidate, same values."""
    overrides = {
        dq.GATE_IDENTITY.key: identity,
        dq.GATE_ORIGIN.key: origin,
        dq.GATE_FORM.key: form,
        dq.DIM_STUDIO.key: studio,
        dq.DIM_AUDIO.key: audio,
        dq.DIM_REFERENCE.key: reference,
    }
    answers: dict = {}
    for key in keys:
        for dimension_key, base in DEFAULTS.items():
            value = (per_key or {}).get(key, {}).get(dimension_key, overrides[dimension_key])
            answers[dq.question_id(key, dimension_key)] = {
                "type": "noul",
                "noul": DEFAULTS[dimension_key] if value is None else float(value),
            }
    if choice is not None:
        answers[dq.choice_id()] = {
            "type": "choice",
            "choice": max(choice, key=choice.get),
            "probabilities": choice,
            "confidence": 0.81,
        }
    return answers


def test_default_project_root_contains_bridge():
    classifier = JevClassifier()
    script = (classifier.project_root / "tools" / "jev.mts").read_text(encoding="utf-8")

    assert script.count("fetch(") == 1
    assert "from 'ai'" not in script


def test_bridge_forwards_contract_without_restating_questions():
    """The bridge must not own the question text -- that would drift from Python."""
    script = (JevClassifier().project_root / "tools" / "jev.mts").read_text(encoding="utf-8")

    assert '"type"' not in script
    assert "noul" not in script
    assert "choice" not in script
    assert "represent exactly the requested song" not in script
    assert "request.questions" in script


def test_asks_one_noul_per_dimension_with_criteria(monkeypatch):
    classifier = JevClassifier()
    captured = {}

    def fake_evaluate(state, questions, model=None):
        captured.update({"state": state, "questions": questions, "model": model})
        return _answers(["candidate_0"])

    monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
    classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])

    expected = {dq.question_id("candidate_0", dimension.key) for dimension in dq.DIMENSIONS}
    assert expected <= set(captured["questions"])
    for qid in expected:
        question = captured["questions"][qid]
        assert question["type"] == classifier.dialect.noul
        assert set(question["criteria"]) == {"true", "false"}
        assert "candidate" in question["instructions"]
        assert "question" in question["instructions"]


def test_each_provider_declares_its_own_dialect():
    """Same judgments, different wire vocabulary for the yes/no primitive."""
    from ytdl_core.kev import KevClassifier

    assert JevClassifier().dialect.noul == "boolean"
    assert KevClassifier().dialect.noul == "noul"
    assert dq.TYPESAFE_DIALECT.choice == dq.GATEWAY_DIALECT.choice == "choice"
    # Only the type name differs; the question itself is identical.
    states = [dq.candidate_state(_candidate("Artist - Song", 120), 0)]
    native = dq.build_questions(states, dq.TYPESAFE_DIALECT)
    gateway = dq.build_questions(states, dq.GATEWAY_DIALECT)
    qid = dq.question_id("candidate_0", dq.GATE_IDENTITY.key)
    assert native[qid]["criteria"] == gateway[qid]["criteria"]
    assert native[qid]["instructions"] == gateway[qid]["instructions"]
    assert {native[qid]["type"], gateway[qid]["type"]} == {"noul", "boolean"}


def test_reads_either_answer_key():
    """A gateway `probability` and a TypeSafe `noul` are the same number."""
    classifier = JevClassifier()
    for answer in ({"type": "noul", "noul": 0.81}, {"type": "boolean", "probability": 0.81}):
        assert classifier._probability(answer, "qid") == pytest.approx(0.81)


def test_state_holds_only_the_target(monkeypatch):
    classifier = JevClassifier()
    captured = {}

    def fake_evaluate(state, questions, model=None):
        captured.update({"state": state})
        return _answers(["candidate_0", "candidate_1"])

    monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
    classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120), _candidate("Other - Song", 100)],
        reference_metadata={"album": "Reference Album", "year": "2024"},
    )

    assert captured["state"] == {
        "target": {
            "artist": "Artist",
            "song": "Song",
            "reference": {"album": "Reference Album", "year": "2024"},
        }
    }


def test_candidate_state_never_leaks_the_heuristic_score():
    candidate = _candidate("Artist - Song", 137)

    payload = dq.candidate_state(candidate, 0)

    assert "heuristic_score" not in payload
    assert (
        "137"
        not in dq.build_questions([payload])[dq.question_id(payload["key"], "studio")][
            "instructions"
        ]["candidate"].values()
    )


def test_choice_question_only_when_several_candidates(monkeypatch):
    classifier = JevClassifier()

    seen: list[dict] = []
    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda state, questions, model=None: seen.append(questions) or _answers(["candidate_0"]),
    )

    classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])
    assert dq.choice_id() not in seen[0]

    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda state, questions, model=None: (
            seen.append(questions) or _answers(["candidate_0", "candidate_1"])
        ),
    )
    classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120), _candidate("Artist - Song (Live)", 110)],
    )
    assert seen[1][dq.choice_id()]["type"] == "choice"
    assert set(seen[1][dq.choice_id()]["criteria"]) == {"candidate_0", "candidate_1"}


def test_gate_failure_vetoes_candidate_even_with_perfect_ranking(monkeypatch):
    classifier = JevClassifier(threshold=0.50)
    answers = _answers(["candidate_0"], studio=1.0, audio=1.0, reference=1.0)
    answers[dq.question_id("candidate_0", dq.GATE_ORIGIN.key)] = {"type": "noul", "noul": 0.05}
    monkeypatch.setattr(classifier, "_evaluate", lambda state, questions, model=None: answers)

    selected, ranked = classifier.select(
        "Artist",
        "Song",
        [_candidate("Some Guy - Song (Cover)", 200)],
    )

    assert selected is None
    assert ranked[0][0]["_decision_eligible"] is False
    assert ranked[0][0]["_decision_failed_gates"] == ["origin"]
    assert ranked[0][0]["_composite_score"] == 0


def test_ranking_score_is_the_weighted_mean_of_ranking_dimensions(monkeypatch):
    classifier = JevClassifier(threshold=0.50)
    answers = _answers(["candidate_0"], studio=1.0, audio=0.5, reference=0.0)
    monkeypatch.setattr(classifier, "_evaluate", lambda state, questions, model=None: answers)

    selected, _ = classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])

    expected = (
        dq.RANK_WEIGHTS[dq.DIM_STUDIO.key] * 1.0
        + dq.RANK_WEIGHTS[dq.DIM_AUDIO.key] * 0.5
        + dq.RANK_WEIGHTS[dq.DIM_REFERENCE.key] * 0.0
    ) / sum(dq.RANK_WEIGHTS.values())
    assert selected is not None
    assert selected["_decision_score"] == pytest.approx(expected)


def test_gates_do_not_double_count_into_the_ranking_score(monkeypatch):
    """Perfect gates must not inflate the ranking score of a poor recording."""
    classifier = JevClassifier(threshold=0.10)
    perfect_gates = _answers(["candidate_0"], studio=0.2, audio=0.2, reference=0.2)
    monkeypatch.setattr(classifier, "_evaluate", lambda state, questions, model=None: perfect_gates)

    selected, _ = classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])

    assert selected is not None
    assert selected["_decision_eligible"] is True
    assert selected["_decision_dimensions"][dq.GATE_IDENTITY.key] == pytest.approx(0.95)
    assert selected["_decision_score"] == pytest.approx(0.2)


def test_selects_candidate_with_highest_score(monkeypatch):
    classifier = JevClassifier(threshold=0.50)
    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda state, questions, model=None: _answers(["candidate_0", "candidate_1"]),
    )

    selected, ranked = classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120), _candidate("Artist - Song (Live)", 110)],
    )

    assert selected is not None
    assert selected["title"] == "Artist - Song"
    assert selected["_decision_selected"] is True
    assert ranked[0][0]["_heuristic_score"] == 120
    assert ranked[0][0]["_decision_contract"] == dq.QUESTION_CONTRACT_VERSION


def test_repeated_runs_average_per_dimension(monkeypatch):
    classifier = JevClassifier(threshold=0.50)
    responses = iter(
        [
            _answers(["candidate_0"], studio=0.20, audio=0.20, reference=0.20),
            _answers(["candidate_0"], studio=0.60, audio=0.60, reference=0.60),
            _answers(["candidate_0"], studio=1.00, audio=1.00, reference=1.00),
        ]
    )
    monkeypatch.setattr(
        classifier, "_evaluate", lambda state, questions, model=None: next(responses)
    )

    selected, _ = classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)], runs=3)

    assert selected is not None
    assert selected["_decision_dimensions"][dq.DIM_STUDIO.key] == pytest.approx(0.60)
    assert selected["_decision_runs"] == 3


def test_unstable_answers_sort_below_stable_ones(monkeypatch):
    """A coin flip that landed high is not evidence, so stability outranks score."""
    classifier = JevClassifier(threshold=0.50)
    both = ["candidate_0", "candidate_1"]
    # candidate_0 swings between perfect and zero; candidate_1 is steady at 0.9.
    noisy_perfect = _answers(
        both,
        per_key={
            "candidate_0": {key: 1.0 for key in DEFAULTS},
            "candidate_1": {key: 0.9 for key in DEFAULTS},
        },
    )
    noisy_zero = _answers(
        both,
        per_key={
            "candidate_0": {key: 0.0 for key in DEFAULTS},
            "candidate_1": {key: 0.9 for key in DEFAULTS},
        },
    )
    runs = iter([noisy_perfect, noisy_zero, noisy_perfect, noisy_zero])
    monkeypatch.setattr(classifier, "_evaluate", lambda state, questions, model=None: next(runs))

    selected, ranked = classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120), _candidate("Artist - Song (Alt)", 120)],
        runs=2,
    )

    assert selected is not None
    assert selected["title"] == "Artist - Song (Alt)"
    assert selected["_decision_stable"] is True
    unstable = ranked[1][0]
    assert unstable["_decision_stable"] is False
    assert unstable["_decision_spread"] > 0.0


def test_choice_probability_and_confidence_are_recorded(monkeypatch):
    classifier = JevClassifier(threshold=0.50)
    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda state, questions, model=None: _answers(
            ["candidate_0", "candidate_1"], choice={"candidate_0": 0.12, "candidate_1": 0.88}
        ),
    )

    selected, ranked = classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120), _candidate("Artist - Song (Alt)", 110)],
    )
    by_title = {entry["title"]: entry for entry, _, _ in ranked}

    assert selected is not None
    assert by_title["Artist - Song"]["_decision_choice_probability"] == pytest.approx(0.12)
    assert by_title["Artist - Song (Alt)"]["_decision_choice_probability"] == pytest.approx(0.88)
    assert by_title["Artist - Song"]["_decision_confidence"] == pytest.approx(0.81)


def test_rejects_invalid_run_count():
    classifier = JevClassifier()

    with pytest.raises(JevEvaluationError, match="at least 1"):
        classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)], runs=0)


def test_three_gates_not_two():
    """identity / origin / form. The third one is what catches an instrumental."""
    assert [d.key for d in dq.GATE_DIMENSIONS] == ["identity", "origin", "form"]


def test_form_gate_vetoes_an_instrumental_that_clears_origin(monkeypatch):
    """An instrumental *by the artist* is officially published, so `origin`
    approves it. Only `form` can tell it is not the song we asked for."""
    classifier = JevClassifier(threshold=0.50)
    answers = _answers(["candidate_0"], identity=0.97, origin=0.90, form=0.04)
    monkeypatch.setattr(classifier, "_evaluate", lambda state, questions, model=None: answers)

    selected, ranked = classifier.select(
        "Artist", "Song", [_candidate("Artist - Song (Instrumental)", 120)]
    )

    assert selected is None
    assert ranked[0][0]["_decision_failed_gates"] == ["form"]
    assert ranked[0][0]["_decision_dimensions"][dq.GATE_ORIGIN.key] == pytest.approx(0.90)


def test_only_the_form_gate_receives_the_description():
    """The description was 60% of the request; only `form` reads it."""
    states = [dq.candidate_state({"title": "t", "description": "d" * 4000}, 0, True)]
    questions = dq.build_questions(states)

    for dimension in dq.DIMENSIONS:
        candidate = questions[dq.question_id("candidate_0", dimension.key)]["instructions"][
            "candidate"
        ]
        if dimension.key in dq.DESCRIPTION_DIMENSIONS:
            assert candidate["description"]
        else:
            assert "description" not in candidate


def test_description_still_reaches_the_form_gate_through_the_full_build():
    candidate = _candidate("Artist - Song", 120)
    candidate["description"] = "Official lyric video with scrolling text"
    states = [dq.candidate_state(candidate, 0, with_description=True)]
    questions = dq.build_questions(states)

    form = questions[dq.question_id("candidate_0", dq.GATE_FORM.key)]
    assert form["instructions"]["candidate"]["description"].startswith("Official lyric")

    identity = questions[dq.question_id("candidate_0", dq.GATE_IDENTITY.key)]
    assert "description" not in identity["instructions"]["candidate"]


def test_hard_rejected_candidates_never_reach_the_model(monkeypatch):
    """The heuristic's title word-matches are not worth second-guessing."""
    classifier = JevClassifier(threshold=0.50)
    seen: list[dict] = []

    def fake_evaluate(state, questions, model=None):
        seen.append(questions)
        return _answers(["candidate_0"])

    monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
    cover = _candidate("Some Guy - Song (Cover)", 140)
    cover["_composite_score"] = -9999
    classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 165), cover],
    )

    keys = {key.split("::")[0] for key in seen[0] if key != dq.choice_id()}
    assert keys == {"candidate_0"}


def test_hard_rejected_candidates_stay_in_the_report(monkeypatch):
    classifier = JevClassifier(threshold=0.50)
    cover = _candidate("Some Guy - Song (Cover)", -9999)
    cover["_composite_score"] = -9999
    monkeypatch.setattr(
        classifier, "_evaluate", lambda state, questions, model=None: _answers(["candidate_0"])
    )

    selected, ranked = classifier.select(
        "Artist", "Song", [_candidate("Artist - Song", 165), cover]
    )

    titles = [entry["title"] for entry, _, _ in ranked]
    assert selected is not None
    assert "Some Guy - Song (Cover)" in titles
    unevaluated = [entry for entry, _, _ in ranked if entry.get("_decision_evaluated") is False]
    assert len(unevaluated) == 1
    assert unevaluated[0]["_composite_score"] == -9999


def test_candidate_cap_keeps_the_top_ones_plus_headroom(monkeypatch):
    classifier = JevClassifier(
        threshold=0.50,
        max_candidates=3,
        eval_headroom=dq.EVALUATION_HEADROOM,
        max_questions=10_000,  # isolate the candidate cap from the question budget
    )
    seen: list[dict] = []

    def fake_evaluate(state, questions, model=None):
        seen.append(questions)
        keys = sorted({key.split("::")[0] for key in questions})
        return _answers(keys)

    monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
    candidates = [_candidate(f"Artist - Song {i}", 200 - i) for i in range(10)]

    selected, ranked = classifier.select("Artist", "Song", candidates)

    evaluated = {key.split("::")[0] for key in seen[0] if key != dq.choice_id()}
    # The cap plus headroom, not exactly the cap: near-ties should not be decided
    # by list order alone.
    assert len(evaluated) == 3 + dq.EVALUATION_HEADROOM
    assert selected is not None
    # Everything the model never saw is still reported, so nothing disappears.
    assert len(ranked) == 10
    assert selected["title"] == "Artist - Song 0"


def test_question_budget_caps_the_request(monkeypatch):
    """Cost is questions, not candidates, so the budget is what must hold.

    At 12 candidates plus 4 of headroom this reached 97 questions for a single
    song -- minutes of generation before any download begins, multiplied again
    by --kev-runs. The budget is what actually keeps latency predictable.
    """
    seen: list[dict] = []
    budget = 32
    classifier = JevClassifier(threshold=0.50, max_candidates=12, max_questions=budget)

    def fake_evaluate(state, questions, model=None):
        seen.append(questions)
        keys = sorted({key.split("::")[0] for key in questions})
        return _answers(keys)

    monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
    candidates = [_candidate(f"Artist - Song {i}", 200 - i) for i in range(40)]

    selected, ranked = classifier.select("Artist", "Song", candidates)

    assert len(seen[0]) <= budget, "the question budget must actually bound the request"
    # Trimming drops the worst-scoring candidates, which are least likely to win.
    assert selected["title"] == "Artist - Song 0"
    # Nothing disappears from the report, even what the model never saw.
    assert len(ranked) == 40


def test_question_budget_is_multiplied_by_runs_in_practice(monkeypatch):
    """The cost that matters is questions x runs; the budget bounds one request."""
    dimension_count = len(dq.DIMENSIONS)
    affordable = (32 - 1) // dimension_count
    assert JevClassifier._question_count(affordable) <= 32
    assert JevClassifier._question_count(affordable + 1) > 32


def test_headroom_can_be_switched_off():
    candidates = [_candidate(f"Artist - Song {i}", 200 - i) for i in range(10)]
    assert dq.select_for_evaluation(candidates, limit=3, headroom=0) == [0, 1, 2]


def test_duplicate_uploads_are_evaluated_once():
    """Search returns the same file from several sources; asking twice is waste."""
    rows = [
        _candidate("Artist - Song", 165),
        _candidate("Artist - Song", 160),  # same title, channel, duration
        _candidate("Artist - Song", 170),  # same again
        _candidate("Artist - Song (Live)", 150),  # genuinely different
    ]

    assert dq.select_for_evaluation(rows, limit=10, headroom=0) == [0, 3]


def test_duplicates_differing_in_duration_are_kept():
    rows = [
        _candidate("Artist - Song", 165),
        {**_candidate("Artist - Song", 165), "duration": 240},
    ]

    assert dq.select_for_evaluation(rows, limit=10, headroom=0) == [0, 1]


def test_candidate_payload_includes_release_metadata():
    candidate = _candidate("Artist - Song", 120)
    candidate.update(
        {
            "description": "Official album recording",
            "upload_date": "20240102",
            "album": "Album",
            "year": "2024",
            "genre": "Gospel",
            "view_count": 1000,
            "is_live": False,
        }
    )

    payload = dq.candidate_state(candidate, 0, with_description=True)

    assert payload["description"] == "Official album recording"
    assert payload["upload_date"] == "20240102"
    assert payload["album"] == "Album"
    assert payload["year"] == "2024"
    assert payload["genre"] == "Gospel"
    assert payload["view_count"] == 1000


def test_description_is_trimmed():
    candidate = _candidate("Artist - Song", 120)
    candidate["description"] = "x" * (dq.DESCRIPTION_LIMIT + 500)

    assert len(dq.candidate_state(candidate, 0, True)["description"]) == dq.DESCRIPTION_LIMIT


def test_select_includes_reference_metadata(monkeypatch):
    classifier = JevClassifier()
    captured = {}

    def fake_evaluate(state, questions, model=None):
        captured.update({"state": state})
        return _answers(["candidate_0"])

    monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
    classifier.select(
        "Artist",
        "Song",
        [_candidate("Artist - Song", 120)],
        reference_metadata={"album": "Reference Album", "year": "2024", "recording_id": "x"},
    )

    assert captured["state"]["target"]["reference"] == {
        "album": "Reference Album",
        "year": "2024",
    }


def test_rejects_when_score_is_below_threshold(monkeypatch):
    classifier = JevClassifier(threshold=0.90)
    monkeypatch.setattr(
        classifier,
        "_evaluate",
        lambda state, questions, model=None: _answers(["candidate_0"], studio=0.5, audio=0.5),
    )

    selected, ranked = classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])

    assert selected is None
    assert ranked[0][0]["_decision_eligible"] is True
    assert ranked[0][0]["_decision_probability"] < 0.90


def test_rejects_invalid_probability(monkeypatch):
    classifier = JevClassifier()
    answers = _answers(["candidate_0"])
    answers[dq.question_id("candidate_0", dq.GATE_IDENTITY.key)] = {"type": "noul", "noul": 1.5}
    monkeypatch.setattr(classifier, "_evaluate", lambda state, questions, model=None: answers)

    with pytest.raises(JevEvaluationError, match="invalid probability"):
        classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])


def test_reports_the_missing_question_id(monkeypatch):
    classifier = JevClassifier()
    answers = _answers(["candidate_0"])
    answers.pop(dq.question_id("candidate_0", dq.DIM_AUDIO.key))
    monkeypatch.setattr(classifier, "_evaluate", lambda state, questions, model=None: answers)

    with pytest.raises(JevEvaluationError, match="no probability for candidate_0::audio"):
        classifier.select("Artist", "Song", [_candidate("Artist - Song", 120)])


def test_bridge_error_identifies_gateway_cause():
    assert "credit card" in JevClassifier._bridge_error("valid credit card required")
    assert "authentication" in JevClassifier._bridge_error("Unauthenticated request")
    assert "rate limit" in JevClassifier._bridge_error("HTTP 429 rate limit")
    assert "temporarily failed" in JevClassifier._bridge_error("HTTP 503 internal server error")


def test_bridge_error_names_a_free_tier_account():
    assert "free tier" in JevClassifier._bridge_error(
        '{"type":"no_providers_available","name":"RestrictedModelsError",'
        '"message":"Free tier users do not have access to this model."}'
    )


def test_bridge_error_names_a_malformed_question_contract():
    assert "malformed" in JevClassifier._bridge_error(
        "Invalid discriminator value. Expected 'choice' | 'score' | 'boolean'"
    )


def test_evaluate_uses_utf8_for_candidate_text(monkeypatch):
    classifier = JevClassifier(project_root=Path(__file__).resolve().parents[1])

    def fake_run(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return subprocess.CompletedProcess(command, 0, '{"answers": {}}', "")

    monkeypatch.setattr("ytdl_core.jev.subprocess.run", fake_run)

    assert classifier._evaluate({"text": "♡"}, {}) == {}
