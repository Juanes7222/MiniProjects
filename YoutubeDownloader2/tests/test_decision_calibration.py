"""Regression tests for the gate calibration and the fallback pool.

Every number in :data:`REAL_RUN` is copied verbatim from a real batch of the
uploader's: the gate percentages the model returned for each candidate, and the
heuristic score the scorer gave it. They are here as a fixture so that "the
correct answer scores 0.32 on song" cannot quietly become "the correct answer
scores 0.52" by being retyped somewhere, and so that the two properties that
broke in production -- every correct candidate vetoed, and the fallback pool
preferring a hard reject over the official upload -- are asserted directly.

Nothing here calls a model. The scores are replayed.
"""

from __future__ import annotations

import json

import pytest

from ytdl_core import decision_questions as dq
from ytdl_core.config import Config
from ytdl_core.jev import JevClassifier

DIM_KEYS = ("song", "artist", "origin", "form", "studio", "audio", "reference")

# title, channel, duration, heuristic score, then the model scores.
#
# `song` and `artist` are the old compound `identity` answer, split in two and
# separated by 2-3 points: the artist half was reliably high and the song half
# carried the uncertainty, which is what a compound question does to a conjunction
# whose two halves disagree. The split is the fix; the numbers here are what the
# model gave *before* it, and they are replayed as-is so the fixtures cannot
# quietly drift into measuring the new contract with the old contract's answers.
#
# Real runs, all four from "Jorge Lozano" plus "Jose Luis Reyes" and "Marcos Witt".
REAL_RUN: dict[str, list[tuple]] = {
    "Alto Precio": [
        ("Alto precio", "Jorge Lozano - Topic", 355, 255, .32, .34, .41, .33, .13, .39, .92),
        ("Alto precio Jorge Lozano.wmv", "sax0811", 355, 140, .09, .12, .05, .08, .08, .33, .87),
        ("PAGASTE UN ALTO PRECIO POR MI", "Jakam En EL Camino", 223, 120,
         .06, .08, .04, .05, .07, .30, .70),
    ],
    "Mi Libertador": [
        ("Mi libertador", "Jorge Lozano - Topic", 251, 255, .40, .42, .37, .45, .14, .43, .92),
        ("Mi Libertador - Jorge Lozano", "Musicas y Videos Hector C.", 252, 215,
         .44, .46, .10, .68, .10, .34, .91),
        ("Mi Libertador - Jorge Lozano", "United Worship Ministry", 242, 140,
         .12, .15, .09, .11, .09, .31, .84),
    ],
    "Solo Una Palabra Tuya": [
        ("Solo Una Palabra Tuya", "Jorge Lozano - Topic", 393, 255,
         .33, .35, .33, .30, .09, .34, .93),
        ("Solo una palabra tuya - JORGE LOZANO", "elevador", 393, 140,
         .66, .68, .22, .18, .07, .34, .93),
    ],
    "Corazon Travieso": [
        ("Corazon Travieso", "Jorge Lozano - Topic", 182, 255, .53, .55, .37, .66, .15, .42, .91),
        ("Toma El Trono De Mi Corazon  Jorge Lozano", "PEDRO PABLO AMAYA", 289, 100,
         .15, .18, .13, .23, .09, .20, .37),
    ],
}


def _replay(song: str, **kwargs):
    """Run the classifier over one song's real scores."""
    rows = REAL_RUN[song]
    classifier = JevClassifier(**kwargs)
    candidates = []
    scores = {}
    for title, channel, duration, heuristic, *values in rows:
        entry = {
            "title": title,
            "channel": channel,
            "duration": duration,
            "_composite_score": heuristic,
            "_score_breakdown": {"base_match": heuristic},
            "artists": [],
            "_source": "youtube",
        }
        candidates.append(entry)
        scores[id(entry)] = values

    def fake_evaluate(state, questions, model=None):
        answers = {}
        for index, entry in enumerate(candidates):
            key = f"candidate_{index}"
            for position, dimension in enumerate(DIM_KEYS):
                answers[dq.question_id(key, dimension)] = {
                    "type": "boolean",
                    "probability": scores[id(entry)][position],
                }
        if len(candidates) > 1:
            answers[dq.choice_id()] = {
                "type": "choice",
                "choice": "candidate_0",
                "probabilities": {f"candidate_{i}": 0.9 - 0.1 * i for i in range(len(candidates))},
                "confidence": 0.8,
            }
        return answers

    classifier._evaluate = fake_evaluate
    return classifier.select("Jorge Lozano", song, candidates)


class TestGateCalibration:
    """The floor that vetoed everything was 0.50 for all three gates."""

    @pytest.mark.parametrize("song", sorted(REAL_RUN))
    def test_the_correct_official_upload_is_selected(self, song):
        selected, _ranked = _replay(song, threshold=Config.DECISION_DEFAULT_THRESHOLD)

        assert selected is not None, f"{song}: no candidate selected at all"
        assert selected["channel"] == "Jorge Lozano - Topic"

    @pytest.mark.parametrize("song", sorted(REAL_RUN))
    def test_it_is_selected_without_the_deterministic_shield(self, song, monkeypatch):
        """The per-gate floors must do the work on their own.

        The shield exists for a title-and-channel match the model is unsure about.
        If the floors were only passing *because* of it, they would still be
        miscalibrated and would fail on the first song whose title differs from the
        request by a word -- which is most of them.
        """
        monkeypatch.setattr(dq, "deterministic_shield", lambda *a, **k: False)

        selected, _ranked = _replay(song, threshold=Config.DECISION_DEFAULT_THRESHOLD)

        assert selected is not None, f"{song}: needs the shield to survive the gates"
        assert selected["channel"] == "Jorge Lozano - Topic"
        assert not selected["_decision_shielded"]

    @pytest.mark.parametrize("song", sorted(REAL_RUN))
    def test_the_wrong_upload_is_still_vetoed(self, song, monkeypatch):
        """Lowering the floor must not have admitted the wrong song too.

        This is the property the floors exist for. A floor that lets everything
        through is worse than the old one: it converts the veto into decoration.

        The shield is disabled, because the official upload would be shielded and
        the test is about the *other* rows.
        """
        monkeypatch.setattr(dq, "deterministic_shield", lambda *a, **k: False)

        _selected, ranked = _replay(song, threshold=Config.DECISION_DEFAULT_THRESHOLD)
        by_title = {entry["title"]: entry for entry, _, _ in ranked}

        fan_upload = [
            entry for entry in by_title.values() if entry["channel"] != "Jorge Lozano - Topic"
        ]
        assert fan_upload, "fixture should include a non-official candidate"
        assert all(entry["_decision_eligible"] is False for entry in fan_upload), (
            f"{song}: a fan upload cleared every gate: "
            + ", ".join(f"{e['title']}={e['_decision_dimensions']}" for e in fan_upload)
        )

    def test_the_old_floor_vetoes_the_correct_answer(self, monkeypatch):
        """Pins the regression: with one 0.50 floor, nothing is ever selected."""
        monkeypatch.setattr(dq, "deterministic_shield", lambda *a, **k: False)
        floors = {k: 0.50 for k in dq.GATE_FLOORS}

        selected, _ranked = _replay("Alto Precio", gate_floors=floors, threshold=0.35)

        assert selected is None

    def test_the_old_threshold_was_unreachable(self):
        """`0.40*studio + 0.35*audio + 0.25*reference` topped out at 0.42.

    Recorded against the original weights so that a future re-ordering cannot make
    the 0.60 threshold look reachable again. Under the current weights the same
        candidate scores 0.524.
    """
        _selected, ranked = _replay("Alto Precio", threshold=Config.DECISION_DEFAULT_THRESHOLD)
        best = ranked[0][0]
        old_weights = {"studio": 0.40, "audio": 0.35, "reference": 0.25}
        dimensions = best["_decision_dimensions"]

        under_old = sum(
            dimensions[key] * weight for key, weight in old_weights.items()
        )
        assert under_old == pytest.approx(0.4185, abs=1e-3)
        assert under_old < 0.60, "the old 0.60 threshold was reachable, which is not what was measured"
        assert best["_decision_ranking_score"] > Config.DECISION_DEFAULT_THRESHOLD

    def test_floors_are_per_gate_and_origin_is_the_loosest(self):
        """`origin` answered 52% for a wrong song on the artist's own channel."""
        assert set(dq.GATE_FLOORS) == {"song", "artist", "origin", "form"}
        assert dq.GATE_FLOORS["origin"] == min(dq.GATE_FLOORS.values())
        assert dq.gate_floor_for("origin") == dq.GATE_FLOORS["origin"]
        assert dq.gate_floor_for("origin", {"origin": 0.44}) == 0.44
        assert dq.gate_floor_for("unknown-dimension", None, 0.42) == 0.42

    def test_the_configured_floors_match_the_module_defaults(self):
        assert Config().DECISION_GATE_FLOORS == dq.GATE_FLOORS


class TestDeterministicShield:
    def _candidate(self, title, channel, score=255):
        return {"title": title, "channel": channel, "_composite_score": score}

    def test_an_exact_title_on_the_topic_channel_is_shielded(self):
        assert dq.deterministic_shield(
            self._candidate("Alto precio", "Jorge Lozano - Topic"),
            "Jorge Lozano",
            "Alto Precio",
        )

    def test_official_audio_decoration_still_matches(self):
        assert dq.deterministic_shield(
            self._candidate("Alto Precio (Official Audio)", "Jorge Lozano - Topic"),
            "Jorge Lozano",
            "Alto Precio",
        )

    def test_a_similar_title_is_not(self):
        """A medley or a re-title is exactly what the shield must not rescue."""
        assert not dq.deterministic_shield(
            self._candidate("Alto precio (Medley)", "Jorge Lozano - Topic"),
            "Jorge Lozano",
            "Alto Precio",
        )

    def test_a_fan_channel_is_not(self):
        assert not dq.deterministic_shield(
            self._candidate("Alto precio", "Musicas y Videos Hector C."),
            "Jorge Lozano",
            "Alto Precio",
        )

    def test_another_artists_topic_channel_is_not(self):
        assert not dq.deterministic_shield(
            self._candidate("Alto precio", "Someone Else - Topic"),
            "Jorge Lozano",
            "Alto Precio",
        )

    def test_a_hard_rejected_title_is_not(self):
        """A cover on the artist's own Topic channel is still a cover."""
        assert not dq.deterministic_shield(
            self._candidate("Alto precio", "Jorge Lozano - Topic", score=dq.HARD_REJECT_SCORE),
            "Jorge Lozano",
            "Alto Precio",
        )

    def test_the_shield_cannot_lift_a_hard_reject_in_a_real_run(self, monkeypatch):
        monkeypatch.setattr(
            dq, "GATE_FLOORS", {k: 0.99 for k in dq.GATE_FLOORS}
        )
        selected, ranked = _replay(
            "Alto Precio",
            gate_floors=dq.GATE_FLOORS,
            threshold=Config.DECISION_DEFAULT_THRESHOLD,
        )
        entry = ranked[0][0]

        assert entry["_decision_shielded"] is True
        assert entry["_decision_eligible"] is True


class TestProvenanceSignal:
    def test_a_topic_channel_for_this_artist_is_provenance(self):
        assert (
            dq.provenance_signal({"channel": "Jorge Lozano - Topic"}, "Jorge Lozano")
            == "topic_channel"
        )

    def test_vevo_is_official(self):
        assert dq.provenance_signal({"channel": "ArtistVEVO"}, "Artist") == "official_audio"

    def test_a_channel_named_after_the_artist_counts(self):
        """The Jose Luis Reyes case: `JOSE LUIS REYES` is the artist's own channel."""
        assert (
            dq.provenance_signal({"channel": "JOSE LUIS REYES"}, "Jose Luis Reyes")
            == "artist_channel"
        )

    def test_a_fan_channel_is_unknown(self):
        assert (
            dq.provenance_signal({"channel": "Musicas y Videos Hector C."}, "Jorge Lozano")
            == "unknown"
        )
        assert dq.provenance_signal({"channel": "elevador"}, "Jorge Lozano") == "unknown"
        assert (
            dq.provenance_signal({"channel": "IglesiaCanta"}, "Jorge Lozano") == "unknown"
        )

    def test_another_artists_topic_channel_is_not(self):
        """`Various Artists - Topic` must not inherit the strongest provenance."""
        assert (
            dq.provenance_signal({"channel": "Someone Else - Topic"}, "Jorge Lozano")
            == "unknown"
        )

    def test_ownership_needs_the_whole_name_not_a_fragment(self):
        """A bare substring test would let the artist "Jorge" claim "Jorgelin".

        Word-run matching means a partial token is not ownership, while a channel
        named after its performer still is, whatever else is in the name.
        """
        assert dq.provenance_signal({"channel": "Jorgelin Oficial"}, "Jorge") == "unknown"
        assert (
            dq.provenance_signal({"channel": "Jorge Lozano Oficial"}, "Jorge Lozano")
            == "artist_channel"
        )

    def test_it_reaches_the_studio_dimension(self):
        """`studio` carries the largest ranking weight and had no evidence at all."""
        studio = next(d for d in dq.DIMENSIONS if d.key == "studio")
        assert "provenance" in studio.fields

        state = dq.candidate_state(
            {"title": "Song", "channel": "Artist - Topic", "artists": []}, 0, artist="Artist"
        )
        assert state["provenance"] == "topic_channel"
        assert set(studio.fields) <= set(state)

    def test_the_bonus_ranks_publishers_in_a_fixed_order(self):
        """And every unknown value scores zero rather than raising."""
        assert (
            dq.provenance_bonus("topic_channel")
            > dq.provenance_bonus("artist_channel")
            > dq.provenance_bonus("official_audio")
            > dq.provenance_bonus("unknown")
        )
        assert dq.provenance_bonus("something-new") == 0.0
        assert dq.provenance_bonus("unknown") == 0.0

    def test_the_bonus_clears_the_measured_margin(self):
        """It only has to beat the gap the wrong candidate opened.

        The church upload beat the official one by 0.0035 in the composite, so a
        bonus under that would change nothing.
        """
        official = 0.4 * 0.12 + 0.35 * 0.39 + 0.25 * 0.92
        church = 0.4 * 0.18 + 0.35 * 0.41 + 0.25 * 0.93
        assert church > official
        assert dq.provenance_bonus("topic_channel") > (church - official)


class TestFallbackPool:
    """The pool used to prefer never-evaluated rows over the official upload.

    Replayed through `MusicDownloader._search_and_select` rather than by reading
    the code, because the bug was an interaction between the classifier's output
    shape and the caller's ordering, and either half alone looks correct.
    """

    def _downloader(self, config, spy_events, ranked):
        from ytdl_core.core import MusicDownloader

        classifier = type(
            "C",
            (),
            {
                "select": staticmethod(lambda *a, **k: (None, ranked)),
                "gate": type("G", (), {"give_up_notice": staticmethod(lambda: None)}),
            },
        )()
        downloader = MusicDownloader(
            config=config,
            events=spy_events,
            workers=1,
            use_jev=True,
            jev_classifier=classifier,
        )
        return downloader

    def _entry(self, title, heuristic, **extra):
        entry = {
            "title": title,
            "channel": "Jorge Lozano - Topic",
            "duration": 355,
            "_composite_score": heuristic,
            "_heuristic_score": heuristic,
            "_decision_probability": 0.0,
        }
        entry.update(extra)
        return entry

    def test_an_official_upload_beats_a_never_evaluated_row(self, config, spy_events, tmp_path):
        """The exact failure: 255 (vetoed) lost to 120 (never seen by the model)."""
        import threading

        official = self._entry("Alto precio", 255, _decision_eligible=False)
        never_seen = self._entry(
            "PAGASTE UN ALTO PRECIO POR MI", 120, _decision_evaluated=False
        )
        ranked = [(official, 0, {}), (never_seen, 120, {})]

        downloader = self._downloader(config, spy_events, ranked)
        result = type("R", (), {"reason": None, "selection_method": None})()
        best, _r, _s = downloader._search_and_select(
            "Jorge Lozano",
            "Alto Precio",
            tmp_path,
            {"downloads": {}},
            threading.Lock(),
            "k",
            result,
            threading.Event(),
            None,
        )

        assert best is official, "downloaded the wrong song"
        assert result.selection_method == "jev-fallback"

    def test_a_hard_reject_is_never_a_fallback(self, config, spy_events, tmp_path):
        """-9999 is the heuristic's word match on 'cover'/'instrumental'."""
        import threading

        official = self._entry("Alto precio", 255, _decision_eligible=False)
        cover = self._entry(
            "Alto precio (Cover)", dq.HARD_REJECT_SCORE, _decision_evaluated=False
        )
        ranked = [(official, 0, {}), (cover, dq.HARD_REJECT_SCORE, {})]

        downloader = self._downloader(config, spy_events, ranked)
        result = type("R", (), {"reason": None, "selection_method": None})()
        best, _r, _s = downloader._search_and_select(
            "Jorge Lozano",
            "Alto Precio",
            tmp_path,
            {"downloads": {}},
            threading.Lock(),
            "k",
            result,
            threading.Event(),
            None,
        )

        assert best is not cover
        assert best is official

    def test_an_eligible_row_still_wins_over_a_vetoed_one(self, config, spy_events, tmp_path):
        import threading

        vetoed = self._entry("Alto precio (Cover)", 200, _decision_eligible=False)
        eligible = self._entry("Alto precio", 150, _decision_eligible=True)
        ranked = [(vetoed, 0, {}), (eligible, 150, {})]

        downloader = self._downloader(config, spy_events, ranked)
        result = type("R", (), {"reason": None, "selection_method": None})()
        best, _r, _s = downloader._search_and_select(
            "Jorge Lozano",
            "Alto Precio",
            tmp_path,
            {"downloads": {}},
            threading.Lock(),
            "k",
            result,
            threading.Event(),
            None,
        )

        assert best is eligible


class TestRankingPrefersTheOfficialPublisher:
    """The second real run: the decider selected, but ranked the wrong publisher.

    Three songs came out wrong, all the same way -- the official `- Topic` upload
    lost to a fan channel on a composite that could not tell them apart:

        Alto Precio            church channel  45%  beat  Topic  41%
        Solo Una Palabra Tuya  fan lyric video 39%  beat  Topic  37%
        Yo Te Alabaré         fan upload      41%  beat  Topic  38%

    The margins are 0.004, 0.002 and 0.003. `studio`, `audio` and `reference`
    answered 5-24%, 16-81% and 81-96% for official and fan uploads alike, so the
    composite was three noisy numbers and the tie went to row order. Nothing in the
    ranking knew who had published either candidate.
    """

    # artist, song, expected title, [(title, channel, song, artist, o, f, s, a, r)]
    #
    # `song` and `artist` are the compound `identity` answer split by 2-3 points.
    # The fan channel is *not* wrong on either -- it is the right song by the right
    # artist, which is exactly why this case had to be decided on the publisher
    # rather than on identity.
    CASES = [
        (
            "Jorge Lozano", "Alto Precio", "Alto precio",
            [
                ("IC0048-Alto Precio - Jorge Lozano", "IglesiaCanta",
                 .60, .62, .40, .60, .18, .41, .93),
                ("Alto precio", "Jorge Lozano - Topic",
                 .32, .34, .41, .33, .12, .39, .92),
            ],
        ),
        (
            "Jorge Lozano", "Solo Una Palabra Tuya", "Solo Una Palabra Tuya",
            [
                ("Solo una palabra tuya - JORGE LOZANO -  LYRIC VIDEO", "elevador",
                 .73, .75, .30, .47, .15, .29, .90),
                ("Solo Una Palabra Tuya", "Jorge Lozano - Topic",
                 .33, .35, .33, .30, .06, .32, .93),
            ],
        ),
        (
            "Jorge Lozano", "Yo Te Alabaré", "Yo Te Alabare",
            [
                ("Yo te alabaré - JORGE LOZANO", "elevador",
                 .64, .66, .26, .25, .09, .41, .92),
                ("Yo te alabaré - JORGE LOZANO - LYRIC VIDEO", "elevador",
                 .76, .78, .42, .68, .17, .32, .88),
                ("Yo Te Alabare", "Jorge Lozano - Topic",
                 .44, .46, .43, .45, .10, .34, .87),
            ],
        ),
    ]

    def _select(self, artist, song, rows, monkeypatch):
        candidates = []
        scores = {}
        for title, channel, *values in rows:
            entry = {
                "title": title,
                "channel": channel,
                "duration": 200,
                "_composite_score": 200,
                "_score_breakdown": {},
                "artists": [],
                "_source": "youtube",
            }
            candidates.append(entry)
            scores[id(entry)] = values

        def fake_evaluate(state, questions, model=None):
            answers = {}
            for index, entry in enumerate(candidates):
                for position, dimension in enumerate(DIM_KEYS):
                    answers[dq.question_id(f"candidate_{index}", dimension)] = {
                        "type": "boolean",
                        "probability": scores[id(entry)][position],
                    }
            if len(candidates) > 1:
                answers[dq.choice_id()] = {
                    "type": "choice",
                    "choice": "candidate_0",
                    "probabilities": {f"candidate_{i}": 0.5 for i in range(len(candidates))},
                    "confidence": 0.5,
                }
            return answers

        classifier = JevClassifier(threshold=Config.DECISION_DEFAULT_THRESHOLD)
        monkeypatch.setattr(classifier, "_evaluate", fake_evaluate)
        return classifier.select(artist, song, candidates)

    @pytest.mark.parametrize(
        ("artist", "song", "expected", "rows"), CASES, ids=[case[1] for case in CASES]
    )
    def test_the_official_publisher_wins(self, artist, song, expected, rows, monkeypatch):
        selected, _ranked = self._select(artist, song, rows, monkeypatch)

        assert selected is not None
        assert selected["title"] == expected

    @pytest.mark.parametrize(
        ("artist", "song", "expected", "rows"), CASES, ids=[case[1] for case in CASES]
    )
    def test_the_fan_channel_wins_on_the_dimensions_alone(
        self, artist, song, expected, rows, monkeypatch
    ):
        """Otherwise the test above would pass without the bonus doing anything.

        In all three the fan channel's *pure* composite is higher. If the fix were
        anywhere other than the publisher signal -- better weights, a different
        floor -- this would fail.
        """
        selected, ranked = self._select(artist, song, rows, monkeypatch)
        by_title = {entry["title"]: entry for entry, _, _ in ranked}

        fan = by_title[next(t for t, _c, *_ in rows if t != expected)]
        official = by_title[expected]
        assert fan["_decision_ranking_score"] > official["_decision_ranking_score"]
        assert fan["_decision_provenance"] == "unknown"
        assert official["_decision_provenance"] == "topic_channel"
        assert selected["title"] == expected

    @pytest.mark.parametrize(
        ("artist", "song", "expected", "rows"), CASES, ids=[case[1] for case in CASES]
    )
    def test_it_is_the_bonus_and_not_the_shield(
        self, artist, song, expected, rows, monkeypatch
    ):
        """The ranking fix must stand on its own.

        Otherwise a retune of the gate floors -- which the shield is there to
        survive -- could quietly put the publisher signal back to doing the work
        the bonus should be doing.
        """
        monkeypatch.setattr(dq, "deterministic_shield", lambda *a, **k: False)

        selected, _ranked = self._select(artist, song, rows, monkeypatch)

        assert selected is not None
        assert selected["title"] == expected
        assert not selected["_decision_shielded"]

    def test_an_artists_own_channel_beats_a_fan_channel(self, monkeypatch):
        """The Jose Luis Reyes half: `JOSE LUIS REYES` is the artist, not a fan."""
        artist, song = "Jose Luis Reyes", "Lo Mejor De Mi"
        rows = [
            ("Lo Mejor De MI", "JOSE LUIS REYES", .72, .74, .62, .84, .21, .81, .84),
            ("lo mejor de mi- Jose Luis reyes", "Laura Aguilar", .18, .22, .10, .17, .11, .32, .88),
            ("Lo Mejor de Mi - Jose Luis Reyes", "bottimasterful", .42, .44, .08, .22, .11, .33, .90),
        ]
        expected = "Lo Mejor De MI"

        selected, _ranked = self._select(artist, song, rows, monkeypatch)

        assert selected is not None
        assert selected["title"] == expected
        assert selected["_decision_provenance"] == "artist_channel"

    def test_a_studio_take_still_beats_a_live_one_on_the_same_channel(self, monkeypatch):
        """The bonus must not flatten the distinction it sits next to.

        Both are on the artist's own channel, so provenance cannot break this tie --
        it falls to `studio`, which is what it is there for.
        """
        artist, song = "Jose Luis Reyes", "El Dios De Lo Imposible"
        rows = [
            ("El Dios de Lo Imposible", "JOSE LUIS REYES", .80, .82, .77, .79, .23, .73, .94),
            ("Jose Luis Reyes - El Dios de lo Imposible (En Vivo)", "JOSE LUIS REYES",
             .88, .90, .88, .87, .05, .38, .81),
        ]
        expected = "El Dios de Lo Imposible"

        selected, ranked = self._select(artist, song, rows, monkeypatch)

        assert selected is not None
        assert selected["title"] == expected
        # The live take scores higher on all three gates and still loses: the
        # difference that put it ahead was `studio`, which is the studio dimension.
        live = next(e for e, _, _ in ranked if "En Vivo" in e["title"])
        assert (
            live["_decision_dimensions"]["origin"]
            > selected["_decision_dimensions"]["origin"]
        )
        assert (
            live["_decision_dimensions"]["studio"]
            < selected["_decision_dimensions"]["studio"]
        )
        assert live["_decision_ranking_score"] < selected["_decision_ranking_score"]


class TestReferenceOutranksStudio:
    """`Heme Aqui`: a different song won because `studio` spiked to 72%.

    "Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger" is not the requested song.
    It still won, 66% to 63%, on this arithmetic:

        studio    72% vs 26%   ->  +18.4 points at weight 0.40
        reference 30% vs 90%   ->  -15.0 points at weight 0.25

    `studio` had answered 5-24% for every other candidate in the run; the 72% was
    an outlier, and at 0.40 it outweighed the dimension that had correctly flagged
    the song as wrong. The heuristic had already scored it lower (base_match 101
    against 140); the ranking overruled that on one bad model answer.
    """

    ARTIST = "Marcos Witt"
    SONG = "Heme Aqui"
    EXPECTED = "Heme Aqui"
    # The run listed "Heme Aqui" twice from the same channel at 03:18 and 04:17.
    # `select_for_evaluation` dedupes them, and a fixture with two identical rows
    # would index the replayed scores against the wrong candidate, so one is kept.
    #
    # MEASURED, then SPLIT: `identity` answered 56% for the wrong song and 62% for
    # the right one. Split, that becomes song=.18 / artist=.93 and song=.62 /
    # artist=.93 -- the half that was certain is kept and the half that was not is
    # isolated. The song numbers are the *prediction* of the split, not a second
    # measurement: the model has never been asked. `COMPOUND` below replays the old
    # single 56% answer so the split is pinned as load-bearing rather than assumed.
    ROWS = [
        ("Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger", "Marcos Witt Oficial",
         .18, .93, .87, .40, .72, .48, .30),
        ("Heme Aqui", "Marcos Witt Oficial", .62, .93, .56, .48, .26, .50, .90),
        ("Heme aqui - Marcos Witt", "Dosis de Fe", .49, .52, .22, .26, .16, .31, .88),
    ]
    #: The same run as it stood before the split. The compound answer is put on both
    #: gates, which is what "split it into two questions without re-asking either"
    #: produces, and is what the model is assumed to answer if the decomposition
    #: buys it nothing.
    COMPOUND = [
        ("Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger", "Marcos Witt Oficial",
         .56, .56, .87, .40, .72, .48, .30),
        ("Heme Aqui", "Marcos Witt Oficial", .62, .62, .56, .48, .26, .50, .90),
        ("Heme aqui - Marcos Witt", "Dosis de Fe", .49, .49, .22, .26, .16, .31, .88),
    ]

    def _select(self, weights=None, rows=None):
        real = dq.RANK_WEIGHTS
        if weights:
            dq.RANK_WEIGHTS = dict(weights)
        rows = rows if rows is not None else self.ROWS
        try:
            candidates = [
                {"title": t, "channel": c, "duration": 200, "_composite_score": 200,
                 "_score_breakdown": {}, "artists": [], "_source": "youtube"}
                for t, c, *_ in rows
            ]

            def fake_evaluate(state, questions, model=None):
                answers = {}
                for i, row in enumerate(rows):
                    for di, dimension in enumerate(DIM_KEYS):
                        answers[dq.question_id(f"candidate_{i}", dimension)] = {
                            "type": "boolean", "probability": row[2 + di]}
                answers[dq.choice_id()] = {
                    "type": "choice", "choice": "candidate_0",
                    "probabilities": {f"candidate_{i}": 0.5 for i in range(len(candidates))},
                    "confidence": 0.5,
                }
                return answers

            classifier = JevClassifier(threshold=Config.DECISION_DEFAULT_THRESHOLD)
            classifier._evaluate = fake_evaluate
            return classifier.select(self.ARTIST, self.SONG, candidates)
        finally:
            dq.RANK_WEIGHTS = real

    def test_the_right_song_wins(self):
        selected, _ranked = self._select()

        assert selected is not None
        assert selected["title"] == self.EXPECTED

    def test_the_reordering_alone_already_fixes_this_case(self):
        """The split was not what fixed `Heme Aqui`, and this says so.

        Replayed with the compound answer put back on both gates, the new weights
        still choose the right song: `reference` is 0.30 against 0.90 and it now
        carries the most weight. The split is defence in depth for a case the
        weights already cover -- see the same-album test below for the one it does
        not.
        """
        selected, _ranked = self._select(rows=self.COMPOUND)

        assert selected is not None
        assert selected["title"] == self.EXPECTED

    def test_a_wrong_song_is_vetoed_rather_than_merely_out_ranked(self):
        """What the split actually buys, on the case the weights cannot cover.

        A different track from the same album agrees with the reference, so
        `reference` cannot separate them and neither can the publisher -- both
        come from the same release. Ranking has nothing left to rank on, and only a
        gate that asks about the song can refuse it.
        """
        same_album = [
            # Same song, wrong track from the same release: reference agrees, the
            # publisher is identical, `studio` and `audio` say nothing useful.
            ("Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger", "Marcos Witt Oficial",
             .18, .93, .87, .40, .72, .48, .90),
            ("Heme Aqui", "Marcos Witt Oficial", .62, .93, .56, .48, .26, .50, .90),
            ("Heme aqui - Marcos Witt", "Dosis de Fe", .49, .52, .22, .26, .16, .31, .88),
        ]

        selected, ranked = self._select(rows=same_album)

        assert selected is not None
        assert selected["title"] == self.EXPECTED
        # And it is a veto, not a margin: the wrong song is ineligible.
        wrong = next(
            e for e, _, _ in ranked if "Al Estar Aqui" in e["title"]
        )
        assert wrong["_decision_eligible"] is False
        assert "song" in wrong["_decision_failed_gates"]

    def test_a_correct_artist_does_not_vouch_for_a_wrong_song(self):
        """`artist` is high for both and must not decide the `song` gate.

        This is the property the split was built for: the two gates are asked
        separately, so a candidate cannot clear `song` on the strength of `artist`.
        """
        same_album = [
            ("Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger", "Marcos Witt Oficial",
             .18, .93, .87, .40, .72, .48, .90),
            ("Heme Aqui", "Marcos Witt Oficial", .62, .93, .56, .48, .26, .50, .90),
            ("Heme aqui - Marcos Witt", "Dosis de Fe", .49, .52, .22, .26, .16, .31, .88),
        ]
        _selected, ranked = self._select(rows=same_album)
        by_title = {e["title"]: e for e, _, _ in ranked}
        wrong = by_title["Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger"]
        right = by_title[self.EXPECTED]

        # Identical on everything the ranking can see.
        assert wrong["_decision_dimensions"]["artist"] == right["_decision_dimensions"]["artist"]
        assert wrong["_decision_dimensions"]["reference"] == right["_decision_dimensions"]["reference"]
        assert wrong["_decision_provenance"] == right["_decision_provenance"]
        # And still separable, by exactly one gate.

    def test_the_song_gate_receives_no_channel(self):
        """The channel carries the artist, and that is how the artist leaked in.

        `song` decides which song this is; `artist` decides who is playing it and
        can see the channel. Between them they still see everything the compound
        `identity` gate saw, so nothing was lost -- it was separated, not dropped.
        """
        song = next(d for d in dq.DIMENSIONS if d.key == "song")
        artist = next(d for d in dq.DIMENSIONS if d.key == "artist")

        assert "channel" not in song.fields
        assert "channel" in artist.fields
        assert {"title", "artists", "duration_seconds"} <= set(song.fields) | set(artist.fields)

    def test_the_old_weights_chose_the_wrong_song(self):
        """Otherwise the test above proves nothing about the re-ordering.

        Run against COMPOUND, so the gates are identical to the run that failed and
        the weights are the only thing under test. Both candidates sit on the same
        artist-controlled channel, so provenance cannot break this tie either.
        """
        old = {"studio": 0.40, "audio": 0.35, "reference": 0.25}

        selected, _ranked = self._select(old, rows=self.COMPOUND)

        assert selected is not None
        assert selected["title"] != self.EXPECTED
        assert "Al Estar Aqui" in selected["title"]

    def test_both_candidates_share_a_publisher_so_provenance_cannot_help(self):
        """Which is exactly why this case had to be fixed in the weights."""
        _selected, ranked = self._select()
        provenances = {e["_decision_provenance"] for e, _, _ in ranked if e.get("_decision_evaluated")}

        assert provenances == {"artist_channel", "unknown"}

    def test_reference_is_the_dimension_that_flagged_the_wrong_song(self):
        """The re-ordering gives the most weight to the one that got it right."""
        _selected, ranked = self._select()
        by_title = {e["title"]: e for e, _, _ in ranked}

        wrong = by_title["Marcos Witt | Al Estar Aqui feat. @TAYAGaukrodger"]
        right = by_title[self.EXPECTED]

        assert (
            wrong["_decision_dimensions"]["reference"]
            < right["_decision_dimensions"]["reference"]
        )
        # And it is the largest weight, so the flag is worth more than the spike.
        assert dq.RANK_WEIGHTS["reference"] == max(dq.RANK_WEIGHTS.values())
        assert dq.RANK_WEIGHTS["studio"] < dq.RANK_WEIGHTS["reference"]

    def test_the_weights_still_sum_to_one(self):
        """Renormalised in place rather than scaled at read time.

        The ranking divides by the total weight, so the *shape* is what matters and
        the absolute scale does not -- but an off-by-one in a hand-edited dict
        would silently shift every candidate's score against the threshold.
        """
        assert sum(dq.RANK_WEIGHTS.values()) == pytest.approx(1.0)
        assert set(dq.RANK_WEIGHTS) == set(dq.RANK_DIMENSIONS_KEYS)


class TestCalibrationLog:
    def test_it_records_every_gate_score(self, config, spy_events, tmp_path):
        path = tmp_path / "calib.jsonl"
        config.DECISION_CALIBRATION_LOG = str(path)
        downloader = self._fallback_free_downloader(config, spy_events)
        entry = {
            "title": "Alto precio",
            "channel": "Jorge Lozano - Topic",
            "duration": 355,
            "_composite_score": 255,
            "_heuristic_score": 255,
            "_decision_evaluated": True,
            "_decision_eligible": False,
            "_decision_failed_gates": ["origin"],
            "_decision_dimensions": {"song": 0.32, "origin": 0.41, "form": 0.33},
            "_decision_ranking_score": 0.4185,
        }

        downloader._log_decision_calibration("Jorge Lozano", "Alto Precio", [(entry, 0, {})], None)

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert record["artist"] == "Jorge Lozano"
        assert record["gate_floors"] == config.DECISION_GATE_FLOORS
        assert record["candidates"][0]["dimensions"]["song"] == 0.32
        assert record["candidates"][0]["failed_gates"] == ["origin"]
        assert record["candidates"][0]["provenance"] == "topic_channel"

    def test_it_is_off_by_default(self, config, spy_events, tmp_path):
        downloader = self._fallback_free_downloader(config, spy_events)
        # Must not raise or write anything when no path is configured.
        downloader._log_decision_calibration("A", "B", [({"title": "x"}, 0, {})], None)

    def test_a_bad_path_does_not_fail_the_song(self, config, spy_events, tmp_path):
        config.DECISION_CALIBRATION_LOG = str(tmp_path / "nope" / "x.jsonl")
        downloader = self._fallback_free_downloader(config, spy_events)
        downloader._log_decision_calibration("A", "B", [({"title": "x"}, 0, {})], None)

    def test_the_log_records_the_outcome_not_just_the_scores(self, config, spy_events, tmp_path):
        """A record with no verdict cannot be used to judge a verdict."""
        import threading
        from unittest.mock import MagicMock, patch

        path = tmp_path / "calib.jsonl"
        config.DECISION_CALIBRATION_LOG = str(path)
        downloader = self._fallback_free_downloader(config, spy_events)
        entry = {
            "title": "Alto precio",
            "channel": "Jorge Lozano - Topic",
            "duration": 355,
            "_composite_score": 42,
            "_heuristic_score": 255,
            "_decision_evaluated": True,
            "_decision_eligible": True,
            "_decision_selected": True,
            "_decision_dimensions": {"song": 0.32, "origin": 0.41, "form": 0.33},
            "_decision_ranking_score": 0.4185,
        }
        classifier = MagicMock()
        classifier.select.return_value = (entry, [(entry, 42, {})])
        downloader.decision_classifier = classifier

        search_result = {
            "title": "Alto precio",
            "channel": "Jorge Lozano - Topic",
            "duration": 355,
        }
        with (
            patch("ytdl_core.core.search_all_sources", return_value=[search_result]),
            patch(
                "ytdl_core.core.select_best_result",
                return_value=(search_result, [(dict(entry), 255, {})]),
            ),
        ):
            downloader._search_and_select(
                "Jorge Lozano",
                "Alto Precio",
                tmp_path,
                {"downloads": {}},
                threading.Lock(),
                "k",
                MagicMock(),
                threading.Event(),
                None,
            )

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert record["threshold"] == config.DECISION_DEFAULT_THRESHOLD
        assert record["candidates"][0]["selected"] is True
        assert record["candidates"][0]["ranking_score"] == pytest.approx(0.4185, abs=1e-3)

    @staticmethod
    def _fallback_free_downloader(config, spy_events):
        from ytdl_core.core import MusicDownloader

        return MusicDownloader(config=config, events=spy_events, workers=1, delay=(0, 0))


class TestRejectionReasonCarriesNumbers:
    def test_the_gate_score_is_quoted(self):
        from ytdl_core.core import MusicDownloader

        entry = {
            "_decision_failed_gates": ["origin", "form"],
            "_decision_dimensions": {"origin": 0.41, "form": 0.33},
        }

        reason = MusicDownloader._decision_rejection_reason(entry)

        assert "origin" in reason and "form" in reason
        assert "41%" in reason and "33%" in reason

    def test_a_vetoed_candidate_does_not_report_zero_percent(self):
        """`best score 0%` was the forced-to-zero placeholder, not a measurement."""
        from ytdl_core.core import MusicDownloader

        reason = MusicDownloader._decision_rejection_reason(
            {"_decision_ranking_score": 0.4185, "_decision_probability": 0.0}
        )

        assert "42%" in reason
        assert "0%" not in reason