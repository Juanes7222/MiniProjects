from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from . import decision_questions as dq


class JevEvaluationError(RuntimeError):
    pass


class JevClassifier:
    """Scores candidates with a System One decision model.

    Transport only. Every judgment -- what is asked, how answers combine, what
    vetoes a candidate -- lives in :mod:`ytdl_core.decision_questions`.
    """

    provider_name = "Jev"

    def __init__(
        self,
        project_root: Path | None = None,
        node_command: str = "node",
        threshold: float = 0.60,
        timeout_seconds: int = 60,
        gate_floor: float = dq.GATE_FLOOR,
        stable_spread: float = dq.STABLE_SPREAD,
        max_candidates: int = dq.MAX_EVALUATED_CANDIDATES,
        eval_headroom: int = dq.EVALUATION_HEADROOM,
    ) -> None:
        package_root = Path(__file__).resolve().parent
        source_root = package_root.parent
        default_root = (
            source_root if (source_root / "tools" / "jev.mts").is_file() else package_root
        )
        self.project_root = project_root or default_root
        self.node_command = node_command
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self.gate_floor = gate_floor
        self.stable_spread = stable_spread
        self.max_candidates = max_candidates
        self.eval_headroom = eval_headroom
        # The hosted Jev bridge goes through Vercel AI Gateway, which spells the
        # yes/no primitive `boolean` and answers under `probability`. The native
        # TypeSafe endpoint spells it `noul`. The judgments are identical.
        self.dialect = dq.GATEWAY_DIALECT

    def select(
        self,
        artist: str,
        song: str,
        candidates: list[dict],
        reference_metadata: dict | None = None,
        runs: int = 1,
    ) -> tuple[dict | None, list[tuple[dict, int, dict[str, int]]]]:
        if not candidates:
            return None, []
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
            raise JevEvaluationError(f"{self.provider_name} runs must be at least 1")

        # The heuristic has already ruled out some candidates; spending model
        # tokens on them costs time and, in a real run, let a model score of 87
        # overrule a correct "instrumental" rejection. Everything the model does
        # not see keeps its heuristic score in the report below.
        keep = dq.select_for_evaluation(candidates, self.max_candidates, self.eval_headroom)
        evaluated_candidates = [candidates[index] for index in keep]
        skipped = [candidates[index] for index in range(len(candidates)) if index not in set(keep)]

        state, candidate_states, questions = dq.build_state_and_questions(
            artist, song, evaluated_candidates, reference_metadata, self.dialect
        )
        model = getattr(self, "model", None)

        # samples[(candidate_key, dimension)] -> list of probabilities, one per run
        samples: dict[tuple[str, str], list[float]] = {}
        choice_probs: dict[str, list[float]] = {}
        choice_confidences: list[float] = []
        for _ in range(runs):
            answers = self._evaluate(state, questions, model)
            for candidate_state in candidate_states:
                key = candidate_state["key"]
                for dimension in dq.DIMENSIONS:
                    qid = dq.question_id(key, dimension.key)
                    samples.setdefault((key, dimension.key), []).append(
                        self._probability(answers.get(qid), qid)
                    )
            if dq.choice_id() in questions:
                choice = answers.get(dq.choice_id())
                if isinstance(choice, dict):
                    # A Choice answer spreads probability over the options and
                    # reports its own confidence -- a second axis on top of the
                    # Noul scores, which say what each candidate is but not how
                    # decisively the model separated them.
                    probs = choice.get("probabilities")
                    if isinstance(probs, dict):
                        for key, value in probs.items():
                            choice_probs.setdefault(str(key), []).append(
                                self._scalar(value, f"{dq.choice_id()}.{key}")
                            )
                    confidence = choice.get("confidence")
                    if confidence is not None:
                        choice_confidences.append(
                            self._scalar(confidence, f"{dq.choice_id()}.confidence")
                        )

        evaluated: list[tuple[dict, int, dict[str, int]]] = []
        for candidate_state, candidate in zip(candidate_states, evaluated_candidates, strict=True):
            key = candidate_state["key"]
            dimension_scores = {
                dimension.key: self._mean(samples[(key, dimension.key)])
                for dimension in dq.DIMENSIONS
            }
            dimension_spread = {
                dimension.key: self._spread(samples[(key, dimension.key)])
                for dimension in dq.DIMENSIONS
            }
            entry = dict(candidate)
            heuristic_score = int(entry.get("_composite_score") or 0)
            heuristic_breakdown = dict(entry.get("_score_breakdown") or {})

            failed_gates = [
                dimension.key
                for dimension in dq.GATE_DIMENSIONS
                if dimension_scores[dimension.key] < self.gate_floor
            ]
            eligible = not failed_gates

            if eligible:
                score = self._ranking_score(dimension_scores)
            else:
                # A vetoed candidate keeps its ranking score visible for debugging
                # but is pushed below the threshold so it can never be selected.
                score = 0.0

            spread = max(dimension_spread.values()) if dimension_spread else 0.0
            stable = spread <= self.stable_spread
            choice_samples = choice_probs.get(key) or []
            choice_probability = self._mean(choice_samples) if choice_samples else None
            confidence = self._mean(choice_confidences) if choice_confidences else None

            entry["_heuristic_score"] = heuristic_score
            entry["_heuristic_breakdown"] = heuristic_breakdown
            entry["_decision_provider"] = self.provider_name
            entry["_decision_contract"] = dq.QUESTION_CONTRACT_VERSION
            entry["_decision_evaluated"] = True
            entry["_decision_score"] = score
            entry["_decision_probability"] = score
            entry["_decision_eligible"] = eligible
            entry["_decision_failed_gates"] = failed_gates
            entry["_decision_dimensions"] = dimension_scores
            entry["_decision_dimension_spread"] = dimension_spread
            entry["_decision_spread"] = spread
            entry["_decision_stable"] = stable
            entry["_decision_choice_probability"] = choice_probability
            entry["_decision_confidence"] = confidence
            entry["_decision_samples"] = [
                self._mean(samples[(key, dimension.key)]) for dimension in dq.DIMENSIONS
            ]
            entry["_decision_runs"] = runs
            entry["_decision_min"] = min(
                (v for values in samples.values() for v in values), default=0.0
            )
            entry["_decision_max"] = max(
                (v for values in samples.values() for v in values), default=0.0
            )
            entry["_decision_threshold"] = self.threshold
            entry["_decision_selected"] = False

            decision_score = int(round(score * 100))
            entry["_composite_score"] = decision_score
            breakdown: dict[str, int] = {
                key_name: int(round(value * 100)) for key_name, value in dimension_scores.items()
            }
            entry["_score_breakdown"] = {**heuristic_breakdown, **breakdown}
            evaluated.append((entry, decision_score, entry["_score_breakdown"]))

        # Unstable answers sort below stable ones: an unstable high score is not
        # evidence, it is a coin flip that happened to land high.
        evaluated.sort(
            key=lambda item: (
                item[0]["_decision_eligible"],
                item[0]["_decision_stable"],
                item[1],
                item[0].get("_heuristic_score", 0),
            ),
            reverse=True,
        )

        # Candidates the model never saw keep their heuristic rank in the report, below
        # everything evaluated, so the caller can still fall back to them.
        evaluated.extend(self._unevaluated(skipped))

        best = evaluated[0][0] if evaluated else None
        if (
            best is None
            or not best.get("_decision_eligible", True)
            or float(best.get("_decision_probability") or 0) < self.threshold
        ):
            return None, evaluated

        selected = dict(best)
        selected["_decision_selected"] = True
        selected_score, selected_breakdown = evaluated[0][1], evaluated[0][2]
        evaluated[0] = (selected, selected_score, selected_breakdown)
        return selected, evaluated

    @staticmethod
    def _unevaluated(candidates: list[dict]) -> list[tuple[dict, int, dict[str, int]]]:
        """Report entries for candidates the model was never asked about."""
        rows: list[tuple[dict, int, dict[str, int]]] = []
        for candidate in candidates:
            entry = dict(candidate)
            entry["_decision_evaluated"] = False
            heuristic_score = int(entry.get("_composite_score") or 0)
            entry["_heuristic_score"] = heuristic_score
            entry["_heuristic_breakdown"] = dict(entry.get("_score_breakdown") or {})
            entry["_composite_score"] = heuristic_score
            breakdown = entry["_score_breakdown"]
            rows.append((entry, heuristic_score, breakdown))
        return rows

    @staticmethod
    def _ranking_score(dimension_scores: dict[str, float]) -> float:
        """Weighted mean of the ranking dimensions.

        Gates are not in here: they already decided eligibility, and folding them
        in would double-count the same evidence.
        """
        total_weight = sum(dq.RANK_WEIGHTS.values())
        if total_weight <= 0:
            return 0.0
        return (
            sum(dimension_scores[key] * weight for key, weight in dq.RANK_WEIGHTS.items())
            / total_weight
        )

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _spread(values: list[float]) -> float:
        return max(values) - min(values) if len(values) > 1 else 0.0

    def _probability(self, answer: Any, question_id: str) -> float:
        """Read a Noul answer: the value the model assigned to 'yes'."""
        value = None
        if isinstance(answer, dict):
            if "noul" in answer:
                value = answer.get("noul")
            elif "probability" in answer:
                value = answer.get("probability")
        if value is None:
            raise JevEvaluationError(
                f"{self.provider_name} returned no probability for {question_id}"
            )
        return self._scalar(value, question_id)

    def _scalar(self, value: Any, question_id: str) -> float:
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise JevEvaluationError(
                f"{self.provider_name} returned an invalid probability for {question_id}"
            ) from exc
        if isinstance(value, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise JevEvaluationError(
                f"{self.provider_name} returned an invalid probability for {question_id}"
            )
        return probability

    def _evaluate(
        self, state: dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        script = self.project_root / "tools" / "jev.mts"
        if not script.is_file():
            raise JevEvaluationError("Jev bridge script was not found")

        request: dict[str, Any] = {"state": state, "questions": questions}
        if model:
            request["model"] = model

        for attempt in range(3):
            try:
                completed = subprocess.run(
                    [self.node_command, str(script)],
                    cwd=self.project_root,
                    input=json.dumps(request, ensure_ascii=False),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise JevEvaluationError("Node.js is required for --jev") from exc
            except subprocess.TimeoutExpired:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise JevEvaluationError("Jev evaluation timed out after 3 attempts")

            if completed.returncode == 0:
                break

            stderr = completed.stderr
            if self._is_retryable_error(stderr) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise JevEvaluationError(self._bridge_error(stderr))

        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise JevEvaluationError("Jev bridge returned invalid JSON") from exc
        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict):
            raise JevEvaluationError("Jev bridge returned no answers")
        return answers

    @staticmethod
    def _bridge_error(stderr: str) -> str:
        lower = stderr.lower()
        if "valid credit card" in lower:
            return "AI Gateway requires a valid credit card before it can service Jev requests"
        if "unauthenticated" in lower or "api key" in lower:
            return "AI Gateway authentication failed"
        if "free tier" in lower or "restrictedmodelserror" in lower:
            return "AI Gateway account is on the free tier, which cannot reach Jev; add credits"
        if "no such model" in lower or "model not found" in lower:
            return "Jev model is not available for this AI Gateway team"
        if "invalid discriminator" in lower or "unprocessable" in lower:
            return "AI Gateway rejected the question contract as malformed"
        if "rate limit" in lower or "429" in lower:
            return "AI Gateway rate limit reached while evaluating Jev"
        if "timed out" in lower or "timeout" in lower:
            return "Jev evaluation timed out"
        if "502" in lower or "503" in lower or "504" in lower or "internal server error" in lower:
            return "AI Gateway temporarily failed while evaluating Jev"
        first_line = next((line.strip() for line in stderr.splitlines() if line.strip()), "")
        if first_line:
            return f"Jev bridge failed: {first_line[:180]}"
        return "Jev bridge failed with an unknown error"

    @staticmethod
    def _is_retryable_error(stderr: str) -> bool:
        lower = stderr.lower()
        return any(
            marker in lower
            for marker in (
                "rate limit",
                "429",
                "502",
                "503",
                "504",
                "internal server error",
                "temporarily",
                "timeout",
                "timed out",
                "econnreset",
                "socket hang up",
            )
        )
