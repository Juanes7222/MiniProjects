from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import decision_questions as dq
from .ratelimit import CircuitBreaker, full_jitter_backoff


class JevEvaluationError(RuntimeError):
    pass


class DecisionGate:
    """Admission control and circuit breaking for a decision-model server.

    A decision model is a *shared, single-device* resource. Every song in the
    batch needs one, so without a gate N pipeline workers each post a full
    candidate set at once and queue behind each other on one GPU. Two things then
    go wrong, and both are silent:

    * requests time out while merely waiting their turn, so songs fall back to
      the heuristic for reasons that have nothing to do with the judgment;
    * each timeout is retried, so the provider costs a multiple of its own
      latency per song -- and with a thousand songs that is hours of waiting for
      a verdict that was never going to arrive.

    So: admit a bounded number of requests at once, and once the provider has
    failed repeatedly, stop asking for the rest of the run and say so once. A
    provider that is down should cost one report, not a timeout per song.
    """

    def __init__(
        self,
        *,
        max_in_flight: int = 1,
        failure_threshold: int = 3,
        cooldown_seconds: float = 120.0,
    ) -> None:
        self.max_in_flight = max(1, int(max_in_flight))
        self.failure_threshold = max(1, int(failure_threshold))
        self._semaphore = threading.BoundedSemaphore(self.max_in_flight)
        self._breaker = CircuitBreaker(
            cooldown_seconds=cooldown_seconds, max_cooldown_seconds=1800.0
        )
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._reported = False
        self.successes = 0
        self.failures = 0

    def admit(self) -> bool:
        """Take an admission slot, or return False if the provider is down."""
        if not self._breaker.allow():
            return False
        self._semaphore.acquire()
        # The breaker may have opened while we waited for a slot.
        if not self._breaker.allow():
            self._semaphore.release()
            return False
        return True

    def release(self) -> None:
        self._semaphore.release()

    def record_success(self) -> None:
        with self._lock:
            self.successes += 1
            self._consecutive_failures = 0
        self._breaker.record_success()

    def record_failure(self) -> bool:
        """Record a failure. True the first time the provider is given up on."""
        newly_tripped = False
        with self._lock:
            self.failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                newly_tripped = True
        if newly_tripped:
            self._breaker.trip()
        return newly_tripped

    def give_up_notice(self) -> Optional[str]:
        """Why the provider was abandoned, or None while it still looks healthy."""
        with self._lock:
            if not self._reported and self._consecutive_failures >= self.failure_threshold:
                self._reported = True
                return (
                    f"{self._consecutive_failures} consecutive decision-model failures; "
                    "skipping it for the rest of this run and using the heuristic "
                    "ranking. The selection quality is unaffected -- the model is a "
                    "second opinion, not the primary ranker."
                )
            return None


class PersistentBridge:
    """A long-lived ``node tools/jev.mts --server`` process.

    The one-shot form spawns Node and re-loads the runtime for every song. That
    is a fixed cost per candidate set, paid on top of a network round trip that
    dominates it anyway -- pure overhead repeated once per song. Keeping one
    process alive for the run removes it.

    Every failure mode degrades to the one-shot path rather than raising: a
    bridge that will not start, dies mid-run, or answers something unusable is
    not a reason to fail the song.
    """

    def __init__(self, script: Path, project_root: Path, node_command: str) -> None:
        self.script = script
        self.project_root = project_root
        self.node_command = node_command
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._disabled = False

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def _ensure_process(self) -> Optional[subprocess.Popen[str]]:
        if self._disabled:
            return None
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if process is not None:
            # Died between calls; fall back rather than respawn-storming.
            self._disabled = True
            return None
        try:
            self._process = subprocess.Popen(
                [self.node_command, str(self.script), "--server"],
                cwd=self.project_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError:
            self._disabled = True
            return None
        return self._process

    def request(self, payload: dict[str, Any], timeout: float) -> Optional[dict[str, Any]]:
        """Send one request and read one response. None means "fall back"."""
        with self._lock:
            process = self._ensure_process()
            if process is None or process.stdin is None or process.stdout is None:
                return None
            try:
                process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (OSError, ValueError):
                self._shutdown()
                return None
        if not line:
            self._shutdown()
            return None
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(response, dict) or "error" in response:
            return None
        return response

    def _shutdown(self) -> None:
        process = self._process
        self._process = None
        self._disabled = True
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._disabled = False
            self._shutdown()


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
        max_in_flight: int = 1,
        failure_threshold: int = 2,
        max_questions: int = 32,
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
        self.max_questions = max(1, int(max_questions))
        # The hosted Jev bridge goes through Vercel AI Gateway, which spells the
        # yes/no primitive `boolean` and answers under `probability`. The native
        # TypeSafe endpoint spells it `noul`. The judgments are identical.
        self.dialect = dq.GATEWAY_DIALECT
        self.bridge = PersistentBridge(
            self.project_root / "tools" / "jev.mts", self.project_root, node_command
        )
        # A decision server is a shared single-device resource; admission and
        # give-up live on the provider so they can be tuned per provider.
        self.gate = DecisionGate(
            max_in_flight=max_in_flight, failure_threshold=failure_threshold
        )

    @staticmethod
    def _question_count(candidate_count: int) -> int:
        """How many questions a candidate count costs: one per dimension, plus the choice."""
        return max(0, candidate_count) * len(dq.DIMENSIONS) + 1

    def _trim_to_question_budget(
        self, candidates: list[dict], keep: list[int]
    ) -> list[int]:
        """Drop the lowest-scoring candidates until the request fits the budget."""
        budget = max(1, int(self.max_questions))
        if self._question_count(len(keep)) <= budget or not keep:
            return keep

        # keep is already ordered best-first by select_for_evaluation, so the
        # tail is exactly the candidates worth giving up.
        affordable = max(1, (budget - 1) // max(1, len(dq.DIMENSIONS)))
        return keep[:affordable]

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

        # Then bound the actual cost. Questions scale as candidates x dimensions,
        # so a candidate cap is only a proxy: at 12 candidates plus headroom this
        # reached 97 questions for one song, which is minutes of generation
        # before any download starts -- and multiplied again by ``runs``.
        # Trimming to a question budget keeps latency predictable regardless of
        # how the dimension list evolves, and drops the *worst-scoring*
        # candidates, which are the ones least likely to win.
        keep = self._trim_to_question_budget(candidates, keep)
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

        # Prefer the long-lived bridge: it amortises Node's start-up across the
        # whole run. Anything it cannot answer falls through to a fresh process,
        # which is the behaviour that has always worked.
        bridged = self.bridge.request(request, self.timeout_seconds)
        if bridged is not None:
            answers = bridged.get("answers")
            if isinstance(answers, dict):
                return answers
        elif not self.bridge.enabled:
            pass  # bridge retired; one-shot is now the only path

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
                    time.sleep(full_jitter_backoff(attempt + 1, base=0.5, cap=5.0))
                    continue
                raise JevEvaluationError("Jev evaluation timed out after 3 attempts")

            if completed.returncode == 0:
                break

            stderr = completed.stderr
            if self._is_retryable_error(stderr) and attempt < 2:
                time.sleep(full_jitter_backoff(attempt + 1, base=0.5, cap=5.0))
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
