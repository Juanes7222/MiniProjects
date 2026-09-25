from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


class JevEvaluationError(RuntimeError):
    pass


class JevClassifier:
    provider_name = "Jev"
    def __init__(
        self,
        project_root: Path | None = None,
        node_command: str = "node",
        threshold: float = 0.60,
        timeout_seconds: int = 60,
    ) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.node_command = node_command
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds

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

        candidate_payloads = [
            self._candidate_payload(candidate, index)
            for index, candidate in enumerate(candidates)
        ]
        payload = {
            "state": {
                "target": {
                    "artist": artist,
                    "song": song,
                    "reference_metadata": self._reference_metadata(reference_metadata),
                },
                "candidates": candidate_payloads,
            },
            "candidates": [
                {
                    "key": candidate["key"],
                    "label": self._candidate_label(candidate),
                }
                for candidate in candidate_payloads
            ],
        }
        probabilities: dict[str, list[float]] = {
            candidate["key"]: [] for candidate in candidate_payloads
        }
        for _ in range(runs):
            answers = self._evaluate(payload)
            for candidate in candidate_payloads:
                probabilities[candidate["key"]].append(
                    self._answer_probability(answers.get(candidate["key"]))
                )

        evaluated: list[tuple[dict, int, dict[str, int]]] = []
        for candidate_payload, candidate in zip(candidate_payloads, candidates, strict=True):
            samples = probabilities[candidate_payload["key"]]
            probability = sum(samples) / len(samples)
            entry = dict(candidate)
            heuristic_score = int(entry.get("_composite_score") or 0)
            breakdown = dict(entry.get("_score_breakdown") or {})
            decision_score = int(round(probability * 100))
            entry["_heuristic_score"] = heuristic_score
            entry["_heuristic_breakdown"] = breakdown
            entry["_decision_provider"] = self.provider_name
            entry["_decision_probability"] = probability
            entry["_decision_samples"] = samples
            entry["_decision_runs"] = len(samples)
            entry["_decision_min"] = min(samples)
            entry["_decision_max"] = max(samples)
            entry["_decision_threshold"] = self.threshold
            entry["_decision_selected"] = False
            entry["_composite_score"] = decision_score
            entry["_score_breakdown"] = {
                **breakdown,
                "decision_probability": decision_score,
            }
            evaluated.append((entry, decision_score, entry["_score_breakdown"]))

        evaluated.sort(
            key=lambda item: (item[1], item[0].get("_heuristic_score", 0)), reverse=True
        )
        if not evaluated or evaluated[0][0]["_decision_probability"] < self.threshold:
            return None, evaluated

        selected = dict(evaluated[0][0])
        selected["_decision_selected"] = True
        selected_score, selected_breakdown = evaluated[0][1], evaluated[0][2]
        evaluated[0] = (selected, selected_score, selected_breakdown)
        return selected, evaluated

    def _answer_probability(self, answer: Any) -> float:
        if not isinstance(answer, dict):
            raise JevEvaluationError(f"{self.provider_name} returned an invalid answer")
        if "probability" in answer:
            return self._probability(answer.get("probability"))
        if "noul" in answer:
            return self._probability(answer.get("noul"))
        raise JevEvaluationError(f"{self.provider_name} returned an invalid probability answer")

    def _evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        script = self.project_root / "tools" / "jev.mts"
        if not script.is_file():
            raise JevEvaluationError("Jev bridge script was not found")

        for attempt in range(3):
            try:
                completed = subprocess.run(
                    [self.node_command, str(script)],
                    cwd=self.project_root,
                    input=json.dumps(payload, ensure_ascii=False),
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
        if "no such model" in lower or "model not found" in lower:
            return "Jev model is not available for this AI Gateway team"
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

    @staticmethod
    def _reference_metadata(metadata: dict | None) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        keys = ("album", "year", "genre", "track_num", "mb_id", "release_id")
        return {key: metadata[key] for key in keys if metadata.get(key) not in (None, "")}

    @staticmethod
    def _first_value(candidate: dict, *keys: str) -> Any:
        for key in keys:
            value = candidate.get(key)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _candidate_payload(candidate: dict, index: int) -> dict[str, Any]:
        return {
            "key": f"candidate_{index}",
            "title": str(candidate.get("title") or ""),
            "channel": str(candidate.get("channel") or candidate.get("uploader") or ""),
            "artists": candidate.get("artists") or [],
            "duration_seconds": candidate.get("duration"),
            "source": str(candidate.get("_source") or "unknown"),
            "heuristic_score": candidate.get("_composite_score", 0),
            "description": str(candidate.get("description") or "")[:1500],
            "upload_date": JevClassifier._first_value(candidate, "upload_date", "release_date"),
            "release_date": JevClassifier._first_value(candidate, "release_date", "upload_date"),
            "album": candidate.get("album"),
            "year": JevClassifier._first_value(candidate, "year", "release_year"),
            "genre": candidate.get("genre"),
            "view_count": candidate.get("view_count"),
            "is_live": bool(candidate.get("is_live")),
        }

    @staticmethod
    def _candidate_label(candidate: dict[str, Any]) -> str:
        title = str(candidate.get("title") or "untitled")
        channel = str(candidate.get("channel") or candidate.get("uploader") or "unknown channel")
        return f'candidate "{title}" uploaded by "{channel}"'

    def _probability(self, value: Any) -> float:
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise JevEvaluationError(
                f"{self.provider_name} returned an invalid probability"
            ) from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise JevEvaluationError(
                f"{self.provider_name} returned an invalid probability"
            )
        return probability
