from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


class JevEvaluationError(RuntimeError):
    pass


class JevClassifier:
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
    ) -> tuple[dict | None, list[tuple[dict, int, dict[str, int]]]]:
        if not candidates:
            return None, []

        candidate_payloads = [self._candidate_payload(candidate, index) for index, candidate in enumerate(candidates)]
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
        answers = self._evaluate(payload)
        evaluated: list[tuple[dict, int, dict[str, int]]] = []

        for candidate_payload, candidate in zip(candidate_payloads, candidates, strict=True):
            answer = answers.get(candidate_payload["key"])
            if not isinstance(answer, dict) or answer.get("type") != "boolean":
                raise JevEvaluationError("Jev returned an invalid boolean answer")
            probability = self._probability(answer.get("probability"))
            entry = dict(candidate)
            heuristic_score = int(entry.get("_composite_score") or 0)
            breakdown = dict(entry.get("_score_breakdown") or {})
            jev_score = int(round(probability * 100))
            entry["_heuristic_score"] = heuristic_score
            entry["_heuristic_breakdown"] = breakdown
            entry["_jev_probability"] = probability
            entry["_jev_threshold"] = self.threshold
            entry["_jev_selected"] = False
            entry["_composite_score"] = jev_score
            entry["_score_breakdown"] = {
                **breakdown,
                "jev_probability": jev_score,
            }
            evaluated.append((entry, jev_score, entry["_score_breakdown"]))

        evaluated.sort(
            key=lambda item: (item[1], item[0].get("_heuristic_score", 0)), reverse=True
        )
        if not evaluated or evaluated[0][0]["_jev_probability"] < self.threshold:
            return None, evaluated

        selected = dict(evaluated[0][0])
        selected["_jev_selected"] = True
        selected_score, selected_breakdown = evaluated[0][1], evaluated[0][2]
        evaluated[0] = (selected, selected_score, selected_breakdown)
        return selected, evaluated

    def _evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        script = self.project_root / "tools" / "jev.mts"
        if not script.is_file():
            raise JevEvaluationError("Jev bridge script was not found")

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
        except subprocess.TimeoutExpired as exc:
            raise JevEvaluationError("Jev evaluation timed out") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.lower()
            if "valid credit card" in stderr:
                raise JevEvaluationError(
                    "AI Gateway requires a valid credit card before it can service Jev requests"
                )
            if "unauthenticated" in stderr or "api key" in stderr:
                raise JevEvaluationError("AI Gateway authentication failed")
            raise JevEvaluationError("Jev bridge failed; verify gateway access and model availability")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise JevEvaluationError("Jev bridge returned invalid JSON") from exc
        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict):
            raise JevEvaluationError("Jev bridge returned no answers")
        return answers

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

    @staticmethod
    def _probability(value: Any) -> float:
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise JevEvaluationError("Jev returned an invalid probability") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise JevEvaluationError("Jev returned an invalid probability")
        return probability
