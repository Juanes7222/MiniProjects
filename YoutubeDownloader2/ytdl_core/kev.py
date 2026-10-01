from __future__ import annotations

import time
from typing import Any

import requests

from . import decision_questions as dq
from .jev import JevClassifier, JevEvaluationError


class KevClassifier(JevClassifier):
    """Local/self-hosted System One server speaking the same question contract.

    Differs from :class:`JevClassifier` only in transport. It shares the payload
    and the scoring, so a question improvement lands on both providers at once.
    """

    provider_name = "Kev"

    def __init__(
        self,
        url: str = "http://127.0.0.1:8009",
        model: str = "kev-latest",
        threshold: float = 0.60,
        timeout_seconds: int = 60,
        gate_floor: float = dq.GATE_FLOOR,
        stable_spread: float = dq.STABLE_SPREAD,
        max_candidates: int = dq.MAX_EVALUATED_CANDIDATES,
    ) -> None:
        super().__init__(
            threshold=threshold,
            timeout_seconds=timeout_seconds,
            gate_floor=gate_floor,
            stable_spread=stable_spread,
            max_candidates=max_candidates,
        )
        self.url = url.rstrip("/")
        self.model = model
        # A local System One server speaks TypeSafe's own dialect, not the
        # gateway's, so the yes/no type is `noul` here and `boolean` for Jev.
        self.dialect = dq.TYPESAFE_DIALECT

    def _evaluate(
        self, state: dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "state": state,
            "model": model or self.model,
            "questions": questions,
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.url}/v1/systemone",
                    json=body,
                    timeout=self.timeout_seconds,
                )
            except requests.Timeout:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise JevEvaluationError("Kev evaluation timed out after 3 attempts")
            except requests.ConnectionError as exc:
                raise JevEvaluationError(
                    f"Kev server is unavailable at {self.url}; start it with kev.serve"
                ) from exc
            except requests.RequestException as exc:
                raise JevEvaluationError("Kev request failed") from exc

            if 200 <= response.status_code < 300:
                break
            if response.status_code == 401 or response.status_code == 403:
                raise JevEvaluationError("Kev authentication failed; check KEV_API_KEY")
            if response.status_code == 404:
                raise JevEvaluationError("Kev endpoint or model was not found")
            if response.status_code == 422:
                raise JevEvaluationError("Kev rejected the evaluation request")
            if response.status_code == 429:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise JevEvaluationError("Kev server rate limit reached")
            if response.status_code >= 500:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise JevEvaluationError("Kev server temporarily failed")
            raise JevEvaluationError(f"Kev server returned HTTP {response.status_code}")

        try:
            data = response.json()
        except ValueError as exc:
            raise JevEvaluationError("Kev server returned invalid JSON") from exc
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise JevEvaluationError("Kev server returned no answers")
        return answers
