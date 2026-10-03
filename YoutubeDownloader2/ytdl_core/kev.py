from __future__ import annotations

import time
from typing import Any

import requests

from . import decision_questions as dq
from .jev import JevClassifier, JevEvaluationError
from .ratelimit import full_jitter_backoff


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
        max_attempts: int = 3,
        max_questions: int = 32,
        max_in_flight: int = 1,
        failure_threshold: int = 2,
        eval_headroom: int = 1,
    ) -> None:
        super().__init__(
            threshold=threshold,
            timeout_seconds=timeout_seconds,
            gate_floor=gate_floor,
            stable_spread=stable_spread,
            max_candidates=max_candidates,
            max_questions=max_questions,
            max_in_flight=max_in_flight,
            failure_threshold=failure_threshold,
            eval_headroom=eval_headroom,
        )
        self.url = url.rstrip("/")
        self.model = model
        # A local System One server speaks TypeSafe's own dialect, not the
        # gateway's, so the yes/no type is `noul` here and `boolean` for Jev.
        self.dialect = dq.TYPESAFE_DIALECT
        self.max_attempts = max(1, int(max_attempts))

    def _evaluate(
        self, state: dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "state": state,
            "model": model or self.model,
            "questions": questions,
        }

        gate = self.gate
        # Refuse straight away once the provider has been given up on. Without
        # this, every remaining song pays the full timeout-and-retry ladder to
        # arrive at a verdict that is not coming -- on a thousand-song batch,
        # hours of waiting to be told the same thing a hundred times.
        if not gate.admit():
            raise JevEvaluationError("decision provider is not responding; skipping it")
        try:
            answers = self._post_with_retries(body)
        except JevEvaluationError:
            gate.record_failure()
            raise
        except Exception as exc:  # noqa: BLE001 - normalised for the caller
            gate.record_failure()
            raise JevEvaluationError(str(exc)) from exc
        else:
            gate.record_success()
            return answers
        finally:
            # Always hand the slot back. The gate admits one request at a time by
            # default, so leaking a slot here would deadlock every later song
            # rather than merely serialise them.
            gate.release()

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        response: requests.Response | None = None
        for attempt in range(self.max_attempts):
            try:
                response = requests.post(
                    f"{self.url}/v1/systemone",
                    json=body,
                    timeout=self.timeout_seconds,
                )
            except requests.Timeout as exc:
                # Deliberately not retried.
                #
                # A timeout is not a transient failure here -- it is the server
                # saying "this computation needs more time". Retrying recomputes
                # the same forward pass from scratch, throws away the GPU work
                # already done, and turns one slow answer into several. If the
                # model is genuinely this slow, the fix is a longer
                # ``--kev-timeout``, not repetition; and if it is not, the gate
                # gives up instead of multiplying the wait.
                raise JevEvaluationError(
                    f"Kev evaluation timed out after {self.timeout_seconds}s "
                    "(not retried: a slow model needs a longer --kev-timeout, "
                    "not another attempt)"
                ) from exc
            except requests.ConnectionError as exc:
                raise JevEvaluationError(
                    f"Kev server is unavailable at {self.url}; start it with kev.serve"
                ) from exc
            except requests.RequestException as exc:
                raise JevEvaluationError("Kev request failed") from exc

            status = response.status_code
            if 200 <= status < 300:
                break
            if status in (401, 403):
                raise JevEvaluationError("Kev authentication failed; check KEV_API_KEY")
            if status == 404:
                raise JevEvaluationError("Kev endpoint or model was not found")
            if status == 422:
                raise JevEvaluationError("Kev rejected the evaluation request")
            if status == 429 or status >= 500:
                if attempt < self.max_attempts - 1:
                    time.sleep(full_jitter_backoff(attempt + 1, base=0.5, cap=5.0))
                    continue
                raise JevEvaluationError(
                    "Kev server rate limit reached"
                    if status == 429
                    else "Kev server temporarily failed"
                )
            raise JevEvaluationError(f"Kev server returned HTTP {status}")

        if response is None:  # pragma: no cover - loop always sets or raises
            raise JevEvaluationError("Kev evaluation produced no response")

        try:
            data = response.json()
        except ValueError as exc:
            raise JevEvaluationError("Kev server returned invalid JSON") from exc
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise JevEvaluationError("Kev server returned no answers")
        return answers