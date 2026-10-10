"""Decider as a decision provider.

Decider (``Mapika/decider``) is an open reproduction of the System One model
class: a family of fine-tuned models that read one state and a set of typed
questions and return a calibrated probability distribution from a single forward
pass, without generating text. Like Kev it serves ``POST /v1/systemone`` and
answers ``noul``, so it shares the transport in :mod:`ytdl_core.systemone` and
the questions in :mod:`ytdl_core.decision_questions` unchanged. What it adds is
choice: it ships from 0.8B to 35B, serves on MPS and CPU as well as CUDA, and
publishes GGUF weights for llama.cpp -- so it is the provider that works on a
machine with no NVIDIA card.

Two things are worth knowing before choosing it over Kev:

* **Knowledge, not judgment, is its weak axis.** Its own model cards put it
  behind Jev on MMLU/GPQA/GSM8K-style knowledge and behind it on multi-hop
  reasoning over long policy text. Judging whether a YouTube upload is the right
  song by the right artist is none of those, but the workload that flatters it
  here is our questions, and that is a claim to measure rather than assume.
* **Its probabilities are the product.** It fits a temperature per question type
  and ships it in the model's own config, and it can re-ask a question it is
  unsure about with its options reversed. Every threshold in this module is set
  against a calibrated probability, so that is worth having.

Serving it locally is supervised by :mod:`ytdl_core.decider_server`.
"""

from __future__ import annotations

import threading

import requests

from . import decision_questions as dq
from .systemone import SystemOneClassifier

__all__ = ["DeciderClassifier"]


class DeciderClassifier(SystemOneClassifier):
    """A local ``decider.serve`` speaking the System One question contract."""

    provider_name = "Decider"
    start_flag = "--decider"
    # decider.serve binds to 127.0.0.1 and ships without authentication, so there
    # is no key to point at. Left empty on purpose rather than invented.
    auth_env = ""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8010",
        model: str | None = None,
        threshold: float = 0.60,
        timeout_seconds: int = 60,
        gate_floor: float | None = None,
        gate_floors: dict[str, float] | None = None,
        stable_spread: float = dq.STABLE_SPREAD,
        max_candidates: int = dq.MAX_EVALUATED_CANDIDATES,
        max_attempts: int = 3,
        max_questions: int = 32,
        max_in_flight: int = 1,
        failure_threshold: int = 2,
        eval_headroom: int = dq.EVALUATION_HEADROOM,
    ) -> None:
        super().__init__(
            url=url,
            model=model,
            threshold=threshold,
            timeout_seconds=timeout_seconds,
            gate_floor=gate_floor,
            gate_floors=gate_floors,
            stable_spread=stable_spread,
            max_candidates=max_candidates,
            max_attempts=max_attempts,
            max_questions=max_questions,
            max_in_flight=max_in_flight,
            failure_threshold=failure_threshold,
            eval_headroom=eval_headroom,
        )
        self._model_lock = threading.Lock()
        self._discovered_model: str | None = None

    def _request_model(self) -> str | None:
        """The checkpoint name for the request body.

        ``decider.serve`` names the loaded checkpoint after the version in the
        model's own ``decider_config.json`` (``decider-v2.1`` and so on), so there
        is no stable string to hardcode the way ``kev-latest`` is: pinning one
        here would 404 against every checkpoint but that one, and the failure
        would look like a missing endpoint. Ask the server instead, once, and
        remember the answer. An explicit ``--decider-model`` still wins.
        """
        if self.model:
            return self.model
        discovered = self._discovered_model
        if discovered:
            return discovered
        with self._model_lock:
            if self._discovered_model:
                return self._discovered_model
            # Never fatal: the field is optional on the wire, and a server too
            # busy to answer /v1/models will still answer /v1/systemone.
            discovered = self._discover_model()
            self._discovered_model = discovered
            return discovered

    def _discover_model(self) -> str | None:
        try:
            response = self._http().get(f"{self.url}/v1/models", timeout=self.timeout_seconds)
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        # The shape has moved between releases -- kev.serve reports a `models`
        # list, decider.serve reports the loaded checkpoint's `name` -- so read
        # both rather than guess one and be wrong on the other.
        name = data.get("name") or data.get("model")
        if isinstance(name, str) and name:
            return name
        models = data.get("models")
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict):
                    candidate = entry.get("name")
                    if isinstance(candidate, str) and candidate:
                        return candidate
        return None
