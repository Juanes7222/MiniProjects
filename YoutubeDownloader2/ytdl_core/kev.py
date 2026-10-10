"""Kev as a decision provider.

Kev (``jaredpalmer/kev``) is a family of small System One decision models you
train and run yourself. It is not a different *kind* of provider than Decider:
both serve ``POST /v1/systemone``, both answer ``noul``, and both are scored by
the same questions in :mod:`ytdl_core.decision_questions`. The only difference is
the endpoint and the four sizes it ships, which is why this module is a URL, a
model name and a name -- not a copy of the transport.

Serving it locally is supervised by :mod:`ytdl_core.kev_server`.
"""

from __future__ import annotations

from . import decision_questions as dq
from .systemone import SystemOneClassifier, SystemOneDecision

# Kept importable under its historical name. The response holder was called this
# while it only ever wrapped a Kev server; it now belongs to the shared System One
# transport, and the old name is an alias rather than a second class.
KevDecision = SystemOneDecision

__all__ = ["KevClassifier", "KevDecision"]


class KevClassifier(SystemOneClassifier):
    """A local Kev server speaking the System One question contract."""

    provider_name = "Kev"
    start_flag = "--kev"
    auth_env = "KEV_API_KEY"

    def __init__(
        self,
        url: str = "http://127.0.0.1:8009",
        model: str = "kev-latest",
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
        # Not dq.EVALUATION_HEADROOM (4): core.py always passes
        # Config.DECISION_HEADROOM (1), so this only applies to a classifier built
        # directly, and 1 is what that path has always used.
        eval_headroom: int = 1,
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
