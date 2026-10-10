"""Transport for a self-hosted System One decision server.

Kev (``kev.serve``) and Decider (``decider.serve``) expose the *same* endpoint:
``POST /v1/systemone``, TypeSafe's wire format, in which the yes/no type is
``noul``. So the HTTP half of a local provider is identical for both -- one
pooled session, the admission gate, the retry policy, and the per-song telemetry
the server already measured. That shared half lives here, and a concrete
provider only supplies a URL, a model name and the words in its error messages.

What a provider may *not* share is in :mod:`ytdl_core.decision_questions`: what
is asked, how the answers combine, and what vetoes a candidate. That is shared
on purpose. A question improvement has to land on every provider at once, or
two providers quietly stop being comparable -- and a score gathered under one
question set cannot be compared against a threshold tuned under another.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

from . import decision_questions as dq
from .jev import JevClassifier, JevEvaluationError
from .ratelimit import full_jitter_backoff


class SystemOneDecision:
    """One provider answer plus the telemetry the server already sent with it.

    The System One response carries ``latency_ms``, ``usage.input_tokens`` /
    ``usage.output_tokens`` and an ``x-typesafe-request-id`` header. All of it
    costs the client nothing to read and is the only direct measurement of what
    the GPU is doing per song, so it is kept rather than discarded along with the
    rest of the body.
    """

    __slots__ = ("answers", "latency_ms", "input_tokens", "output_tokens", "request_id")

    def __init__(
        self,
        answers: dict[str, Any],
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.answers = answers
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.request_id = request_id

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return int(self.input_tokens or 0) + int(self.output_tokens or 0)


class SystemOneClassifier(JevClassifier):
    """A decision model served over ``POST /v1/systemone``.

    Transport only, exactly like its parent: every judgment lives in
    :mod:`ytdl_core.decision_questions`. Subclasses declare *who* they are --
    the display name, the flag that starts them, and how they authenticate --
    and inherit the transport unchanged.
    """

    # How this provider is named in a report, an error and the CLI banner.
    provider_name = "System One"
    # The flag that starts it, quoted in the "server is unavailable" message so
    # the fix is on the same line as the diagnosis.
    start_flag = "--decision"
    # The environment variable holding a bearer key, when the server has one.
    # Empty means the endpoint is unauthenticated.
    auth_env = ""
    # The flag that raises the per-evaluation timeout.
    timeout_flag = "--decision-timeout"

    def __init__(
        self,
        url: str,
        model: str | None,
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
        eval_headroom: int = 1,
    ) -> None:
        super().__init__(
            threshold=threshold,
            timeout_seconds=timeout_seconds,
            gate_floor=gate_floor,
            gate_floors=gate_floors,
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
        # One pooled connection set for the whole run.
        #
        # `requests.post` opens a fresh TCP connection and a fresh TLS handshake
        # per call, which for a request made once per song is a per-request
        # constant that buys nothing: the server is local, the payload is the
        # large part, and the answers are already batched. It is also visible
        # from the other side -- with no keep-alive every line in the server's
        # access log shows a different ephemeral source port.
        #
        # The pool is sized to the gate's in-flight allowance because that is the
        # hard ceiling on simultaneous connections; anything above it would just
        # queue inside urllib3 instead of at the server.
        self._session_lock = threading.Lock()
        self._session: requests.Session | None = None
        self._session_maxsize = max(1, int(max_in_flight))
        # Telemetry across the run. Read by the session summary.
        self.telemetry_lock = threading.Lock()
        self.request_count = 0
        self.batch_count = 0
        self.token_count = 0
        self.latency_total_ms = 0.0
        self.latency_max_ms = 0.0

    def _http(self) -> requests.Session:
        """The shared pooled session, created once.

        ``requests.Session`` is not documented as thread-safe, but the urllib3
        connection pool underneath one is, and the pool is the part that has to be
        shared -- ``metadata._shared_session`` relies on the same reasoning.
        ``pool_maxsize`` defaults to 10 regardless of how many threads call it,
        so it is raised to the gate's allowance explicitly.
        """
        session = self._session
        if session is None:
            with self._session_lock:
                session = self._session
                if session is None:
                    session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        pool_connections=max(2, self._session_maxsize),
                        pool_maxsize=max(2, self._session_maxsize),
                        max_retries=0,
                    )
                    session.mount("https://", adapter)
                    session.mount("http://", adapter)
                    # One decision server, addressed by name.
                    session.headers.update({"accept": "application/json"})
                    self._session = session
        return session

    def close(self) -> None:
        """Release the pooled connections."""
        with self._session_lock:
            session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def telemetry(self) -> dict[str, float | int]:
        """What the run actually cost the decision model."""
        with self.telemetry_lock:
            count = self.request_count
            return {
                "requests": count,
                "tokens": self.token_count,
                "total_latency_ms": round(self.latency_total_ms, 1),
                "mean_latency_ms": round(self.latency_total_ms / count, 1) if count else 0.0,
                "max_latency_ms": round(self.latency_max_ms, 1),
                "max_in_flight": self._session_maxsize,
            }

    def _request_model(self) -> str | None:
        """The name to put in the request body, or None to omit the field."""
        return self.model

    def _evaluate(
        self, state: dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"state": state, "questions": questions}
        name = model or self._request_model()
        if name:
            body["model"] = name

        gate = self.gate
        # Refuse straight away once the provider has been given up on. Without
        # this, every remaining song pays the full timeout-and-retry ladder to
        # arrive at a verdict that is not coming -- on a thousand-song batch,
        # hours of waiting to be told the same thing a hundred times.
        if not gate.admit():
            raise JevEvaluationError("decision provider is not responding; skipping it")
        try:
            decision = self._post_with_retries(body)
        except JevEvaluationError:
            gate.record_failure()
            raise
        except Exception as exc:  # noqa: BLE001 - normalised for the caller
            gate.record_failure()
            raise JevEvaluationError(str(exc)) from exc
        else:
            gate.record_success()
            self._record_telemetry(decision)
            return decision.answers
        finally:
            # Always hand the slot back. The gate admits one request at a time by
            # default, so leaking a slot here would deadlock every later song
            # rather than merely serialise them.
            gate.release()

    def _record_telemetry(self, decision: SystemOneDecision) -> None:
        """Accumulate what the server reported about this evaluation."""
        with self.telemetry_lock:
            self.request_count += 1
            tokens = decision.total_tokens
            if tokens:
                self.token_count += tokens
            if decision.latency_ms is not None:
                self.latency_total_ms += float(decision.latency_ms)
                self.latency_max_ms = max(self.latency_max_ms, float(decision.latency_ms))

    def _post_with_retries(self, body: dict[str, Any]) -> SystemOneDecision:
        response: requests.Response | None = None
        session = self._http()
        for attempt in range(self.max_attempts):
            try:
                response = session.post(
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
                # ``timeout_flag``, not repetition; and if it is not, the gate
                # gives up instead of multiplying the wait.
                raise JevEvaluationError(
                    f"{self.provider_name} evaluation timed out after "
                    f"{self.timeout_seconds}s (not retried: a slow model needs a "
                    f"longer {self.timeout_flag}, not another attempt)"
                ) from exc
            except requests.ConnectionError as exc:
                raise JevEvaluationError(
                    f"{self.provider_name} server is unavailable at {self.url}; "
                    f"start it with {self.start_flag}"
                ) from exc
            except requests.RequestException as exc:
                raise JevEvaluationError(f"{self.provider_name} request failed") from exc

            status = response.status_code
            if 200 <= status < 300:
                break
            if status in (401, 403):
                hint = f"; check {self.auth_env}" if self.auth_env else ""
                raise JevEvaluationError(
                    f"{self.provider_name} authentication failed{hint}"
                )
            if status == 404:
                raise JevEvaluationError(
                    f"{self.provider_name} endpoint or model was not found"
                )
            if status == 413:
                raise JevEvaluationError(
                    f"{self.provider_name} rejected the request as too large; lower "
                    f"--decision-questions"
                )
            if status == 422:
                raise JevEvaluationError(
                    f"{self.provider_name} rejected the evaluation request"
                )
            if status == 429 or status >= 500:
                if attempt < self.max_attempts - 1:
                    time.sleep(full_jitter_backoff(attempt + 1, base=0.5, cap=5.0))
                    continue
                raise JevEvaluationError(
                    f"{self.provider_name} server rate limit reached"
                    if status == 429
                    else f"{self.provider_name} server temporarily failed"
                )
            raise JevEvaluationError(f"{self.provider_name} server returned HTTP {status}")

        if response is None:  # pragma: no cover - loop always sets or raises
            raise JevEvaluationError(f"{self.provider_name} evaluation produced no response")

        try:
            data = response.json()
        except ValueError as exc:
            raise JevEvaluationError(
                f"{self.provider_name} server returned invalid JSON"
            ) from exc
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise JevEvaluationError(f"{self.provider_name} server returned no answers")
        # Keep the measurements the server already made. They are the only direct
        # reading of per-song GPU cost, and they arrive whether or not anyone
        # looks at them.
        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        latency = data.get("latency_ms")
        # Read defensively: every field here is diagnostic, and none of them is
        # worth failing an otherwise-good evaluation over.
        headers = getattr(response, "headers", None)
        get_header = getattr(headers, "get", None)
        request_id = get_header("x-typesafe-request-id") if callable(get_header) else None
        return SystemOneDecision(
            answers,
            latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
            input_tokens=_int_or_none(usage.get("input_tokens")),
            output_tokens=_int_or_none(usage.get("output_tokens")),
            request_id=request_id,
        )


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
