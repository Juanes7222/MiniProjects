"""Supervision of a local Kev System One server.

Kev is a decision model: it reads one state and a set of typed questions about
it and returns a calibrated probability distribution, in a single forward pass and
without generating text. It is fast, but only if it is allowed to be. Three
things in this module exist purely to stop the server being started in a way
that quietly throws that away, and a fourth to notice when it happens anyway.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, TextIO
from urllib.parse import urlparse

import requests

# The fused kernels are pinned by kev itself and this must match it exactly:
# kev.fused_qwen35 refuses any other release rather than tolerating it.
FLA_VERSION = "0.5.2"

# flash-linear-attention is only installed in the *Kev* environment, never in the
# application's. The application's own interpreter has no business carrying a
# multi-gigabyte CUDA stack it never imports, and the two resolve differently.
#
# Triton is the other half and the awkward one: it is not distributed for Windows
# on PyPI, and its binaries are compiled against a specific libtorch, so the
# series has to track the torch build rather than being taken latest.
_TRITON_WINDOWS_SERIES = {(2, 7): "3.3", (2, 8): "3.4", (2, 9): "3.5"}


class KevServerError(RuntimeError):
    pass


class KevServerManager:
    def __init__(
        self,
        root: Path,
        run: str = "jaredpalmer/kev-4b@v1.0",
        url: str = "http://127.0.0.1:8009",
        port: int = 8009,
        startup_timeout: int = 900,
        update: bool = True,
        insist_fused: bool | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.expanduser()
        self.run = run
        self.url = url.rstrip("/")
        self.port = port
        self.startup_timeout = startup_timeout
        self.update = update
        # True = use the fused kernels without probing, False = decline them
        # without probing, None = let the probe decide.
        self.insist_fused = insist_fused
        self.on_step = on_step or (lambda message: None)
        self.process: subprocess.Popen | None = None
        self.log_file: TextIO | None = None
        self._job_handle: int | None = None
        self._previous_sigterm_handler: Any = None
        # Detected during _sync_environment, reused by _verify_cuda so the
        # hardware is only ever probed once per start.
        self._gpu_name: str | None = None
        # Whether the fused Qwen3.5 kernels actually *run* here. Tri-state on
        # purpose: None = not decided yet, True/False = measured. See
        # _install_fused_kernels.
        self._fusable: bool | None = None

    def start(self) -> str:
        if not self._is_local_url():
            self._step(f"Kev: checking remote endpoint {self.url}")
            if not self._healthy():
                raise KevServerError(f"Kev server did not respond at {self.url}")
            return self.url

        if self._healthy():
            info = self._server_info()
            if not self._server_uses_cuda(info):
                raise KevServerError("A Kev server is already running, but it is not using CUDA")
            self._step("Kev: CUDA server is already available; reusing it")
            self._report_capabilities()
            return self.url

        self._ensure_repository()
        self._sync_environment()
        self._verify_cuda()
        self._step(f"Kev: starting {self.run} on CUDA through kev.serve")
        self._start_process()
        self._install_termination_handler()
        self._wait_until_ready()
        # Up is not the same as usable at batch speed, and neither is the same as
        # running the fast path. Find out now, not from a thousand timeouts.
        self._report_capabilities()
        self._report_latency()
        return self.url

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if self._job_handle is not None:
                self._terminate_job(self._job_handle)
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            else:
                killpg = getattr(os, "killpg", None)
                getpgid = getattr(os, "getpgid", None)
                if killpg is not None and getpgid is not None:
                    try:
                        killpg(getpgid(self.process.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    killpg = getattr(os, "killpg", None)
                    getpgid = getattr(os, "getpgid", None)
                    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                    if killpg is not None and getpgid is not None:
                        try:
                            killpg(getpgid(self.process.pid), sigkill)
                        except ProcessLookupError:
                            pass
                self.process.kill()
                self.process.wait(timeout=10)
        if self._job_handle is not None:
            self._close_handle(self._job_handle)
            self._job_handle = None
        self._restore_termination_handler()
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self.process = None

    # -- repository -----------------------------------------------------------

    def _ensure_repository(self) -> None:
        if not self.root.exists():
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self._step(f"Kev: downloading repository to {self.root}")
            self._run(
                ["git", "clone", "https://github.com/jaredpalmer/kev.git", str(self.root)],
                self.root.parent,
            )
            return

        if not (self.root / ".git").is_dir():
            raise KevServerError(f"Kev directory is not a Git repository: {self.root}")

        if not self.update:
            self._step("Kev: update skipped by configuration")
            return

        status = self._run(["git", "status", "--porcelain"], self.root)
        if status.strip():
            self._step("Kev: local changes detected; keeping the current version")
            return

        self._step("Kev: checking repository updates")
        self._run(["git", "pull", "--ff-only"], self.root)

    # -- environment ----------------------------------------------------------

    def _venv_python(self) -> Path:
        return self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _nvidia_gpu_name(self) -> str | None:
        """Name of the first NVIDIA GPU, or None. Does not require torch.

        Detecting the hardware by importing torch is circular here: the sync
        deliberately excludes torch (it is a multi-gigabyte download installed
        separately with a CUDA-specific backend), so the one thing we need torch
        for is the one thing torch is not there to answer. ``nvidia-smi`` is part
        of the NVIDIA driver, is always present when a CUDA device is, and tells
        us what we need before anything is installed.
        """
        executable = shutil_which("nvidia-smi")
        if executable is None:
            return None
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        names = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
        return names[0] if names else None

    def _sync_environment(self) -> None:
        if shutil_which("uv") is None:
            raise KevServerError("uv was not found; install it to manage Kev")

        self._gpu_name = self._nvidia_gpu_name()

        if self._gpu_name is None:
            # No CUDA device: install the ordinary CPU torch so the environment
            # is at least coherent, and let _verify_cuda explain the real problem.
            # Excluding torch here would instead leave a broken venv behind and
            # report the absence as a missing-module error.
            self._step("Kev: no NVIDIA GPU found; syncing CPU dependencies")
            self._run(
                ["uv", "sync", "--extra", "serve", "--inexact"],
                self.root,
                env=self._environment(),
            )
            return

        self._step(f"Kev: CUDA GPU found ({self._gpu_name}); syncing dependencies")
        self._run(
            [
                "uv",
                "sync",
                "--extra",
                "serve",
                "--inexact",
                "--no-install-package",
                "torch",
            ],
            self.root,
            env=self._environment(),
        )
        self._install_cuda_torch()

    def _install_cuda_torch(self) -> None:
        python_path = self._venv_python()
        self._step("Kev: installing CUDA-enabled PyTorch (cu128)")
        # 2.7 is the floor on purpose: it is the first release whose cu128 wheels
        # carry Blackwell (sm_120) kernels, which is what current consumer cards
        # need. Older builds import fine and then fail at runtime with
        # "no kernel image is available for execution on the device".
        self._run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_path),
                "--torch-backend",
                "cu128",
                "--reinstall",
                "torch>=2.7,<2.9",
            ],
            self.root,
            env=self._environment(),
        )
        self._install_fused_kernels()

    # -- fused Qwen3.5 kernels ------------------------------------------------
    #
    # Every current Kev checkpoint is a Qwen3.5/Qwen3.8 hybrid backbone whose
    # Gated DeltaNet layers are recurrent. kev.serve serves *fused* Triton kernels
    # for them when flash-linear-attention is importable at exactly the pinned
    # version -- about a third less GPU time per batch -- and otherwise falls back
    # to PyTorch's reference layers, printing one banner line.
    #
    # "Importable" turns out to be a poor proxy for "runs". On the reference
    # machine (RTX 5060 Ti, sm_120, Windows, triton-windows) a trivial Triton
    # kernel launched correctly and the package imported at exactly the pinned
    # version, and then the fused DeltaNet kernels never returned: the first real
    # evaluation hung indefinitely while the identical request with fused
    # declined answered in 2.5 s. An import check cannot see that. So the decision
    # is made by running a real evaluation under a hard timeout.

    def _installed_torch_version(self) -> tuple[int, int] | None:
        """``(major, minor)`` of the torch in the Kev venv, or None if unknown."""
        try:
            completed = subprocess.run(
                [str(self._venv_python()), "-c", "import torch;print(torch.__version__)"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        raw = (completed.stdout or "").strip().split("+", 1)[0]
        parts = raw.split(".")
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return None
        return int(parts[0]), int(parts[1])

    def _triton_requirement(self) -> str | None:
        """The Triton build matching this machine, or None to take the default.

        On Linux the Triton shipped with the cu128 torch build already satisfies
        fla, so there is nothing to pin. On Windows it does not exist and
        ``triton-windows`` is the only source, so it has to be matched to the
        torch version deliberately rather than taken latest.
        """
        if os.name != "nt":
            return None
        version = self._installed_torch_version()
        if version is None:
            return None
        series = _TRITON_WINDOWS_SERIES.get(version[:2])
        if series is not None:
            return f"triton-windows~={series}.0"
        self._step(
            f"Kev: torch {version[0]}.{version[1]} has no known triton-windows match; "
            "installing the latest and verifying the kernel actually answers"
        )
        return "triton-windows"

    def _fused_kernel_probe(self) -> tuple[bool, str]:
        """Whether kev's own fused-kernel precondition holds. Never raises.

        Asks kev the same question its serving path asks, rather than a
        hand-rolled approximation: only kev's gate decides whether the fused path
        is taken at all.
        """
        script = (
            "import sys;"
            f"sys.path.insert(0, {str(self.root)!r});"
            "from kev.checkpoint import fused_available;"
            "print('FUSED=' + str(fused_available()))"
        )
        try:
            completed = subprocess.run(
                [str(self._venv_python()), "-c", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        output = (completed.stdout or "") + (completed.stderr or "")
        lines = [line for line in output.splitlines() if line.strip()]
        if "FUSED=True" in output:
            return True, lines[-1] if lines else "available"
        return False, (lines[-1] if lines else "no output")[:200]

    def _probe_fused_latency(self, budget_seconds: float = 120.0) -> bool:
        """Run one real evaluation with fused forced on; did it answer?

        The only honest test. Uses a throwaway server on its own port so the real
        one is untouched, runs the request on its own thread under a hard
        timeout, and kills the server either way. A stall therefore costs
        ``budget_seconds`` and reports "no", rather than hanging the caller.
        """
        probe_port = self.port + 1
        env = self._environment()
        env.pop("KEV_FUSED", None)  # ask for fused whatever the current answer is
        log_path = self.root / "ytdl-kev-fused-probe.log"
        try:
            log: Any = log_path.open("a", encoding="utf-8")
        except OSError:
            log = subprocess.DEVNULL

        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "--extra",
                    "serve",
                    "python",
                    "-m",
                    "kev.serve",
                    "--run",
                    self.run,
                    "--port",
                    str(probe_port),
                ],
                cwd=self.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                ),
            )
            base = f"http://127.0.0.1:{probe_port}"
            deadline = time.monotonic() + self.startup_timeout
            up = False
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    return False
                try:
                    requests.get(f"{base}/v1/models", timeout=3)
                    up = True
                    break
                except requests.RequestException:
                    time.sleep(3)
            if not up:
                return False

            payload = self._probe_payload()
            answer: list[bool] = []

            def _ask() -> None:
                try:
                    response = requests.post(
                        f"{base}/v1/systemone", json=payload, timeout=budget_seconds
                    )
                    answer.append(response.status_code < 500)
                except Exception:
                    answer.append(False)

            worker = threading.Thread(target=_ask, daemon=True)
            worker.start()
            worker.join(timeout=budget_seconds)
            return bool(answer) and answer[0]
        except Exception:
            return False
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            capture_output=True,
                            timeout=30,
                            check=False,
                        )
                    else:
                        proc.terminate()
                    proc.wait(timeout=15)
                except Exception:
                    pass
            if log is not subprocess.DEVNULL:
                try:
                    log.close()
                except Exception:
                    pass

    def _install_fused_kernels(self) -> None:
        """Install flash-linear-attention (and Triton where needed), then prove
        the fused path answers on this machine.

        Best-effort by design. A missing fused path costs throughput, not
        correctness -- the model still answers, just slower -- so failing the
        whole startup over it would trade a performance problem for an
        availability one. What it must never do is fail *silently*, and it must
        never accept an import as proof of a working kernel.
        """
        if self.insist_fused is True:
            self._fusable = True
            self._step("Kev: --kev-fused given; using the fused Qwen3.5 kernels unprobed")
            return
        if self.insist_fused is False:
            self._fusable = False
            self._step("Kev: --no-kev-fused given; declining the fused Qwen3.5 kernels")
            return

        available, _detail = self._fused_kernel_probe()
        if not available:
            triton_requirement = self._triton_requirement()
            self._step(
                f"Kev: installing fused kernels (flash-linear-attention=={FLA_VERSION})"
            )
            packages = [f"flash-linear-attention=={FLA_VERSION}"]
            if triton_requirement:
                packages.append(triton_requirement)
            try:
                self._run(
                    ["uv", "pip", "install", "--python", str(self._venv_python()), *packages],
                    self.root,
                    env=self._environment(),
                )
            except KevServerError as exc:
                self._fusable = False
                self._step(
                    f"Kev: fused kernels not installed ({exc}). The model still works, but "
                    "the Qwen3.5 Gated DeltaNet layers will run PyTorch's slower reference "
                    "path. Install them manually with: uv pip install --python .venv "
                    f"flash-linear-attention=={FLA_VERSION}"
                )
                return
            available, _detail = self._fused_kernel_probe()

        if not available:
            self._fusable = False
            self._step(
                "Kev: fused kernels unavailable; the Qwen3.5 Gated DeltaNet layers will "
                "run PyTorch's slower reference path. Install them with: uv pip install "
                f"--python .venv flash-linear-attention=={FLA_VERSION}"
            )
            return

        # Importable at the pinned version. Now find out whether it runs.
        self._fusable = True
        self._step(
            "Kev: flash-linear-attention present; verifying the fused kernels actually "
            "answer on this GPU (an import check cannot tell)"
        )
        if self._probe_fused_latency():
            self._step(f"Kev: fused Qwen3.5 kernels verified end-to-end (FLA {FLA_VERSION})")
            return

        self._fusable = False
        self._step(
            "Kev: fused Qwen3.5 kernels are installed but DID NOT ANSWER a real evaluation "
            "within the probe budget, so they are being declined (KEV_FUSED=0). Some "
            "Triton/platform combinations import cleanly and then never return. The model "
            "stays correct and roughly a third slower per batch; pass --kev-fused to insist "
            "if you know they work here."
        )

    # -- CUDA verification ----------------------------------------------------

    def _cuda_probe(self) -> str:
        return self._run(
            [
                "uv",
                "run",
                "--no-sync",
                "--extra",
                "serve",
                "python",
                "-c",
                "import torch; print('CUDA=' + str(torch.cuda.is_available())); print('GPU=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))",
            ],
            self.root,
            env=self._environment(),
        )

    def _verify_cuda(self) -> None:
        if self._gpu_name is None:
            raise KevServerError(
                "Kev needs an NVIDIA GPU with CUDA, and no NVIDIA device was found on "
                "this machine (nvidia-smi is missing or reports no GPUs). Install the "
                "NVIDIA driver, or run with --jev / without --kev to use the hosted "
                "decision model instead."
            )
        self._step("Kev: verifying CUDA availability")
        output = self._cuda_probe()
        if "CUDA=True" not in output:
            # The hardware is there but torch cannot see it: a driver too old for
            # the card, or a torch build without kernels for its architecture.
            raise KevServerError(
                f"CUDA is present ({self._gpu_name}) but the installed PyTorch cannot "
                "use it. This usually means an NVIDIA driver older than the card, or a "
                "torch build without kernels for its architecture; reinstalling with "
                "--torch-backend cu128 fixes the latter."
            )
        gpu = next((line[4:] for line in output.splitlines() if line.startswith("GPU=")), "unknown")
        self._step(f"Kev: CUDA detected on {gpu}")

    # -- probes and reporting -------------------------------------------------

    def _probe_payload(self) -> dict[str, Any]:
        """A request shaped exactly like a real evaluation.

        This has to be built with the same question contract the pipeline uses. An
        earlier version sent an empty ``questions`` dict and reported the latency
        of that as 0.0s -- the server answers a request with nothing to infer
        instantly, so the probe measured nothing at all and cheerfully reported a
        healthy server while every real evaluation took minutes. A benchmark that
        skips the work is worse than no benchmark.
        """
        from . import decision_questions as dq

        candidates = [
            {
                "id": f"probe{i}",
                "title": f"Probe Artist - Probe Song {i}",
                "channel": "Probe Artist",
                "uploader": "Probe Artist",
                "artists": ["Probe Artist"],
                "duration": 200 + i,
                "_composite_score": 200 - i,
            }
            for i in range(3)
        ]
        state, _candidate_states, questions = dq.build_state_and_questions(
            "Probe Artist", "Probe Song", candidates, None, dq.TYPESAFE_DIALECT
        )
        return {"state": state, "model": "kev-latest", "questions": questions}

    def _model_card(self) -> dict[str, Any]:
        """The loaded checkpoint's own description of itself, or {}."""
        info = self._server_info()
        if not isinstance(info, dict):
            return {}
        models = info.get("models")
        if isinstance(models, list) and models and isinstance(models[0], dict):
            return models[0]
        return {}

    def _report_capabilities(self) -> None:
        """Report what the server is *actually* doing, from the server itself.

        ``/v1/models`` is the only honest source. The environment we set describes
        intent; ``cuda_graphs`` being non-null is what the loaded model is really
        doing. Reporting it is the difference between noticing a silent
        degradation now and noticing it from a thousand identical timeouts.
        """
        card = self._model_card()
        if not card:
            self._step("Kev: could not read server capabilities from /v1/models")
            return

        if card.get("cuda_graphs"):
            self._step("Kev: CUDA graphs active")
        else:
            self._step(
                "Kev: WARNING -- CUDA graphs are OFF. kev.serve enables them by default "
                "on CUDA because a serving pass is ~2,000 kernel launches and replaying "
                "graphs cuts warm latency several-fold (measured here: 3338 ms -> 2470 ms "
                "per evaluation). Something is setting KEV_CUDA_GRAPHS=0."
            )

        self._step(
            f"Kev: serving {card.get('run')} on {card.get('device')} via "
            f"{card.get('backend')} ({card.get('dtype')}), temperature "
            f"{card.get('temperature')}"
        )

        prefix = card.get("prefix_cache")
        if isinstance(prefix, dict):
            extra = (
                f", {prefix.get('oom_retries')} OOM retries"
                if prefix.get("oom_retries")
                else ""
            )
            self._step(
                f"Kev: state cache {prefix.get('hits')} hits / {prefix.get('misses')} misses, "
                f"{prefix.get('cached_states')} states held{extra}"
            )

        self._report_batching()

        # Whether the fused path is live is *our* decision, not something
        # /v1/models reports, so it is stated from the decision we made.
        if self._fusable is False:
            self._step(
                "Kev: fused Qwen3.5 kernels DECLINED after a probe (or by flag). The "
                "Gated DeltaNet layers use PyTorch's reference path, which costs roughly "
                "a third more GPU time per batch."
            )
        elif self._fusable is True:
            self._step(f"Kev: fused Qwen3.5 kernels active (flash-linear-attention {FLA_VERSION})")

    def _report_batching(self) -> None:
        """Report whether the server is actually batching concurrent requests.

        Kev drains up to ``MAX_BATCH`` queued requests into one model pass, so
        several in-flight clients finish sooner in total than one at a time. That
        only happens if the client sends several at once, which makes this the one
        number that says whether the GPU is being fed or idling between requests.
        """
        batches = self._model_card().get("batches")
        if isinstance(batches, dict):
            self._step(
                f"Kev: {batches.get('count')} model passes for {batches.get('requests')} "
                f"requests ({batches.get('queued')} queued)"
            )

    def _report_latency(self, budget_seconds: float = 20.0) -> None:
        """Time one real evaluation and warn if the server is too slow to use.

        Knowing a server is *up* says nothing about whether it is usable at batch
        speed. This model can load and answer correctly while being far too slow
        for a thousand-song run -- the usual cause is a hybrid architecture whose
        compiled kernels are absent, leaving a reference PyTorch fallback that
        transformers itself describes as "much slower". Finding that out from a
        thousand identical timeouts, each after a full retry ladder, is the worst
        possible way to find out.

        A probe that times out has already answered the question -- that is the
        slow path, reported with its remedy, not a broken probe.
        """
        try:
            payload = self._probe_payload()
        except Exception:  # noqa: BLE001 - never block startup on the probe
            payload = {
                "state": {"artist": "probe", "song": "probe", "candidates": []},
                "model": "kev-latest",
                "questions": {},
            }

        started = time.monotonic()
        self._step("Kev: timing a real evaluation to confirm batch speed")
        timed_out = False
        try:
            response = requests.post(
                f"{self.url}/v1/systemone",
                json=payload,
                # Ceiling, not budget: we only need to know it is far too slow.
                timeout=budget_seconds * 3,
            )
            elapsed = time.monotonic() - started
        except requests.Timeout:
            timed_out = True
            elapsed = time.monotonic() - started
        except requests.RequestException as exc:
            self._step(f"Kev: latency probe could not reach the server ({type(exc).__name__})")
            return

        question_count = len(payload.get("questions") or {})
        if timed_out:
            self._step(
                f"Kev: TOO SLOW -- one evaluation ({question_count} questions) did not "
                f"finish within {elapsed:.0f}s. The cost is the question count, so lower "
                "--decision-questions first; --kev-timeout raises the ceiling if the "
                "server is healthy but slow."
            )
            if self._fusable is False:
                self._step(
                    "Kev: note the fused Qwen3.5 kernels were already probed and declined, "
                    "so the Qwen3.5 layers are on PyTorch's reference path. On a platform "
                    f"where flash-linear-attention {FLA_VERSION} does run, that is worth "
                    "roughly a third of the GPU time; pass --kev-fused to insist on it."
                )
            self._step(
                "Continuing with the heuristic ranking, which remains fully functional."
            )
            return

        if response.status_code >= 400:
            # The server answered without evaluating anything, so this timing says
            # nothing about inference speed. Reporting it as "within budget" would
            # be the exact failure this probe exists to prevent.
            self._step(
                f"Kev: latency probe was rejected (HTTP {response.status_code}) after "
                f"{elapsed:.1f}s, so it measured no inference. Treating the model as "
                "unverified; if real evaluations time out, lower --decision-questions."
            )
            return

        if elapsed > budget_seconds:
            self._step(
                f"Kev: SLOW -- one evaluation ({question_count} questions) took "
                f"{elapsed:.0f}s (budget {budget_seconds:.0f}s). If every song pays this, "
                "the run will take hours. Lower --decision-questions, or raise "
                "--kev-timeout if the server is healthy but slow."
            )
        else:
            self._step(
                f"Kev: evaluation latency {elapsed:.1f}s for {question_count} questions "
                "-- within budget"
            )

    # -- lifecycle ------------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        """Environment for the Kev server and its installer.

        ``KEV_CUDA_GRAPHS`` is deliberately **not** set. It is tri-state in
        ``kev.checkpoint.LoadOptions.from_env`` -- unset means "let kev.serve
        decide" -- and kev.serve's CUDA default is True, because a serving pass is
        ~2,000 kernel launches and replaying graphs cuts warm latency
        several-fold. Measured on the reference machine, enabling it took a
        31-question evaluation from 3338 ms to 2470 ms. Setting it to "0", as
        this code used to, threw that away.

        ``KEV_FUSED`` **is** set, to the measured answer rather than a constant.
        Leaving it unset asks kev.serve to use ``fused_available()``, which is
        only an import check -- on hardware where the kernels do not run that
        check passes and the server then hangs on its first real request.
        :meth:`_probe_fused_latency` runs an actual evaluation under a hard
        timeout, so "usable" here means "answered", not "imported".

        Both used to be pinned to "0" together, which silently disabled the CUDA
        graphs win and -- because kev.serve only prints its "fused kernels off"
        notice when the flag was *unset* -- disabled the warning too.
        """
        env = os.environ.copy()
        env["UV_TORCH_BACKEND"] = "cu128"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["KEV_BACKEND"] = "torch"
        env["KEV_DTYPE"] = "bf16"
        if self._fusable is False:
            env["KEV_FUSED"] = "0"
        else:
            # Undecided, or the probe said yes. Leaving it unset keeps kev.serve's
            # own notice visible if the kernels turn out to be missing anyway.
            env.pop("KEV_FUSED", None)
        return env

    def _install_termination_handler(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        self._previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_termination)

    def _handle_termination(self, _signum: int, _frame: Any) -> None:
        self.stop()
        raise KeyboardInterrupt

    def _restore_termination_handler(self) -> None:
        if self._previous_sigterm_handler is None:
            return
        try:
            signal.signal(signal.SIGTERM, self._previous_sigterm_handler)
        except ValueError:
            pass
        self._previous_sigterm_handler = None

    def _start_process(self) -> None:
        log_path = self.root / "ytdl-kev.log"
        self.log_file = log_path.open("a", encoding="utf-8")
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            start_new_session = False
        else:
            creationflags = 0
            start_new_session = True
        self.process = subprocess.Popen(
            [
                "uv",
                "run",
                "--no-sync",
                "--extra",
                "serve",
                "python",
                "-m",
                "kev.serve",
                "--run",
                self.run,
                "--port",
                str(self.port),
            ],
            cwd=self.root,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=self._environment(),
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
        self._job_handle = self._create_kill_on_close_job()

    def _create_kill_on_close_job(self) -> int | None:
        if os.name != "nt" or self.process is None:
            return None
        try:
            import ctypes
            from ctypes import wintypes

            class JobObjectBasicLimitInformation(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class IoCounters(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class ExtendedLimitInformation(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JobObjectBasicLimitInformation),
                    ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None
            info = ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = 0x00002000
            configured = kernel32.SetInformationJobObject(
                job, 9, ctypes.byref(info), ctypes.sizeof(info)
            )
            process_handle = getattr(self.process, "_handle", None)
            assigned = (
                configured
                and process_handle is not None
                and kernel32.AssignProcessToJobObject(job, process_handle)
            )
            if not assigned:
                kernel32.CloseHandle(job)
                return None
            return int(job)
        except (AttributeError, OSError):
            return None

    @staticmethod
    def _terminate_job(handle: int) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).TerminateJobObject(handle, 1)
        except (AttributeError, OSError):
            return

    @staticmethod
    def _close_handle(handle: int) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        except (AttributeError, OSError):
            return

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_message = 0.0
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise KevServerError(
                    f"Kev exited during startup; check {self.root / 'ytdl-kev.log'}"
                )
            info = self._server_info()
            if info is not None:
                if not self._server_uses_cuda(info):
                    self.stop()
                    raise KevServerError("Kev started without CUDA; refusing CPU inference")
                self._step(f"Kev: server ready on CUDA at {self.url}")
                return
            now = time.monotonic()
            if now - last_message >= 10:
                self._step("Kev: loading weights; the first model download may take a while")
                last_message = now
            time.sleep(2)
        self.stop()
        raise KevServerError(
            f"Kev did not respond within {self.startup_timeout} seconds; check "
            f"{self.root / 'ytdl-kev.log'}"
        )

    def _server_info(self) -> dict[str, Any] | None:
        try:
            response = requests.get(f"{self.url}/v1/models", timeout=2)
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _healthy(self) -> bool:
        return self._server_info() is not None

    @staticmethod
    def _server_uses_cuda(info: dict[str, Any] | None) -> bool:
        if not isinstance(info, dict):
            return False
        models = info.get("models")
        if not isinstance(models, list):
            return False
        return any(
            str(model.get("device") or "").lower().startswith("cuda")
            for model in models
            if isinstance(model, dict)
        )

    def _is_local_url(self) -> bool:
        host = (urlparse(self.url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

    @staticmethod
    def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            raise KevServerError(f"Required command not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise KevServerError(f"Command timed out: {command[0]}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            message = detail[-1] if detail else "no details"
            raise KevServerError(f"{command[0]} failed: {message[:180]}")
        return completed.stdout

    def _step(self, message: str) -> None:
        self.on_step(message)


def shutil_which(command: str) -> str | None:
    import shutil

    return shutil.which(command)