"""Supervision of a local Decider System One server.

Decider (``Mapika/decider``) ships its own server, ``decider.serve``, behind
``uvicorn``. This module is the counterpart to :mod:`ytdl_core.kev_server` for it,
and it is deliberately much smaller, because most of what Kev's supervision has
to do does not apply here:

* There is no repository to clone. ``decider-ai`` is an ordinary package, so the
  environment is a venv and a pinned install rather than a ``git clone`` and a
  ``git pull`` on every run. That also removes the gap where the *weights* are
  pinned to a tag but the *server code* tracks whatever HEAD happens to be.
* There is no CUDA to verify. ``decider.serve`` picks CUDA, then MPS, then CPU on
  its own, which means this provider also runs on a machine with no NVIDIA card
  -- the case where Kev's supervision stops with "no NVIDIA device was found".
* There is no Triton or fused-kernel archaeology. ``decider.serve`` captures its
  CUDA graphs at start-up when the device has them, so there is no flag whose
  silent default would quietly throw that away.

What is left is the part that is genuinely the same job either way: install the
environment once, start the process, wait until it answers, prove with a real
evaluation that it is fast enough to be worth using, and make sure it is dead
when the run ends.

The environment is left at ``decider.serve``'s own defaults on purpose. It
already warms its graphs, already fits a temperature per question type from the
model's config, and already selects a device. Overriding those from here would
mean re-deciding, on a different machine, questions its authors measured on
theirs.
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

from . import decision_questions as dq
from .procutil import (
    close_handle,
    create_kill_on_close_job,
    nvidia_gpu_name,
    run_command,
    shutil_which,
    spawn_flags,
    terminate_process_tree,
    venv_python,
)

# Pinned so the server is reproducible, the way `--kev-run` pins the Kev
# checkpoint. Bump this deliberately, not incidentally: a different patch release
# can change the served probabilities, and every threshold in the app is set
# against a calibrated probability.
#
# The [serve] extra is not optional decoration. It carries fastapi, uvicorn and
# httpx; the base package is the library and its torch stack, and
# `python -m uvicorn decider.serve:app` against it dies with "No module named
# uvicorn" -- after the whole of torch has already been downloaded.
DEFAULT_PACKAGE = "decider-ai[serve]==1.9.0"

# The Gated DeltaNet layers of the Qwen3.5 backbone are Triton kernels from
# flash-linear-attention, pinned by kev and shipped transitively by decider-ai.
# `fla` imports fine without Triton, but every kernel it exports becomes a
# ModuleNotFoundError-backed stub, and transformers then silently falls back to
# its reference PyTorch layers -- correct, an order of magnitude slower. So the
# pin must match kev's exactly, and Triton must be installed from somewhere:
# it does not exist on PyPI for Windows, and the right source there is
# triton-windows, series-matched to the torch build in the venv.
FLA_VERSION = "0.5.2"
_TRITON_WINDOWS_SERIES = {(2, 7): "3.3", (2, 8): "3.4", (2, 9): "3.5"}

# Via DECIDER_T_BUCKETS/DECIDER_B_BUCKETS. The 1.9.0 defaults (17 length buckets
# up to 8192, 6 batch buckets up to 32) capture 89 CUDA graphs at start-up, about
# 25 GB of reserved VRAM, which on a 16 GB card silently spills into shared
# memory and costs ~4 minutes of start-up. With one request in flight and short
# states, only the (1, small T) cells matter -- measure_decider.py calls that
# shape "b1-short" -- so that is what we ask for unless the operator overrides.
DEFAULT_GRAPH_T_BUCKETS = "64,128,256,512,1024"
DEFAULT_GRAPH_B_BUCKETS = "1"

# Materialise a pinned checkpoint into a local directory. Run inside the model's
# own venv, so huggingface_hub comes from there and not from the application.
_SNAPSHOT_SCRIPT = (
    "import sys\n"
    "from huggingface_hub import snapshot_download\n"
    "print(snapshot_download(sys.argv[1], revision=sys.argv[2], "
    "local_dir=sys.argv[3]))\n"
)

# decider.serve names the loaded checkpoint after the version in the model's own
# decider_config.json, so the client asks the server rather than assuming.
_HEALTH_PATH = "/health"
_MODELS_PATH = "/v1/models"
_SYSTEMONE_PATH = "/v1/systemone"


class DeciderServerError(RuntimeError):
    pass


class DeciderServerManager:
    def __init__(
        self,
        root: Path,
        run: str = "Mapika/decider-4b@v2",
        url: str = "http://127.0.0.1:8010",
        port: int = 8010,
        startup_timeout: int = 900,
        package: str = DEFAULT_PACKAGE,
        install: bool = True,
        device: str | None = None,
        probe_budget_seconds: float = 20.0,
        max_questions: int | None = None,
        # Two values that decider.serve reads from its environment and we want
        # to control from here -- see DEFAULT_GRAPH_T_BUCKETS above.
        graph_t_buckets: str = DEFAULT_GRAPH_T_BUCKETS,
        graph_b_buckets: str = DEFAULT_GRAPH_B_BUCKETS,
        insist_kernels: bool | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.expanduser()
        self.run = run
        self.url = url.rstrip("/")
        self.port = port
        self.startup_timeout = startup_timeout
        self.package = package
        self.install = install
        # None means "let decider.serve decide": CUDA, else MPS, else CPU.
        self.device = device
        self.probe_budget_seconds = float(probe_budget_seconds)
        # None means read Config.DECISION_MAX_QUESTIONS at probe time, so the
        # probe is sized by the same budget the pipeline will really spend.
        self.max_questions = max_questions
        self.graph_t_buckets = str(graph_t_buckets).strip()
        self.graph_b_buckets = str(graph_b_buckets).strip()
        # Tri-state for the fla/GDN kernels, the way Kev's supervisor does it:
        # True = use them unprobed, False = decline them, None = probe and report.
        self.insist_kernels = insist_kernels
        self._kernels_usable: bool | None = None
        self.on_step = on_step or (lambda message: None)
        self.process: subprocess.Popen | None = None
        self.log_file: TextIO | None = None
        self._log_offset = 0
        self._job_handle: int | None = None
        self._previous_sigterm_handler: Any = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> str:
        if not self._is_local_url():
            self._step(f"Decider: checking remote endpoint {self.url}")
            if not self._healthy():
                raise DeciderServerError(f"Decider server did not respond at {self.url}")
            return self.url

        if self._healthy():
            self._step("Decider: server is already available; reusing it")
            self._report_capabilities()
            return self.url

        self._ensure_environment()
        self._step(f"Decider: starting {self.run} through decider.serve")
        self._start_process()
        self._install_termination_handler()
        self._wait_until_ready()
        # Up is not the same as usable at batch speed. Find out now, not from a
        # thousand identical timeouts.
        self._report_capabilities()
        self._report_latency(budget_seconds=self.probe_budget_seconds)
        return self.url

    def stop(self) -> None:
        if self.process is not None:
            terminate_process_tree(self.process, self._job_handle)
        if self._job_handle is not None:
            close_handle(self._job_handle)
            self._job_handle = None
        self._restore_termination_handler()
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self.process = None

    # -- environment ----------------------------------------------------------

    def _venv_python(self) -> Path:
        return venv_python(self.root)

    def _split_run(self) -> tuple[str, str | None]:
        """``repo_id``, and the pinned revision when one was given.

        ``--decider-run`` accepts the same ``repo_id@revision`` spelling as
        ``--kev-run`` so the two read alike. That spelling is *not* understood by
        HuggingFace, though: ``from_pretrained`` rejects ``Mapika/decider-4b@v2``
        outright with "Repo id must use alphanumeric chars", which is why the
        revision is resolved here instead of handed over.
        """
        run = self.run
        if "@" not in run:
            return run, None
        repo_id, _, revision = run.rpartition("@")
        if not repo_id or not revision:
            raise DeciderServerError(
                f"--decider-run {run!r} is not a usable checkpoint name; expected "
                "'repo_id', 'repo_id@revision' or a local directory"
            )
        return repo_id, revision

    def _resolved_model(self) -> str:
        """What to hand ``decider.serve`` as ``DECIDER_MODEL``.

        ``decider.serve`` has exactly one model variable and no revision knob, so a
        pinned checkpoint has to be materialised locally first. Without this,
        pinning is not available at all and the weights would track ``main`` -- the
        exact hole TODO A10 was written about for Kev.
        """
        repo_id, revision = self._split_run()
        if revision is None:
            return repo_id

        target = self.root / "models" / f"{repo_id.replace('/', '--')}@{revision}"
        marker = target / ".ytdl-revision"
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == revision:
            return str(target)

        self._step(f"Decider: fetching {repo_id} at revision {revision}")
        # Run inside the model's own environment. huggingface_hub arrives with
        # decider-ai, and the application's interpreter has no business importing
        # a multi-gigabyte torch stack it never uses.
        output = self._run(
            [
                str(self._venv_python()),
                "-c",
                _SNAPSHOT_SCRIPT,
                repo_id,
                revision,
                str(target),
            ],
            self.root,
            timeout=3600,
        )
        resolved = next((line.strip() for line in reversed(output.splitlines()) if line.strip()), "")
        if not resolved:
            raise DeciderServerError(
                f"Decider: fetching {repo_id}@{revision} produced no path"
            )
        marker_path = Path(resolved) / ".ytdl-revision"
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(revision, encoding="utf-8")
        except OSError:
            # A missing marker only costs a re-check on the next run.
            pass
        return resolved

    def _ensure_environment(self) -> None:
        """Create the venv and install the pinned package, once.

        The installed version is recorded next to the venv so a re-run neither
        reinstalls nor silently drifts: if the pin in the flag changes, the
        marker no longer matches and the install happens again.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        python = self._venv_python()
        if not python.exists():
            self._step(f"Decider: creating environment at {self.root / '.venv'}")
            self._run(["uv", "venv", "--python", "3.12", str(self.root / ".venv")], self.root)

        if not self.install:
            self._step("Decider: install skipped by configuration")
            return

        marker = self.root / "ytdl-decider-package.txt"
        try:
            installed = marker.read_text(encoding="utf-8").strip()
        except OSError:
            installed = ""
        if installed == self.package:
            self._step(f"Decider: {self.package} already installed")
            return

        self._step(f"Decider: installing {self.package} (first run downloads torch)")
        self._run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                self.package,
            ],
            self.root,
            env=self._environment(),
            # The download is large and the machine may be slow, not broken.
            timeout=3600,
        )
        self._verify_importable()
        self._ensure_gpu_torch()
        self._ensure_gdn_kernels()
        marker.write_text(self.package, encoding="utf-8")

    def _torch_probe(self) -> tuple[str | None, bool]:
        """``(version, cuda_available)`` for the torch inside the venv."""
        script = "import torch;print(torch.__version__);print(torch.cuda.is_available())"
        try:
            output = self._run([str(self._venv_python()), "-c", script], self.root, timeout=300)
        except DeciderServerError:
            return None, False
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        version = lines[0] if lines else None
        available = len(lines) > 1 and lines[1].lower() == "true"
        return version, available

    def _ensure_gpu_torch(self) -> None:
        """Install a CUDA build of torch when there is a GPU to install it for.

        A plain ``pip install torch`` resolves to the **CPU-only** wheel on Windows
        -- the local version reads ``+cpu`` -- and nothing complains. decider.serve
        then resolves its device to CPU, loads the weights into RAM, and answers
        every request correctly, orders of magnitude slower. That is the worst
        shape of failure to debug: the run works, the verdicts are right, and a
        1554-song batch simply takes all day.

        So this asks the driver whether a GPU exists and, if so, whether the venv's
        torch can see it. If not, it reinstalls torch from the cu128 index and
        says so out loud. cu128 is not arbitrary: it is the first index whose
        wheels carry Blackwell (sm_120) kernels, which is what current consumer
        cards need. Older builds import cleanly and then fail at runtime with "no
        kernel image is available for execution on the device".

        Skipped when there is no NVIDIA GPU, because CPU is then the honest answer
        rather than a silent downgrade -- and skipped when the operator asked for a
        device explicitly, because that request outranks our inference.
        """
        if self.device:
            return
        gpu = nvidia_gpu_name()
        if gpu is None:
            return

        version, available = self._torch_probe()
        if available:
            return

        current = f" (installed: {version})" if version else ""
        self._step(
            f"Decider: {gpu} is present but the venv's torch cannot use it{current}. "
            "Installing the CUDA (cu128) build of torch -- the CPU wheel resolves by "
            "default and would otherwise run the whole batch in RAM."
        )
        self._run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self._venv_python()),
                "--torch-backend",
                "cu128",
                # uv only honours --torch-backend when torch is actually being
                # installed; without --reinstall this is a no-op against an
                # already-satisfied requirement.
                "--reinstall",
                "torch",
            ],
            self.root,
            timeout=3600,
        )
        version, available = self._torch_probe()
        if available:
            self._step(f"Decider: torch {version} now sees the GPU")
        else:
            # Not fatal. The batch still completes correctly, just slowly, and
            # forcing a failure here would take away the option the operator
            # deliberately chose.
            self._step(
                "Decider: WARNING -- still no CUDA in the venv's torch after installing "
                "the cu128 build. The run will work in RAM and be very slow. Check the "
                "driver, or pass --decider-device explicitly to accept a device."
            )

    def _installed_torch_version(self) -> tuple[int, int] | None:
        """``(major, minor)`` of the torch in the venv, or None if unknown."""
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
        fla. On Windows it does not exist and ``triton-windows`` is the only
        source, so it has to be matched to the torch version deliberately rather
        than taken latest.
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
            f"Decider: torch {version[0]}.{version[1]} has no known triton-windows match; "
            "installing the latest and verifying the kernel actually answers"
        )
        return "triton-windows"

    def _fla_kernel_probe(self) -> tuple[bool, str]:
        """Whether the fla Triton kernels are importable *and* resolve.

        `fla` imports fine without Triton; what breaks is the kernel export, which
        transformers only discovers when the model loads. The honest question is
        therefore about the function, not the package, and asking it from the venv
        that will actually serve is the only way to get the right answer.
        """
        script = (
            "import importlib;"
            "try:\n"
            "    m = importlib.import_module('fla.ops.gated_delta_rule');"
            "    f = getattr(m, 'chunk_gated_delta_rule', None);"
            "    assert f is not None, 'no export';"
            "    print('FLA_OK=' + repr(f.__name__));"
            "except Exception as e:\n"
            "    print('FLA_FAIL=' + type(e).__name__ + ': ' + str(e)[:120]);"
        )
        try:
            completed = subprocess.run(
                [str(self._venv_python()), "-c", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        output = (completed.stdout or "") + (completed.stderr or "")
        lines = [line for line in output.splitlines() if line.strip()]
        if "FLA_OK=" in output:
            return True, lines[-1] if lines else "available"
        return False, (lines[-1] if lines else "no output")[:200]

    def _ensure_gdn_kernels(self) -> None:
        """Install flash-linear-attention + Triton, and prove they resolve.

        Best-effort by design: a missing GDN Triton kernel costs speed, not
        correctness -- decider falls back to PyTorch's reference layers and
        answers fine. But it must never be *silent*, because the same slot a
        slower forward takes is the one a thousand timeouts would have found from
        the other end; see the TODO A2/A3 note about an import check passing where
        a real kernel run silently does not.

        Skipped entirely when the operator asked for a device explicitly, because
        that request outranks our inference -- and because a fused-path install
        may not even work under MPS.
        """
        if self.insist_kernels is False:
            self._kernels_usable = False
            self._step("Decider: --no-decider-kernels given; leaving Triton and fla alone.")
            return
        if self.insist_kernels is True:
            self._kernels_usable = True
            self._step("Decider: --decider-kernels given; insisting on the GDN Triton kernels")
            return

        gpu_name = nvidia_gpu_name()
        if gpu_name is None:
            # Without a GPU the CPU path is what decider.serve's own default is;
            # no point forcing triton-windows there.
            self._kernels_usable = False
            return

        available, detail = self._fla_kernel_probe()
        if not available:
            packages = [f"flash-linear-attention=={FLA_VERSION}"]
            triton_requirement = self._triton_requirement()
            if triton_requirement:
                packages.append(triton_requirement)
            self._step(
                f"Decider: installing GDN Triton kernels ({', '.join(packages)}); "
                f"reason: {detail}"
            )
            try:
                self._run(
                    ["uv", "pip", "install", "--python", str(self._venv_python()), *packages],
                    self.root,
                    env=self._environment(),
                )
            except DeciderServerError as exc:
                self._kernels_usable = False
                self._step(
                    f"Decider: GDN kernels could not be installed ({exc}). The model still "
                    "works but the Gated DeltaNet layers fall back to PyTorch's reference "
                    "path, which is slower. Install them manually with: uv pip install "
                    f"--python .venv flash-linear-attention=={FLA_VERSION}"
                )
                return
            available, detail = self._fla_kernel_probe()

        if available:
            self._kernels_usable = True
            self._step(
                f"Decider: GDN Triton kernels resolvable (flash-linear-attention {FLA_VERSION})"
            )
        else:
            self._kernels_usable = False
            self._step(
                f"Decider: GDN Triton kernels still do not resolve after install ({detail}). "
                "The model answers fine but is slower. Check the driver, or pass "
                "--decider-device cpu to stop paying Triton's import cost."
            )

    def _verify_importable(self) -> None:
        """Prove the server can actually be imported, before anything is started.

        The failure this catches is quiet until it is expensive: a missing extra
        leaves a venv with a complete torch stack and no uvicorn, so the process
        dies instantly but only after the install spent minutes downloading.
        Asking the interpreter takes a second and names the actual problem.
        """
        try:
            self._run(
                [str(self._venv_python()), "-c", "import decider.serve, uvicorn"],
                self.root,
                timeout=300,
            )
        except DeciderServerError as exc:
            raise DeciderServerError(
                f"{self.package} is installed but its server cannot be imported ({exc}). "
                "If the message names a missing module, the install is missing an extra: "
                "the [serve] extra carries fastapi and uvicorn."
            ) from exc

    def _environment(self) -> dict[str, str]:
        """Environment for the installer and the server.

        ``DECIDER_DEVICE`` is set only when the operator asked for one. Left
        unset it is the string ``auto``, which is decider.serve's own default and
        resolves to CUDA, else MPS, else CPU -- so the one case that must work
        without configuration is the case we do not touch.

        ``DECIDER_T_BUCKETS`` / ``DECIDER_B_BUCKETS`` narrow the CUDA-graph grid
        to the cells our traffic needs, rather than having the whole 89-cell
        default grid captured and then mostly unused. See
        DEFAULT_GRAPH_T_BUCKETS above for why. They are only set when non-empty:
        an empty string means "let decider.serve decide".
        """
        env = os.environ.copy()
        if self.device:
            env["DECIDER_DEVICE"] = self.device
        else:
            env.pop("DECIDER_DEVICE", None)
        if self.graph_t_buckets:
            env["DECIDER_T_BUCKETS"] = self.graph_t_buckets
        else:
            env.pop("DECIDER_T_BUCKETS", None)
        if self.graph_b_buckets:
            env["DECIDER_B_BUCKETS"] = self.graph_b_buckets
        else:
            env.pop("DECIDER_B_BUCKETS", None)
        return env

    def _start_process(self) -> None:
        log_path = self.log_path
        self.log_file = log_path.open("a", encoding="utf-8")
        # The log is append-only, so it still holds every earlier attempt. Record
        # where this one starts: quoting the tail of a *previous* failure as the
        # reason for this one is worse than quoting nothing.
        self._log_offset = self.log_file.tell()
        creationflags, start_new_session = spawn_flags()
        env = self._environment()
        # The checkpoint is chosen by the server through its environment, the
        # same way its own serve.sh does it.
        env["DECIDER_MODEL"] = self._resolved_model()
        self.process = subprocess.Popen(
            [
                str(self._venv_python()),
                "-m",
                "uvicorn",
                "decider.serve:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=self.root,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
        self._job_handle = create_kill_on_close_job(self.process)

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

    # -- readiness and reporting ----------------------------------------------

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_message = 0.0
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise DeciderServerError(
                    f"Decider exited during startup: {self._log_tail()}. "
                    f"Full log: {self.log_path}"
                )
            if self._server_info() is not None:
                self._step(f"Decider: server ready at {self.url}")
                return
            now = time.monotonic()
            if now - last_message >= 10:
                self._step("Decider: loading weights; the first model download may take a while")
                last_message = now
            time.sleep(2)
        self.stop()
        raise DeciderServerError(
            f"Decider did not respond within {self.startup_timeout} seconds. "
            f"Last output: {self._log_tail()}"
        )

    @property
    def log_path(self) -> Path:
        return self.root / "ytdl-decider.log"

    def _log_tail(self, lines: int = 3) -> str:
        """The last few real lines this run produced, or a placeholder.

        A bare "check the log" sends the operator to a file they then have to
        open and scroll. The reason a server exits during start-up is almost
        always in its last line -- a missing module, a bad revision, an OOM --
        and that line is worth quoting.
        """
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                if self._log_offset:
                    handle.seek(self._log_offset)
                content = handle.read()
        except OSError:
            return "no output was captured"
        tail = [line.strip() for line in content.splitlines() if line.strip()]
        if not tail:
            return "no output was captured"
        return " | ".join(tail[-lines:])[:300]

    def _server_info(self) -> dict[str, Any] | None:
        """Whatever the server says about itself, or None if it is not up."""
        for path in (_HEALTH_PATH, _MODELS_PATH):
            try:
                response = requests.get(f"{self.url}{path}", timeout=2)
            except requests.RequestException:
                continue
            if response.status_code != 200:
                continue
            try:
                data = response.json()
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
        return None

    def _engine_stats(self) -> dict[str, Any] | None:
        """What the engine reports about itself, or None if /stats is not live.

        The one number that matters is ``graphs``: it is the count of CUDA graphs
        captured at start-up, and it is the difference between a 12-second start
        and a 4-minute one on this hardware.
        """
        try:
            response = requests.get(f"{self.url}/stats", timeout=2)
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

    def _report_capabilities(self) -> None:
        """Report what the server says it is doing, in its own words.

        ``/health`` is the only honest source. The environment this module sets
        describes intent; what the loaded model reports is what is really
        happening. The one that matters most here is the device, because a CPU
        fallback answers correctly and is orders of magnitude slower, which is
        exactly the kind of degradation a thousand identical timeouts would
        otherwise be the first to reveal.
        """
        info = self._server_info() or {}
        name = info.get("model") or self.run
        device = info.get("device") or "unknown device"
        temperature = info.get("temperature")
        by_type = info.get("temperature_by_type")

        detail = f"Decider: serving {name} on {device}"
        if isinstance(by_type, dict) and by_type:
            rendered = ", ".join(f"{key} {value}" for key, value in sorted(by_type.items()))
            detail += f", temperatures {rendered}"
        elif isinstance(temperature, (int, float)):
            detail += f", temperature {temperature}"
        self._step(detail)

        if isinstance(device, str) and device.strip().lower().startswith("cpu"):
            self._step(
                "Decider: running on CPU. It answers correctly and it will be slow -- "
                "expect seconds per evaluation, not milliseconds. Lower "
                "--decision-questions, or run with --kev on an NVIDIA GPU."
            )
        if info.get("rejected_too_large"):
            self._step(
                f"Decider: {info['rejected_too_large']} request(s) already rejected as too "
                "large; lower --decision-questions."
            )

        stats = self._engine_stats()
        if stats is not None:
            self._report_graphs(stats)
        elif self._kernels_usable is False and device is not None:
            self._step(
                "Decider: no /stats endpoint; the engine grid and GDN-kernel state are "
                "not observable on this server version."
            )

    def _report_graphs(self, stats: dict[str, Any]) -> None:
        """How many CUDA graphs were captured, and what kernel path is live.

        Both are the difference between a usable and an unusable batch. The count
        is what a 12-second start vs a 4-minute one looks like; the GDN path is
        how one evaluation at 31 questions reads 1.8 s rather than tens of
        seconds. Neither is in /health, so neither is reported anywhere else.
        """
        graphs = stats.get("graphs")
        if isinstance(graphs, int):
            sealed = "sealed" if stats.get("sealed") else "unsealed"
            self._step(f"Decider: engine captured {graphs} CUDA graphs ({sealed})")
            if graphs == 0:
                self._step(
                    "Decider: WARNING -- no CUDA graphs were captured. decider.serve enables "
                    "them by default on CUDA; if this was meant to be disabled, check "
                    "DECIDER_WARMUP and the engine device."
                )

        engine = stats.get("engine") or {}
        if isinstance(engine, dict):
            captures = engine.get("graph_captures")
            eager = engine.get("eager_forwards")
            if isinstance(captures, int) and isinstance(eager, int):
                self._step(
                    f"Decider: engine forwards: {captures} graph captures, {eager} eager"
                )

        # Our own decision about the GDN kernels is ours to report, the same way
        # Kev's is; /stats does not say it.
        if self._kernels_usable is False:
            self._step(
                "Decider: GDN kernels DECLINED or unavailable. The Qwen3.5 Gated DeltaNet "
                "layers run PyTorch's reference path, which costs roughly an order of "
                "magnitude more GPU time per evaluation. Install with: uv pip install "
                f"--python .venv flash-linear-attention=={FLA_VERSION}"
            )
        elif self._kernels_usable is True:
            self._step(
                f"Decider: GDN Triton kernels active (flash-linear-attention {FLA_VERSION})"
            )

    def _report_latency(self, budget_seconds: float = 20.0) -> None:
        """Time one real evaluation and warn if the server is too slow to use.

        Knowing a server is *up* says nothing about whether it is usable at batch
        speed. This one can load and answer correctly while being far too slow
        for a thousand-song run. Finding that out from a thousand identical
        timeouts, each after a full retry ladder, is the worst possible way to
        find out.

        The probe has to be the same *size* as the real payload. Cost is
        superlinear in the question count on this hardware -- measured with
        Kev's server at one request in flight: 19 questions 3.1 s, 25 questions
        10.1 s, 31 questions 29.8 s -- so a small probe sits on the flat part of
        the curve and passes while the questions the pipeline actually sends are
        already past the knee. The candidate count is therefore derived from the
        question budget the pipeline will really spend, using the same arithmetic
        as ``JevClassifier._trim_to_question_budget``.
        """
        try:
            payload = self._probe_payload()
        except Exception:  # noqa: BLE001 - never block startup on the probe
            payload = {"state": {"target": {"artist": "probe", "song": "probe"}}, "questions": {}}

        started = time.monotonic()
        self._step("Decider: timing a real evaluation to confirm batch speed")
        timed_out = False
        try:
            response = requests.post(
                f"{self.url}{_SYSTEMONE_PATH}",
                json=payload,
                # Ceiling, not budget: we only need to know it is far too slow.
                timeout=budget_seconds * 3,
            )
            elapsed = time.monotonic() - started
        except requests.Timeout:
            timed_out = True
            elapsed = time.monotonic() - started
        except requests.RequestException as exc:
            self._step(
                f"Decider: latency probe could not reach the server ({type(exc).__name__})"
            )
            return

        question_count = len(payload.get("questions") or {})
        if timed_out:
            self._step(
                f"Decider: TOO SLOW -- one evaluation ({question_count} questions) did not "
                f"finish within {elapsed:.0f}s. The cost is the question count, so lower "
                "--decision-questions first; --decision-timeout raises the ceiling if the "
                "server is healthy but slow."
            )
            self._step("Continuing with the heuristic ranking, which remains fully functional.")
            return

        if response.status_code >= 400:
            # The server answered without evaluating anything, so this timing says
            # nothing about inference speed. Reporting it as "within budget" would
            # be the exact failure this probe exists to prevent.
            self._step(
                f"Decider: latency probe was rejected (HTTP {response.status_code}) after "
                f"{elapsed:.1f}s, so it measured no inference. Treating the model as "
                "unverified; if real evaluations time out, lower --decision-questions."
            )
            return

        if elapsed > budget_seconds:
            self._step(
                f"Decider: SLOW -- one evaluation ({question_count} questions) took "
                f"{elapsed:.0f}s (budget {budget_seconds:.0f}s). If every song pays this, "
                "the run will take hours. Lower --decision-questions, or raise "
                "--decision-timeout if the server is healthy but slow."
            )
        else:
            self._step(
                f"Decider: evaluation latency {elapsed:.1f}s for {question_count} questions "
                "-- within budget"
            )

    def _probe_payload(self) -> dict[str, Any]:
        budget = self._question_budget()
        affordable = max(1, (budget - 1) // max(1, len(dq.DIMENSIONS)))
        candidates = [
            {
                "id": f"probe{index}",
                "title": f"Probe Artist - Probe Song {index}",
                "channel": "Probe Artist",
                "uploader": "Probe Artist",
                "artists": ["Probe Artist"],
                "duration": 200 + index,
                "_composite_score": 200 - index,
            }
            for index in range(affordable)
        ]
        state, _candidate_states, questions = dq.build_state_and_questions(
            "Probe Artist", "Probe Song", candidates, None, dq.TYPESAFE_DIALECT
        )
        # No `model` key: decider.serve does not need one, and naming a
        # checkpoint here would only risk a 404 against the loaded one.
        return {"state": state, "questions": questions}

    def _question_budget(self) -> int:
        if self.max_questions is not None:
            return max(1, int(self.max_questions))
        from .config import Config

        return max(1, int(Config.DECISION_MAX_QUESTIONS))

    # -- small helpers --------------------------------------------------------

    def _is_local_url(self) -> bool:
        host = (urlparse(self.url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

    def _run(
        self,
        command: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: int = 1800,
    ) -> str:
        return run_command(command, cwd, env, error=DeciderServerError, timeout=timeout)

    def _step(self, message: str) -> None:
        self.on_step(message)


__all__ = ["DeciderServerError", "DeciderServerManager", "DEFAULT_PACKAGE", "shutil_which"]
