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


class KevServerError(RuntimeError):
    pass


class KevServerManager:
    def __init__(
        self,
        root: Path,
        run: str = "jaredpalmer/kev-4b",
        url: str = "http://127.0.0.1:8009",
        port: int = 8009,
        startup_timeout: int = 900,
        update: bool = True,
        on_step: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.expanduser()
        self.run = run
        self.url = url.rstrip("/")
        self.port = port
        self.startup_timeout = startup_timeout
        self.update = update
        self.on_step = on_step or (lambda message: None)
        self.process: subprocess.Popen | None = None
        self.log_file: TextIO | None = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self._job_handle: int | None = None

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
            return self.url

        self._ensure_repository()
        self._sync_environment()
        self._verify_cuda()
        self._step(f"Kev: starting {self.run} on CUDA through kev.serve")
        self._start_process()
        self._install_signal_handlers()
        self._wait_until_ready()
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
        self._restore_signal_handlers()
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self.process = None

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

    def _sync_environment(self) -> None:
        if shutil_which("uv") is None:
            raise KevServerError("uv was not found; install it to manage Kev")
        self._step("Kev: syncing Python dependencies")
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
        if "CUDA=True" in self._cuda_probe():
            return
        python_path = (
            self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        self._step("Kev: installing CUDA-enabled PyTorch")
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
                "torch>=2.6,<2.9",
            ],
            self.root,
            env=self._environment(),
        )

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
        self._step("Kev: verifying CUDA availability")
        output = self._cuda_probe()
        if "CUDA=True" not in output:
            self._step("Kev: CUDA PyTorch was not detected; installing the CUDA build")
            self._install_cuda_torch()
            output = self._cuda_probe()
        if "CUDA=True" not in output:
            raise KevServerError(
                "CUDA is not available in the Kev environment; refusing to start on CPU"
            )
        gpu = next((line[4:] for line in output.splitlines() if line.startswith("GPU=")), "unknown")
        self._step(f"Kev: CUDA detected on {gpu}")

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["UV_TORCH_BACKEND"] = "cu128"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["KEV_BACKEND"] = "torch"
        env["KEV_DTYPE"] = "bf16"
        env["KEV_CUDA_GRAPHS"] = "0"
        env["KEV_FUSED"] = "0"
        return env

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def handle(signum, frame):
            self.stop()
            raise KeyboardInterrupt

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except ValueError:
                pass
        self._previous_signal_handlers.clear()

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
