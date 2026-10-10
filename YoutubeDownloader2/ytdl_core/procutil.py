"""Process and venv plumbing shared by the local decision servers.

Two decision servers can be supervised -- ``kev.serve`` and ``decider.serve`` --
and the awkward parts of supervising either one are not about decisions at all.
They are about Windows.

A decision server is a long-lived heavyweight process holding a GPU, started
once per run and killed at the end of it. On Windows, killing the parent is not
enough: the interpreter spawns worker processes, and orphaned ones keep the
device locked long after the run "finished". The fix is a Job Object with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, which makes the kernel reap the whole
tree the moment the last handle closes -- including when this process is killed
without ever running cleanup. On POSIX the equivalent is a process group, which
is what ``start_new_session`` gives us.

Everything here is deliberately provider-agnostic. It knows how to run a command
and how to clean up afterwards; it has no opinion about what was launched.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

# JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
_KILL_ON_JOB_CLOSE = 0x00002000
# JobObjectExtendedLimitInformation
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class CommandError(RuntimeError):
    """A setup command failed, timed out, or is not installed."""


def shutil_which(command: str) -> str | None:
    return shutil.which(command)


def nvidia_gpu_name() -> str | None:
    """Name of the first NVIDIA GPU, or None. Does not require torch.

    Detecting the hardware by importing torch is circular: the environment being
    checked is the one whose torch is in question, so asking it "do you have a
    GPU" answers the wrong question. ``nvidia-smi`` is the driver, not the
    framework, and it is the thing that decides whether a CUDA build is worth
    installing at all.
    """
    if os.name == "nt" and shutil_which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
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
    first = next((line.strip() for line in (completed.stdout or "").splitlines() if line.strip()), "")
    return first or None


def venv_python(root: Path) -> Path:
    """The interpreter inside a virtual environment rooted at ``root``."""
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    error: type[RuntimeError] = CommandError,
    timeout: int = 1800,
) -> str:
    """Run a setup command and return its stdout.

    The last line of stderr is what gets quoted on failure. Long enough to be
    the actual cause, short enough to fit in a panel -- setup commands routinely
    print a traceback followed by the one line that matters.
    """
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise error(f"Required command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise error(f"Command timed out: {command[0]}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        message = detail[-1] if detail else "no details"
        raise error(f"{command[0]} failed: {message[:180]}")
    return completed.stdout


def spawn_flags() -> tuple[int, bool]:
    """``(creationflags, start_new_session)`` for a hidden, separable child."""
    if os.name == "nt":
        # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        return (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            False,
        )
    return 0, True


def create_kill_on_close_job(process: subprocess.Popen[Any] | None) -> int | None:
    """A Job Object that kills ``process`` and its tree when closed.

    Returns the handle, or None when this is not Windows or the job could not be
    created or assigned. None is not an error: the caller falls back to
    ``taskkill /T``, which does not reap on an unclean exit. Best effort by
    design -- the alternative is a GPU that stays locked.
    """
    if os.name != "nt" or process is None:
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
        info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        configured = kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        )
        process_handle = getattr(process, "_handle", None)
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


def terminate_job(handle: int) -> None:
    """Kill everything in the job. Best effort."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.WinDLL("kernel32", use_last_error=True).TerminateJobObject(handle, 1)
    except (AttributeError, OSError):
        return


def close_handle(handle: int) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
    except (AttributeError, OSError):
        return


def terminate_process_tree(process: subprocess.Popen[Any], job_handle: int | None) -> None:
    """Stop a supervised child and everything it spawned.

    Job first, then the OS-specific tree kill, then a plain kill. The order
    matters: the job is the only step that works when the parent is terminated
    rather than exiting, and ``taskkill /T`` is the fallback for when the job
    could not be created.
    """
    if process.poll() is not None:
        return
    if job_handle is not None:
        terminate_job(job_handle)
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    else:
        _signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name != "nt":
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        _signal_group(process.pid, sigkill)
    try:
        process.kill()
    except OSError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _signal_group(pid: int, sig: int) -> None:
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if killpg is None or getpgid is None:
        return
    try:
        killpg(getpgid(pid), sig)
    except (ProcessLookupError, OSError):
        pass
