from __future__ import annotations


import pytest

import os

from ytdl_core.kev_server import KevServerError, KevServerManager


def _manager(tmp_path, steps=None):
    return KevServerManager(tmp_path / "kev", on_step=(steps.append if steps is not None else None))


def _fake_nvidia_smi(monkeypatch, *, returncode=0, stdout="", present=True):
    monkeypatch.setattr("ytdl_core.kev_server.shutil_which", lambda name: "nvidia-smi" if present else None)

    class Completed:
        pass

    completed = Completed()
    completed.returncode = returncode
    completed.stdout = stdout
    completed.stderr = ""

    def fake_run(command, **kwargs):
        return completed

    monkeypatch.setattr("ytdl_core.kev_server.subprocess.run", fake_run)


class TestGpuDetection:
    def test_reports_the_first_gpu(self, monkeypatch, tmp_path):
        _fake_nvidia_smi(monkeypatch, stdout="NVIDIA GeForce RTX 5060 Ti\n")
        assert _manager(tmp_path)._nvidia_gpu_name() == "NVIDIA GeForce RTX 5060 Ti"

    def test_missing_nvidia_smi_is_not_a_cuda_device(self, monkeypatch, tmp_path):
        _fake_nvidia_smi(monkeypatch, present=False)
        assert _manager(tmp_path)._nvidia_gpu_name() is None

    def test_failing_query_is_not_a_cuda_device(self, monkeypatch, tmp_path):
        _fake_nvidia_smi(monkeypatch, returncode=9)
        assert _manager(tmp_path)._nvidia_gpu_name() is None

    def test_empty_output_is_not_a_cuda_device(self, monkeypatch, tmp_path):
        _fake_nvidia_smi(monkeypatch, stdout="\n  \n")
        assert _manager(tmp_path)._nvidia_gpu_name() is None


class TestSyncEnvironment:
    def test_cuda_gpu_excludes_torch_then_installs_cu128(self, monkeypatch, tmp_path):
        """The regression this fixes: detection must not need torch.

        The sync deliberately excludes torch, so probing the hardware by
        importing torch could never succeed on a clean install -- the setup died
        with a missing-module error before it reached the GPU check.
        """
        commands: list[list[str]] = []
        steps: list[str] = []
        manager = _manager(tmp_path, steps)
        _fake_nvidia_smi(monkeypatch, stdout="NVIDIA GeForce RTX 5060 Ti\n")
        monkeypatch.setattr("ytdl_core.kev_server.shutil_which", lambda name: "nvidia-smi")
        monkeypatch.setattr(
            KevServerManager, "_run", lambda self, cmd, cwd, env=None: commands.append(cmd) or ""
        )

        manager._sync_environment()

        assert commands[0][:2] == ["uv", "sync"]
        assert "--no-install-package" in commands[0]
        assert "torch" in commands[0]
        assert commands[1][:3] == ["uv", "pip", "install"]
        assert "cu128" in commands[1]
        assert any("RTX 5060 Ti" in step for step in steps)

    def test_no_gpu_syncs_cpu_torch_instead_of_excluding_it(self, monkeypatch, tmp_path):
        """Leaving torch out with no GPU would leave a broken venv behind."""
        commands: list[list[str]] = []
        manager = _manager(tmp_path)
        _fake_nvidia_smi(monkeypatch, present=False)
        # uv present, nvidia-smi absent.
        monkeypatch.setattr(
            "ytdl_core.kev_server.shutil_which", lambda name: "uv" if name == "uv" else None
        )
        monkeypatch.setattr(
            KevServerManager, "_run", lambda self, cmd, cwd, env=None: commands.append(cmd) or ""
        )

        manager._sync_environment()

        assert commands == [["uv", "sync", "--extra", "serve", "--inexact"]]

    def test_missing_uv_is_reported_clearly(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ytdl_core.kev_server.shutil_which", lambda name: None)
        with pytest.raises(KevServerError, match="uv was not found"):
            _manager(tmp_path)._sync_environment()


class TestVerifyCuda:
    def test_no_gpu_is_reported_before_anything_is_installed(self, monkeypatch, tmp_path):
        manager = _manager(tmp_path)
        manager._gpu_name = None
        monkeypatch.setattr(
            manager,
            "_cuda_probe",
            lambda: (_ for _ in ()).throw(AssertionError("must not probe torch")),
        )
        with pytest.raises(KevServerError, match="no NVIDIA device was found"):
            manager._verify_cuda()

    def test_gpu_present_but_torch_cannot_use_it(self, monkeypatch, tmp_path):
        manager = _manager(tmp_path)
        manager._gpu_name = "NVIDIA GeForce RTX 5060 Ti"
        monkeypatch.setattr(manager, "_cuda_probe", lambda: "CUDA=False\nGPU=none")
        with pytest.raises(KevServerError, match="cannot use it"):
            manager._verify_cuda()

    def test_success_reports_the_gpu(self, monkeypatch, tmp_path):
        steps: list[str] = []
        manager = _manager(tmp_path, steps)
        manager._gpu_name = "NVIDIA GeForce RTX 5060 Ti"
        monkeypatch.setattr(
            manager, "_cuda_probe", lambda: "CUDA=True\nGPU=NVIDIA GeForce RTX 5060 Ti"
        )
        manager._verify_cuda()
        assert any("RTX 5060 Ti" in step for step in steps)


def test_existing_local_server_is_reused(tmp_path):
    steps = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    manager._healthy = lambda: True
    manager._server_info = lambda: {"models": [{"device": "cuda"}]}

    assert manager.start() == "http://127.0.0.1:8009"
    assert any("reusing it" in step for step in steps)


def test_local_startup_runs_lifecycle(monkeypatch, tmp_path):
    steps = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    health_values = iter([False, False, True])
    monkeypatch.setattr(manager, "_healthy", lambda: next(health_values))
    monkeypatch.setattr(manager, "_ensure_repository", lambda: steps.append("repository"))
    monkeypatch.setattr(manager, "_sync_environment", lambda: steps.append("environment"))
    monkeypatch.setattr(manager, "_verify_cuda", lambda: steps.append("cuda"))
    monkeypatch.setattr(manager, "_start_process", lambda: steps.append("process"))
    monkeypatch.setattr(manager, "_wait_until_ready", lambda: steps.append("ready"))
    monkeypatch.setattr(manager, "_report_capabilities", lambda: steps.append("capabilities"))
    monkeypatch.setattr(manager, "_report_latency", lambda: steps.append("latency"))

    assert manager.start() == "http://127.0.0.1:8009"
    assert steps == [
        "repository",
        "environment",
        "cuda",
        # Pinned: an unpinned Hub name tracks the branch, so the same command
        # would not load the same weights next month.
        "Kev: starting jaredpalmer/kev-4b@v1.0 on CUDA through kev.serve",
        "process",
        "ready",
        "capabilities",
        "latency",
    ]


def test_startup_verifies_cuda_graphs_are_actually_on(tmp_path):
    """``/v1/models`` reports ``cuda_graphs: null`` when they are off.

    Reporting what the loaded model is doing -- rather than what the environment
    asked for -- is the only way a silent degradation gets noticed.
    """
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    manager._server_info = lambda: {"models": [{"cuda_graphs": None}]}

    manager._report_capabilities()
    assert any("CUDA graphs are OFF" in step for step in steps)


def test_startup_reports_active_capabilities(tmp_path):
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    manager._server_info = lambda: {
        "models": [
            {
                "cuda_graphs": {"count": 3},
                "run": "jaredpalmer/kev-4b@v1.0",
                "device": "cuda",
                "backend": "torch",
                "dtype": "bfloat16",
                "temperature": 2.41,
                "prefix_cache": {"hits": 2, "misses": 5, "cached_states": 1, "oom_retries": 0},
                "batches": {"count": 4, "requests": 7, "queued": 0},
            }
        ]
    }
    manager._fusable = True

    manager._report_capabilities()
    assert any("CUDA graphs active" in step for step in steps)
    assert any("fused Qwen3.5 kernels active" in step for step in steps)
    assert any("2 hits / 5 misses" in step for step in steps)
    assert not any("WARNING" in step for step in steps)


def test_startup_states_the_fused_decision_it_made(tmp_path):
    """fused is not in /v1/models, so the report states our own decision."""
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    manager._server_info = lambda: {"models": [{"cuda_graphs": {"count": 1}}]}
    manager._fusable = False

    manager._report_capabilities()
    assert any("fused Qwen3.5 kernels DECLINED" in step for step in steps)


def test_batching_report_says_whether_the_gpu_is_fed(tmp_path):
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    manager._server_info = lambda: {"models": [{"batches": {"count": 4, "requests": 7, "queued": 0}}]}

    manager._report_batching()
    assert any("4 model passes for 7 requests" in step for step in steps)


def test_server_environment_does_not_disable_cuda_graphs_or_fused_kernels(tmp_path):
    """The regression this whole exercise exists to prevent.

    ``KEV_CUDA_GRAPHS=0`` threw away a measured 26% latency win, and
    ``KEV_FUSED=0`` did it silently: kev.serve only prints its "fused kernels
    off" notice when the flag is *unset*, so pinning it to 0 disabled the kernels
    *and* the warning that they were off. Neither flag may be set by default.
    """
    env = KevServerManager(tmp_path / "kev")._environment()
    assert "KEV_CUDA_GRAPHS" not in env
    assert "KEV_FUSED" not in env
    assert env["KEV_DTYPE"] == "bf16"


def test_a_measured_fused_decline_reaches_the_server_environment(tmp_path):
    """Once the probe says the kernels do not run, they must actually be declined."""
    manager = KevServerManager(tmp_path / "kev")
    manager._fusable = False
    assert manager._environment()["KEV_FUSED"] == "0"

    manager._fusable = True
    assert "KEV_FUSED" not in manager._environment()


def test_fused_flags_skip_the_probe(tmp_path):
    steps: list[str] = []
    insisting = KevServerManager(tmp_path / "kev", insist_fused=True, on_step=steps.append)
    insisting._fused_kernel_probe = lambda: pytest.fail("must not probe when told what to do")
    insisting._probe_fused_latency = lambda *a, **k: pytest.fail("must not probe when told")
    insisting._install_fused_kernels()
    assert insisting._fusable is True

    declining = KevServerManager(tmp_path / "kev", insist_fused=False, on_step=steps.append)
    declining._fused_kernel_probe = lambda: pytest.fail("must not probe when told what to do")
    declining._install_fused_kernels()
    assert declining._fusable is False
    assert declining._environment()["KEV_FUSED"] == "0"


def test_fused_is_declined_when_it_imports_but_does_not_answer(tmp_path, monkeypatch):
    """Importable is not runnable: on some Triton builds the kernels never return.

    This is the observed failure on RTX 5060 Ti / sm_120 / triton-windows, where a
    trivial Triton kernel launched fine, fla imported at exactly the pinned
    version, and the first real evaluation hung indefinitely.
    """
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    monkeypatch.setattr(manager, "_fused_kernel_probe", lambda: (True, "available"))
    monkeypatch.setattr(manager, "_probe_fused_latency", lambda *a, **k: False)

    manager._install_fused_kernels()

    assert manager._fusable is False
    assert manager._environment()["KEV_FUSED"] == "0"
    assert any("DID NOT ANSWER" in step for step in steps)


def test_fused_is_kept_when_it_answers(tmp_path, monkeypatch):
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    monkeypatch.setattr(manager, "_fused_kernel_probe", lambda: (True, "available"))
    monkeypatch.setattr(manager, "_probe_fused_latency", lambda *a, **k: True)

    manager._install_fused_kernels()

    assert manager._fusable is True
    assert "KEV_FUSED" not in manager._environment()
    assert any("verified end-to-end" in step for step in steps)


def test_windows_triton_series_tracks_the_torch_build(tmp_path, monkeypatch):
    """triton-windows binaries are compiled against one libtorch."""
    manager = KevServerManager(tmp_path / "kev")
    for torch_version, expected in (("2.7.1+cu128", "3.3"), ("2.8.0+cu128", "3.4")):
        monkeypatch.setattr(manager, "_installed_torch_version", lambda v=torch_version: (
            int(v[0]),
            int(v[2]),
        ))
        monkeypatch.setattr(os, "name", "nt")
        requirement = manager._triton_requirement()
        assert requirement is not None and expected in requirement, (
            f"torch {torch_version} should map to triton {expected}, got {requirement}"
        )


def test_slow_server_is_reported_with_the_remedy(monkeypatch, tmp_path):
    """"Up" is not "usable at batch speed"; the gap must be named, not discovered."""
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)

    class SlowResponse:
        status_code = 200

    monkeypatch.setattr("ytdl_core.kev_server.requests.post", lambda *a, **k: SlowResponse())
    ticks = iter([0.0, 45.0, 45.0])
    monkeypatch.setattr("ytdl_core.kev_server.time.monotonic", lambda: next(ticks, 45.0))

    manager._report_latency(budget_seconds=20.0)
    assert any("SLOW" in step for step in steps)
    assert any("kev-timeout" in step for step in steps)


def test_probe_timeout_is_reported_as_slow_with_the_remedy(monkeypatch, tmp_path):
    """A probe that times out has already answered the question.

    It used to report "could not complete", which is the one message that tells
    the user nothing -- while the timeout itself is the actual diagnosis.
    """
    import requests as _requests

    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    monkeypatch.setattr(
        "ytdl_core.kev_server.requests.post",
        lambda *a, **k: (_ for _ in ()).throw(_requests.Timeout("slow")),
    )
    ticks = iter([0.0, 90.0])
    monkeypatch.setattr("ytdl_core.kev_server.time.monotonic", lambda: next(ticks, 90.0))

    manager._report_latency(budget_seconds=20.0)
    assert any("TOO SLOW" in step for step in steps)
    assert any("decision-questions" in step for step in steps)
    assert not any("could not" in step for step in steps)


def test_timeout_names_the_fused_decline_when_that_is_the_cause(monkeypatch, tmp_path):
    """A slow server is usually the reference DeltaNet path, and the fix for that
    is not "install flash-linear-attention" -- it is often already installed and
    declining to run. The message has to reflect what was actually measured."""
    import requests as _requests

    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)
    manager._fusable = False
    monkeypatch.setattr(
        "ytdl_core.kev_server.requests.post",
        lambda *a, **k: (_ for _ in ()).throw(_requests.Timeout("slow")),
    )
    ticks = iter([0.0, 90.0])
    monkeypatch.setattr("ytdl_core.kev_server.time.monotonic", lambda: next(ticks, 90.0))

    manager._report_latency(budget_seconds=20.0)
    assert any("already probed and declined" in step for step in steps)
    assert any("--kev-fused" in step for step in steps)


def test_fast_server_is_not_warned_about(monkeypatch, tmp_path):
    steps: list[str] = []
    manager = KevServerManager(tmp_path / "kev", on_step=steps.append)

    class Response:
        status_code = 200

    monkeypatch.setattr("ytdl_core.kev_server.requests.post", lambda *a, **k: Response())
    ticks = iter([0.0, 1.5])
    monkeypatch.setattr("ytdl_core.kev_server.time.monotonic", lambda: next(ticks, 1.5))

    manager._report_latency(budget_seconds=20.0)
    assert any("within budget" in step for step in steps)
    assert not any("SLOW" in step for step in steps)


def test_remote_endpoint_is_checked_without_local_lifecycle(monkeypatch, tmp_path):
    steps = []
    manager = KevServerManager(
        tmp_path / "kev",
        url="https://kev.example.test",
        on_step=steps.append,
    )
    monkeypatch.setattr(manager, "_healthy", lambda: True)
    monkeypatch.setattr(
        manager,
        "_ensure_repository",
        lambda: (_ for _ in ()).throw(AssertionError("must not clone")),
    )

    assert manager.start() == "https://kev.example.test"
    assert any("remote endpoint" in step for step in steps)
