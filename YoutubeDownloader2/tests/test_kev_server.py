from __future__ import annotations


import pytest

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
    monkeypatch.setattr(manager, "_report_latency", lambda: steps.append("latency"))

    assert manager.start() == "http://127.0.0.1:8009"
    assert steps == [
        "repository",
        "environment",
        "cuda",
        "Kev: starting jaredpalmer/kev-4b on CUDA through kev.serve",
        "process",
        "ready",
        "latency",
    ]


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
    assert any("flash-linear-attention" in step for step in steps)
    assert not any("could not" in step for step in steps)


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
