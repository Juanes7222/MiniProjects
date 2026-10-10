"""Supervision of a local Decider server.

Most of what Kev's supervision has to do -- verifying CUDA, installing Triton,
probing fused kernels, refusing to start without a GPU -- has no Decider
equivalent, because decider.serve resolves its own device and ships its own
server. What is left is the part that is genuinely the same job either way, and
that is what these tests pin: the environment is created once and not reinstalled
on every run, the checkpoint reaches the server, readiness is decided by asking
the server, and a CPU fallback is reported rather than discovered through a
thousand identical timeouts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from ytdl_core import decision_questions as dq
from ytdl_core.config import Config
from ytdl_core.decider_server import (
    DEFAULT_GRAPH_B_BUCKETS,
    DEFAULT_GRAPH_T_BUCKETS,
    DEFAULT_PACKAGE,
    DeciderServerError,
    DeciderServerManager,
)
from ytdl_core.jev import JevClassifier


@pytest.fixture(autouse=True)
def _no_host_gpu(monkeypatch):
    """Never let the machine running the suite decide what these tests assert.

    Without this, a developer with an NVIDIA card silently takes the CUDA branch
    and the suite passes or fails depending on whose laptop it is on. Tests that
    care about the GPU override this explicitly.
    """
    monkeypatch.setattr("ytdl_core.decider_server.nvidia_gpu_name", lambda: None)


def _manager(tmp_path, steps=None, **kwargs):
    return DeciderServerManager(
        tmp_path / "decider",
        on_step=(steps.append if steps is not None else None),
        **kwargs,
    )


def _running(monkeypatch, info: dict | None = None) -> list[str]:
    """Make every readiness probe answer, and report what was asked."""
    asked: list[str] = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    payload = info if info is not None else {"model": "decider-v2.1", "device": "cuda:0"}

    def fake_get(url, timeout=None):
        asked.append(url)
        return Response(payload)

    monkeypatch.setattr(requests, "get", fake_get)
    return asked


class TestReadyServerIsReused:
    def test_an_existing_local_server_is_reused(self, monkeypatch, tmp_path):
        _running(monkeypatch)
        manager = _manager(tmp_path)
        assert manager.start() == "http://127.0.0.1:8010"

    def test_a_cpu_server_is_reported_not_refused(self, monkeypatch, tmp_path):
        # The whole point of Decider here is that it runs without an NVIDIA card.
        # Kev's supervisor stops with "no NVIDIA device was found"; this one says
        # so out loud and carries on, because a CPU server does answer.
        _running(monkeypatch, {"model": "decider-v2b", "device": "cpu"})
        steps: list[str] = []
        assert _manager(tmp_path, steps).start() == "http://127.0.0.1:8010"
        assert any("on CPU" in step for step in steps)

    def test_a_remote_endpoint_is_checked_without_a_local_lifecycle(self, monkeypatch, tmp_path):
        _running(monkeypatch)
        manager = _manager(tmp_path, url="http://gpu-box.internal:8010")
        assert manager.start() == "http://gpu-box.internal:8010"

    def test_an_unreachable_remote_endpoint_is_an_error(self, monkeypatch, tmp_path):
        def fake_get(url, timeout=None):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", fake_get)
        with pytest.raises(DeciderServerError, match="did not respond"):
            _manager(tmp_path, url="http://gpu-box.internal:8010").start()


class TestEnvironment:
    def test_it_installs_the_pinned_package_into_a_venv(self, monkeypatch, tmp_path):
        commands: list[list[str]] = []

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            commands.append(command)
            if command[:2] == ["uv", "venv"]:
                # Make the interpreter exist so the install step is reached.
                python = tmp_path / "decider" / ".venv" / "Scripts" / "python.exe"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.touch()
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        _running(monkeypatch)
        _manager(tmp_path)._ensure_environment()

        assert commands[0][:2] == ["uv", "venv"]
        assert DEFAULT_PACKAGE in commands[1]
        assert commands[1][:2] == ["uv", "pip"]

    def test_a_second_run_does_not_reinstall(self, monkeypatch, tmp_path):
        python = tmp_path / "decider" / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        calls: list[list[str]] = []

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            calls.append(command)
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        steps: list[str] = []
        _manager(tmp_path, steps)._ensure_environment()
        installs = len([c for c in calls if c[:2] == ["uv", "pip"]])
        assert installs == 1

        # The marker written by the first pass is what makes the second one a no-op.
        _manager(tmp_path, steps)._ensure_environment()
        installs = len([c for c in calls if c[:2] == ["uv", "pip"]])
        assert installs == 1
        assert any("already installed" in step for step in steps)

    def test_skip_install_leaves_the_package_alone(self, monkeypatch, tmp_path):
        python = tmp_path / "decider" / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        calls: list[list[str]] = []
        monkeypatch.setattr(
            "ytdl_core.decider_server.run_command",
            lambda command, **kwargs: calls.append(command),
        )
        steps: list[str] = []
        _manager(tmp_path, steps, install=False)._ensure_environment()
        # The venv may still be created -- that costs nothing and needs no
        # network. What must not happen is the install.
        assert [c for c in calls if c[:2] == ["uv", "pip"]] == []
        assert any("install skipped" in step for step in steps)

    def test_the_device_is_left_to_the_server_unless_asked(self, tmp_path):
        # Unset is `auto`, which is decider.serve's own default and resolves to
        # CUDA, else MPS, else CPU. Setting it to something here would take that
        # decision away from the one that can actually see the machine.
        assert "DECIDER_DEVICE" not in _manager(tmp_path)._environment()
        assert _manager(tmp_path, device="cuda:1")._environment()["DECIDER_DEVICE"] == "cuda:1"


class TestGraphBucketEnvironment:
    """The default grid of CUDA graphs decider.serve captures is 89 cells and
    about 25 GB of reserved VRAM on a 4B checkpoint, which on a 16 GB card spills
    into shared memory and costs ~4 minutes of start-up. Narrowing the grid is
    the difference between a usable and an unusable provider; these tests pin the
    narrowed default and the way it reaches the process.
    """

    def test_the_default_narrows_the_grid_for_a_serial_client(self, tmp_path):
        env = _manager(tmp_path)._environment()
        assert env["DECIDER_T_BUCKETS"] == DEFAULT_GRAPH_T_BUCKETS
        assert env["DECIDER_B_BUCKETS"] == DEFAULT_GRAPH_B_BUCKETS
        # A serial client needs batch 1 only, and the pipeline's state is short,
        # so the narrow default is all small cells.
        assert DEFAULT_GRAPH_B_BUCKETS == "1"
        assert all(int(t) <= 1024 for t in DEFAULT_GRAPH_T_BUCKETS.split(","))

    def test_an_empty_override_leaves_decider_serve_its_default(self, tmp_path):
        # An operator who wants the wide grid back has to be able to get it.
        env = _manager(tmp_path, graph_t_buckets="", graph_b_buckets="")._environment()
        assert "DECIDER_T_BUCKETS" not in env
        assert "DECIDER_B_BUCKETS" not in env

    def test_a_custom_override_reaches_the_environment(self, tmp_path):
        env = _manager(
            tmp_path, graph_t_buckets="64,128,256", graph_b_buckets="1,2"
        )._environment()
        assert env["DECIDER_T_BUCKETS"] == "64,128,256"
        assert env["DECIDER_B_BUCKETS"] == "1,2"

    def test_the_server_process_receives_the_buckets(self, monkeypatch, tmp_path):
        manager = _manager(tmp_path, run="Mapika/decider-4b")
        (tmp_path / "decider").mkdir(parents=True, exist_ok=True)
        captured: dict = {}

        class Fake:
            pid = 1234

            def poll(self):
                return None

        def fake_popen(command, cwd=None, env=None, **kwargs):
            captured.update({"env": env})
            return Fake()

        monkeypatch.setattr("ytdl_core.decider_server.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "ytdl_core.decider_server.create_kill_on_close_job", lambda process: None
        )
        monkeypatch.setattr("ytdl_core.decider_server.spawn_flags", lambda: (0, True))
        monkeypatch.setattr("ytdl_core.decider_server.terminate_process_tree", lambda *a: None)

        manager._start_process()
        manager.log_file.close()
        manager.log_file = None
        manager.process = None
        assert captured["env"]["DECIDER_T_BUCKETS"] == DEFAULT_GRAPH_T_BUCKETS
        assert captured["env"]["DECIDER_B_BUCKETS"] == DEFAULT_GRAPH_B_BUCKETS


class TestGdnKernels:
    """`fla` imports cleanly without Triton, so an import check cannot see the
    missing kernels, and transformers quietly falls back to its reference path --
    correct, an order of magnitude slower. What has to be asked is whether the
    *kernel* resolves, and the answer reported either way.
    """

    def _fla_probe_script(self) -> str:
        return "import importlib\n"

    def test_no_gpu_leaves_the_kernels_alone(self, tmp_path):
        manager = _manager(tmp_path)
        manager._fla_kernel_probe = pytest.fail  # would raise if called
        manager._ensure_gdn_kernels()
        assert manager._kernels_usable is False

    def test_a_resolvable_kernel_is_reported_and_not_reinstalled(self, monkeypatch, tmp_path):
        calls: list[list[str]] = []
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )
        monkeypatch.setattr("ytdl_core.decider_server.run_command", lambda c, **k: calls.append(c) or "")

        manager = _manager(tmp_path)
        manager._fla_kernel_probe = lambda: (True, "FLA_OK='chunk_gated_delta_rule'")
        steps: list[str] = []
        manager.on_step = steps.append
        manager._ensure_gdn_kernels()

        assert manager._kernels_usable is True
        assert [c for c in calls if c[:2] == ["uv", "pip"]] == []
        assert any("resolvable" in step for step in steps)

    def test_an_unresolvable_kernel_is_installed_and_reprobed(self, monkeypatch, tmp_path):
        installs: list[list[str]] = []
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )
        monkeypatch.setattr("ytdl_core.decider_server.os.name", "nt")
        monkeypatch.setattr(
            "ytdl_core.decider_server.subprocess.run",
            lambda command, **k: type("R", (), {"returncode": 0, "stdout": "2.8.0\n"})(),
        )

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            installs.append(command)
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)

        manager = _manager(tmp_path)
        probes = iter([(False, "ModuleNotFoundError: triton"), (True, "FLA_OK")])
        manager._fla_kernel_probe = lambda: next(probes)
        steps: list[str] = []
        manager.on_step = steps.append
        manager._ensure_gdn_kernels()

        assert manager._kernels_usable is True
        install = next(c for c in installs if c[:2] == ["uv", "pip"])
        assert any("flash-linear-attention==" in part for part in install[5:])
        # On Windows the Triton wheel is the only source, so it must be asked for
        # alongside fla rather than assumed to already be there.
        assert any("triton" in part for part in install[5:])

    def test_an_install_failure_degrades_to_the_reference_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            raise DeciderServerError("uv pip install failed")

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        manager = _manager(tmp_path)
        manager._fla_kernel_probe = lambda: (False, "ModuleNotFoundError: triton")
        steps: list[str] = []
        manager.on_step = steps.append
        manager._ensure_gdn_kernels()

        assert manager._kernels_usable is False
        assert any("could not be installed" in step for step in steps)

    def test_a_still_unresolvable_kernel_after_install_is_a_warning_not_an_error(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )
        monkeypatch.setattr("ytdl_core.decider_server.os.name", "nt")
        monkeypatch.setattr(
            "ytdl_core.decider_server.subprocess.run",
            lambda command, **k: type("R", (), {"returncode": 0, "stdout": "2.8.0\n"})(),
        )
        monkeypatch.setattr(
            "ytdl_core.decider_server.run_command", lambda command, cwd=None, env=None, error=None, timeout=None: ""
        )
        manager = _manager(tmp_path)
        manager._fla_kernel_probe = lambda: (False, "still broken")
        steps: list[str] = []
        manager.on_step = steps.append
        manager._ensure_gdn_kernels()

        # The run still completes; it is just slower.
        assert manager._kernels_usable is False
        assert any("still do not resolve" in step for step in steps)

    def test_explicit_insist_and_decline_skip_the_probe(self, tmp_path):
        insisting = _manager(tmp_path, insist_kernels=True)
        insisting._fla_kernel_probe = (lambda: pytest.fail("must not probe when told what to do"))
        insisting._ensure_gdn_kernels()
        assert insisting._kernels_usable is True

        declining = _manager(tmp_path, insist_kernels=False)
        declining._fla_kernel_probe = (lambda: pytest.fail("must not probe when told what to do"))
        declining._ensure_gdn_kernels()
        assert declining._kernels_usable is False


class TestProcess:
    def test_the_checkpoint_reaches_the_server_through_the_environment(self, monkeypatch, tmp_path):
        # Unpinned, so no snapshot download is attempted and this test is only
        # about how the checkpoint is handed over.
        manager = _manager(tmp_path, run="Mapika/decider-12b")
        (tmp_path / "decider").mkdir(parents=True, exist_ok=True)
        captured: dict = {}

        class Fake:
            pid = 1234
            _handle = None

            def poll(self):
                return None

        def fake_popen(command, cwd=None, env=None, **kwargs):
            captured.update({"command": command, "env": env})
            return Fake()

        monkeypatch.setattr("ytdl_core.decider_server.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "ytdl_core.decider_server.create_kill_on_close_job", lambda process: None
        )
        monkeypatch.setattr("ytdl_core.decider_server.spawn_flags", lambda: (0, True))
        monkeypatch.setattr("ytdl_core.decider_server.terminate_process_tree", lambda *a: None)

        manager._start_process()
        try:
            # The checkpoint is chosen by the server through its environment, the
            # same way decider's own serve.sh passes it.
            assert captured["env"]["DECIDER_MODEL"] == "Mapika/decider-12b"
            assert "decider.serve:app" in captured["command"]
            assert "uvicorn" in captured["command"]
        finally:
            manager.log_file.close()
            manager.log_file = None
            manager.process = None


class TestPinnedCheckpoint:
    """decider.serve has one model variable and no revision knob.

    HuggingFace does not understand kev's `repo_id@revision` spelling either --
    from_pretrained rejects it with "Repo id must use alphanumeric chars" -- so a
    pinned checkpoint has to be materialised locally before the server sees it.
    Without this, pinning is unavailable and the weights track main, which is the
    hole TODO A10 was written about for Kev.
    """

    def test_an_unpinned_name_is_handed_over_as_is(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "ytdl_core.decider_server.run_command",
            lambda command, **kwargs: pytest.fail("an unpinned name must not be downloaded"),
        )
        assert _manager(tmp_path, run="Mapika/decider-4b")._resolved_model() == "Mapika/decider-4b"

    def test_a_pinned_name_is_downloaded_into_a_local_directory(self, monkeypatch, tmp_path):
        seen: list[list[str]] = []

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            seen.append(command)
            target = Path(command[-1])
            target.mkdir(parents=True, exist_ok=True)
            return f"{target}\n"

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        steps: list[str] = []
        resolved = _manager(tmp_path, steps, run="Mapika/decider-4b@v2")._resolved_model()

        # [python, "-c", script, repo_id, revision, target]
        assert seen[0][-3] == "Mapika/decider-4b"
        assert seen[0][-2] == "v2"
        assert resolved.endswith("Mapika--decider-4b@v2")
        assert any("fetching Mapika/decider-4b at revision v2" in step for step in steps)

    def test_a_pinned_name_is_not_downloaded_twice(self, monkeypatch, tmp_path):
        calls: list[list[str]] = []

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            calls.append(command)
            target = Path(command[-1])
            target.mkdir(parents=True, exist_ok=True)
            (target / ".ytdl-revision").write_text("v2", encoding="utf-8")
            return f"{target}\n"

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        manager = _manager(tmp_path, run="Mapika/decider-4b@v2")
        manager._resolved_model()
        manager._resolved_model()
        assert len(calls) == 1

    def test_a_malformed_pin_is_refused(self, tmp_path):
        with pytest.raises(DeciderServerError, match="not a usable checkpoint name"):
            _manager(tmp_path, run="Mapika/decider-4b@")._split_run()

    def test_a_local_directory_needs_no_revision(self, tmp_path):
        assert _manager(tmp_path, run="C:/models/decider")._split_run() == ("C:/models/decider", None)


class TestProbePayload:
    def test_the_probe_is_sized_by_the_real_question_budget(self, tmp_path):
        # A small probe sits on the flat part of the cost curve and passes while
        # the questions the pipeline really sends are already past the knee. So
        # the probe has to be derived from the budget, using the same arithmetic
        # the classifier uses to trim candidates.
        budget = Config().DECISION_MAX_QUESTIONS
        manager = _manager(tmp_path, max_questions=budget)
        payload = manager._probe_payload()

        affordable = max(1, (budget - 1) // len(dq.DIMENSIONS))
        assert len(payload["questions"]) == JevClassifier._question_count(affordable)
        assert len(payload["questions"]) <= budget
        assert "model" not in payload
        assert payload["state"]["target"]["artist"] == "Probe Artist"

    def test_it_falls_back_to_the_config_when_no_budget_is_given(self, tmp_path):
        manager = _manager(tmp_path)
        assert manager._question_budget() == Config().DECISION_MAX_QUESTIONS

    def test_a_smaller_budget_produces_a_smaller_probe(self, tmp_path):
        big = _manager(tmp_path, max_questions=32)._probe_payload()
        small = _manager(tmp_path, max_questions=7)._probe_payload()
        assert len(small["questions"]) < len(big["questions"])


class TestCapabilities:
    def test_it_reports_the_checkpoint_the_server_named(self, monkeypatch, tmp_path):
        _running(monkeypatch, {"model": "decider-v11", "device": "cuda:0", "temperature": 1.3})
        steps: list[str] = []
        _manager(tmp_path, steps)._report_capabilities()
        assert any("decider-v11" in step for step in steps)

    def test_it_reports_the_per_type_temperatures_when_there_are_them(self, monkeypatch, tmp_path):
        # decider.serve fits a temperature per question type, and every threshold
        # in this app is set against those probabilities, so they are worth seeing.
        _running(
            monkeypatch,
            {
                "model": "decider-v2",
                "device": "cuda:0",
                "temperature_by_type": {"choice": 1.5, "noul": 0.05},
            },
        )
        steps: list[str] = []
        _manager(tmp_path, steps)._report_capabilities()
        assert any("choice 1.5" in step and "noul 0.05" in step for step in steps)

    def test_it_surfaces_rejected_requests(self, monkeypatch, tmp_path):
        _running(
            monkeypatch,
            {"model": "decider-v2", "device": "cuda:0", "rejected_too_large": 4},
        )
        steps: list[str] = []
        _manager(tmp_path, steps)._report_capabilities()
        assert any("--decision-questions" in step for step in steps)


class TestGraphCapabilities:
    """The engine's own numbers are not in /health, so they are read from /stats
    and reported: how many CUDA graphs were captured, whether the engine sealed,
    and whether the GDN Triton kernels are live. The alternative to reporting
    these is discovering each one from a thousand timeouts.
    """

    def _stats(self, monkeypatch, payload: dict) -> None:
        """Make /stats answer with a payload, /health and /v1/models via _running.

        The ordinary `_running` helper answers every GET the same, so routing by
        URL is what lets this and that coexist.
        """
        default = {"model": "decider-v2", "device": "cuda:0"}

        def fake_get(url, timeout=None):
            class Response:
                status_code = 200

                def json(self):
                    return payload if url.endswith("/stats") else default

            return Response()

        monkeypatch.setattr(requests, "get", fake_get)

    def test_reported_graph_count_covers_the_captured_grid(self, monkeypatch, tmp_path):
        _running(monkeypatch, {"model": "decider-v2", "device": "cuda:0"})
        self._stats(monkeypatch, {"graphs": 5, "sealed": True, "engine": {"eager_forwards": 0}})
        steps: list[str] = []
        _manager(tmp_path, steps)._report_capabilities()
        assert any("5 CUDA graphs (sealed)" in step for step in steps)

    def test_a_zero_graph_engine_warns_rather_than_passes_silently(self, monkeypatch, tmp_path):
        _running(monkeypatch, {"model": "decider-v2", "device": "cuda:0"})
        self._stats(monkeypatch, {"graphs": 0, "sealed": False})
        steps: list[str] = []
        _manager(tmp_path, steps)._report_capabilities()
        assert any("WARNING" in step and "no CUDA graphs" in step for step in steps)

    def test_a_declined_kernel_is_reported_the_same_way_kevs_is(self, monkeypatch, tmp_path):
        self._stats(monkeypatch, {"graphs": 5, "sealed": True})
        steps: list[str] = []
        manager = _manager(tmp_path, steps, insist_kernels=False)
        # The flag records intent; _ensure_gdn_kernels is what materialises the
        # state the report tells the truth from.
        manager._ensure_gdn_kernels()
        manager._report_capabilities()
        assert any("DECLINED" in step for step in steps)
        assert any("reference path" in step for step in steps)

    def test_missing_stats_leaves_kernel_state_unreported(self, monkeypatch, tmp_path):
        # A decline that /stats knows nothing about is still reported: the engine
        # count is what /stats owns, the kernel decision is ours.
        self._stats(monkeypatch, {})
        steps: list[str] = []
        manager = _manager(tmp_path, steps, insist_kernels=False)
        manager._ensure_gdn_kernels()
        manager._report_capabilities()

    def test_no_stats_endpoint_degrades_quietly(self, monkeypatch, tmp_path):
        # An older decider.serve without /stats must not block the report.
        _running(monkeypatch, {"model": "decider-v2", "device": "cuda:0"})
        real_get = requests.get

        def fail_get(url, timeout=None):
            if url.endswith("/stats"):
                raise requests.ConnectionError("no such route")
            return real_get(url, timeout=timeout)

        monkeypatch.setattr(requests, "get", fail_get)
        steps: list[str] = []
        _manager(tmp_path, steps)._report_capabilities()
        assert any("decider-v2" in step for step in steps)


class TestLatencyReport:
    def _post(self, monkeypatch, *, status=200, raises=None, elapsed=None):
        class Response:
            status_code = status
            headers: dict = {}

            def json(self):
                return {"answers": {}}

        def fake_post(url, json=None, timeout=None):
            if elapsed:
                import time

                time.sleep(elapsed)
            if raises is not None:
                raise raises
            return Response()

        monkeypatch.setattr(requests, "post", fake_post)

    def test_a_slow_probe_is_reported_with_its_remedy(self, monkeypatch, tmp_path):
        _running(monkeypatch)
        self._post(monkeypatch, elapsed=0.05)
        steps: list[str] = []
        _manager(tmp_path, steps)._report_latency(budget_seconds=0.001)
        assert any("SLOW" in step and "--decision-questions" in step for step in steps)

    def test_a_timed_out_probe_is_the_answer_not_a_broken_probe(self, monkeypatch, tmp_path):
        _running(monkeypatch)
        self._post(monkeypatch, raises=requests.Timeout())
        steps: list[str] = []
        _manager(tmp_path, steps)._report_latency(budget_seconds=1.0)
        assert any("TOO SLOW" in step for step in steps)
        # And it must not claim the model is fine.
        assert not any("within budget" in step for step in steps)

    def test_a_rejected_probe_does_not_report_a_latency(self, monkeypatch, tmp_path):
        # The server answered without evaluating, so the timing measured nothing.
        _running(monkeypatch)
        self._post(monkeypatch, status=422)
        steps: list[str] = []
        _manager(tmp_path, steps)._report_latency(budget_seconds=60.0)
        assert any("measured no inference" in step for step in steps)
        assert not any("within budget" in step for step in steps)

    def test_a_fast_probe_is_reported_as_within_budget(self, monkeypatch, tmp_path):
        _running(monkeypatch)
        self._post(monkeypatch)
        steps: list[str] = []
        _manager(tmp_path, steps)._report_latency(budget_seconds=60.0)
        assert any("within budget" in step for step in steps)


class TestPackage:
    def test_the_pin_carries_the_serve_extra(self):
        # The base package is the library and its torch stack. Without [serve]
        # there is no uvicorn, so `python -m uvicorn decider.serve:app` dies --
        # after the whole of torch has been downloaded.
        assert "[serve]" in DEFAULT_PACKAGE
        assert "==" in DEFAULT_PACKAGE

    def test_the_install_is_verified_before_anything_is_started(self, monkeypatch, tmp_path):
        # Catching a broken environment takes a second here and several minutes
        # later, once torch is on disk and the process has already been spawned.
        python = tmp_path / "decider" / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        (tmp_path / "decider" / "ytdl-decider-package.txt").write_text("stale", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            calls.append(command)
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        _manager(tmp_path)._ensure_environment()

        assert any(c[-1] == "import decider.serve, uvicorn" for c in calls)

    def test_a_broken_install_names_the_missing_extra(self, monkeypatch, tmp_path):
        python = tmp_path / "decider" / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        (tmp_path / "decider" / "ytdl-decider-package.txt").write_text("stale", encoding="utf-8")

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            if command[-1] == "import decider.serve, uvicorn":
                raise DeciderServerError("python.exe failed: No module named uvicorn")
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        with pytest.raises(DeciderServerError, match=r"\[serve\] extra"):
            _manager(tmp_path)._ensure_environment()

    def test_the_marker_is_not_written_when_verification_fails(self, monkeypatch, tmp_path):
        # Otherwise the next run would see a matching marker, skip the install,
        # and fail the same way forever.
        python = tmp_path / "decider" / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        (tmp_path / "decider" / "ytdl-decider-package.txt").write_text("stale", encoding="utf-8")

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            if command[-1] == "import decider.serve, uvicorn":
                raise DeciderServerError("No module named uvicorn")
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        with pytest.raises(DeciderServerError):
            _manager(tmp_path)._ensure_environment()
        assert (tmp_path / "decider" / "ytdl-decider-package.txt").read_text() == "stale"


class TestLogTail:
    def test_it_quotes_the_reason_rather_than_only_the_path(self, tmp_path):
        log = tmp_path / "decider" / "ytdl-decider.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "some earlier attempt\npython.exe: No module named uvicorn\n", encoding="utf-8"
        )
        assert "No module named uvicorn" in _manager(tmp_path)._log_tail()

    def test_it_ignores_output_from_an_earlier_attempt(self, tmp_path):
        # The log is append-only. Quoting the previous run's failure as the reason
        # for this one is worse than quoting nothing.
        log = tmp_path / "decider" / "ytdl-decider.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("old failure: CUDA out of memory\n", encoding="utf-8")

        manager = _manager(tmp_path)
        manager._log_offset = len("old failure: CUDA out of memory\n")
        log.write_text(
            "old failure: CUDA out of memory\nnew failure: bad revision\n", encoding="utf-8"
        )
        tail = manager._log_tail()
        assert "bad revision" in tail
        assert "CUDA out of memory" not in tail

    def test_an_empty_log_says_so_rather_than_raising(self, tmp_path):
        assert "no output" in _manager(tmp_path)._log_tail()


class TestGpuTorch:
    """A plain `pip install torch` resolves to the CPU wheel and nothing complains.

    The result is the worst shape of failure to debug: the run works, the verdicts
    are right, and a 1554-song batch simply takes all day in RAM.
    """

    def _prepared(self, tmp_path):
        python = tmp_path / "decider" / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        (tmp_path / "decider" / "ytdl-decider-package.txt").write_text("stale", encoding="utf-8")

    def test_a_cpu_torch_is_replaced_when_a_gpu_is_present(self, monkeypatch, tmp_path):
        self._prepared(tmp_path)
        installs: list[list[str]] = []
        state = {"cuda": False}

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            if command[-1].startswith("import torch"):
                # A CPU build until the CUDA one has actually been installed.
                return f"2.11.0+{'cu128' if state['cuda'] else 'cpu'}\n{state['cuda']}\n"
            if command[:2] == ["uv", "pip"]:
                installs.append(command)
                if command[-1] == "torch":
                    state["cuda"] = True
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )
        steps: list[str] = []
        _manager(tmp_path, steps)._ensure_environment()

        torch_install = next(c for c in installs if c[-1] == "torch")
        assert "cu128" in torch_install
        # Without --reinstall uv treats this as already satisfied and does nothing.
        assert "--reinstall" in torch_install
        assert any("cannot use it" in step for step in steps)
        assert any("now sees the GPU" in step for step in steps)

    def test_a_cuda_torch_is_left_alone(self, monkeypatch, tmp_path):
        self._prepared(tmp_path)
        installs: list[list[str]] = []

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            if command[-1].startswith("import torch"):
                return "2.11.0+cu128\nTrue\n"
            if command[:2] == ["uv", "pip"]:
                installs.append(command)
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )
        _manager(tmp_path)._ensure_environment()

        assert [c for c in installs if c[-1] == "torch"] == []

    def test_no_gpu_means_no_cuda_install(self, monkeypatch, tmp_path):
        # CPU is then the honest answer, not a silent downgrade.
        self._prepared(tmp_path)
        installs: list[list[str]] = []

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            if command[-1].startswith("import torch"):
                return "2.14.1+cpu\nFalse\n"
            if command[:2] == ["uv", "pip"]:
                installs.append(command)
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        _manager(tmp_path)._ensure_environment()

        assert [c for c in installs if c[-1] == "torch"] == []

    def test_an_explicit_device_outranks_our_inference(self, monkeypatch, tmp_path):
        self._prepared(tmp_path)
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            raise AssertionError("must not touch torch when a device was requested")

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        with pytest.raises(AssertionError):
            _manager(tmp_path, device="cpu")._ensure_environment()

    def test_a_still_cpu_torch_warns_but_does_not_fail(self, monkeypatch, tmp_path):
        # The batch still completes correctly, just slowly. Taking that option away
        # would be worse than saying so.
        self._prepared(tmp_path)

        def fake_run(command, cwd=None, env=None, error=None, timeout=None):
            if command[-1].startswith("import torch"):
                return "2.14.1+cpu\nFalse\n"
            return ""

        monkeypatch.setattr("ytdl_core.decider_server.run_command", fake_run)
        monkeypatch.setattr(
            "ytdl_core.decider_server.nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5060 Ti"
        )
        steps: list[str] = []
        _manager(tmp_path, steps)._ensure_environment()
        assert any("WARNING" in step and "in RAM" in step for step in steps)


class TestReadiness:
    def test_a_server_that_exits_during_startup_says_why(self, monkeypatch, tmp_path):
        def fake_get(url, timeout=None):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", fake_get)

        class Dead:
            pid = 1

            def poll(self):
                return 1

        monkeypatch.setattr("ytdl_core.decider_server.subprocess.Popen", lambda *a, **k: Dead())
        monkeypatch.setattr(
            "ytdl_core.decider_server.create_kill_on_close_job", lambda process: None
        )
        monkeypatch.setattr("ytdl_core.decider_server.spawn_flags", lambda: (0, True))
        monkeypatch.setattr("ytdl_core.decider_server.terminate_process_tree", lambda *a: None)

        manager = _manager(tmp_path, url="http://127.0.0.1:8010")
        manager.process = Dead()
        with pytest.raises(DeciderServerError, match="exited during startup"):
            manager._wait_until_ready()

    def test_the_health_endpoint_is_asked_first(self, monkeypatch, tmp_path):
        asked = _running(monkeypatch)
        assert _manager(tmp_path)._server_info() is not None
        assert asked[0].endswith("/health")
