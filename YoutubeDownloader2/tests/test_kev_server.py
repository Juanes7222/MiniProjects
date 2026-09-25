from __future__ import annotations

from ytdl_core.kev_server import KevServerManager


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

    assert manager.start() == "http://127.0.0.1:8009"
    assert steps == [
        "repository",
        "environment",
        "cuda",
        "Kev: starting jaredpalmer/kev-4b on CUDA through kev.serve",
        "process",
        "ready",
    ]


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
