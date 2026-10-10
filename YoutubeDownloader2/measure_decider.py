"""Measure what decider.serve does to VRAM, per configuration.

Run: python measure_decider.py
Not part of the test suite; a bench harness for the bucket decision.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path.home() / ".cache" / "ytdl" / "decider"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
MODEL = str(ROOT / "models" / "Mapika--decider-4b@v2")
PORT = 8021

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ytdl_core import decision_questions as dq  # noqa: E402


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def get(path: str, timeout: float = 10.0):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode())


def post(path: str, payload: dict, timeout: float = 300.0):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return body, time.monotonic() - started


def probe_payload(budget: int = 32) -> dict:
    affordable = max(1, (budget - 1) // len(dq.DIMENSIONS))
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
        for i in range(affordable)
    ]
    state, _states, questions = dq.build_state_and_questions(
        "Probe Artist", "Probe Song", candidates, None, dq.TYPESAFE_DIALECT
    )
    return {"state": state, "questions": questions}


def run(label: str, extra_env: dict[str, str]) -> None:
    env = os.environ.copy()
    env["DECIDER_MODEL"] = MODEL
    env.pop("DECIDER_DEVICE", None)
    env.update(extra_env)

    baseline = gpu_used_mib()
    log = ROOT / f"measure-{label}.log"
    process = subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "decider.serve:app", "--host", "127.0.0.1",
         "--port", str(PORT)],
        cwd=str(ROOT),
        stdout=log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        started = time.monotonic()
        deadline = started + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                print(f"[{label}] server exited rc={process.returncode}")
                print(log.read_text(encoding="utf-8", errors="replace")[-1500:])
                return
            try:
                get("/health", timeout=3)
                break
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(3)
        else:
            print(f"[{label}] never became ready")
            return

        startup = time.monotonic() - started
        time.sleep(5)
        after_start = gpu_used_mib()
        health = get("/health")
        stats = get("/stats")
        engine = (stats.get("engine") or {}) if isinstance(stats, dict) else {}

        payload = probe_payload()
        body, elapsed = post("/v1/systemone", payload)
        answers = body.get("answers") or {}
        during = gpu_used_mib()

        print(f"[{label}]")
        print(f"  startup        {startup:.0f}s")
        print(f"  device         {health.get('device')}")
        print(f"  graph_captures {engine.get('graph_captures')}")
        print(f"  eager_forwards {engine.get('eager_forwards')}")
        print(f"  VRAM idle      {after_start - baseline:+5d} MiB  (total {after_start})")
        print(f"  VRAM peak      {during - baseline:+5d} MiB  (total {during})")
        print(f"  questions      {len(payload['questions'])}  answers {len(answers)}")
        print(f"  latency        {elapsed:.2f}s")
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        time.sleep(3)


CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("default", {}),
    ("b1-short", {"DECIDER_B_BUCKETS": "1", "DECIDER_T_BUCKETS": "64,128,256,512,1024"}),
    ("b1-tiny", {"DECIDER_B_BUCKETS": "1", "DECIDER_T_BUCKETS": "64,128,256,512"}),
    ("warmup0", {"DECIDER_WARMUP": "0"}),
]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"baseline VRAM (desktop only): {gpu_used_mib()} MiB")
    for label, extra in CONFIGS:
        if only and label != only:
            continue
        run(label, extra)
