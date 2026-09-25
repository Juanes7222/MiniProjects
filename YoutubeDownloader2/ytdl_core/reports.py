from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .json_io import write_json_atomic

_CSV_FIELDS = [
    "artist",
    "song",
    "status",
    "source",
    "url",
    "matched_title",
    "fuzzy_score",
    "duration_seconds",
    "file_path",
    "file_size_bytes",
    "md5",
    "musicbrainz_enriched",
    "album",
    "year",
    "genre",
    "composite_score",
    "heuristic_score",
    "decision_provider",
    "decision_probability",
    "decision_runs",
    "selection_method",
    "fingerprint_verified",
    "fingerprint_confidence",
    "fingerprint_matched_title",
    "silence_ratio",
    "duration_verified",
]


def export_report(
    results: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    writers = {
        "json": _write_json,
        "csv": _write_csv,
        "m3u": _write_m3u,
    }
    for output_format in formats:
        writer = writers.get(output_format)
        if writer:
            writer(results, output_dir, timestamp)


def update_json_file(path: Path, results: list[dict[str, Any]]) -> None:
    annotated: dict[str, dict[str, dict[str, str]]] = {}
    for result in results:
        artist = result.get("artist", "")
        song = result.get("song", "")
        entry: dict[str, str] = {"status": result.get("status", "unknown")}
        if result.get("status") in ("downloaded", "verified"):
            entry["file"] = result.get("file_path", "")
        elif result.get("status") == "failed":
            entry["reason"] = result.get("reason", "Unknown error")
        annotated.setdefault(artist, {})[song] = entry
    write_json_atomic(path, annotated)


def _write_json(
    results: list[dict[str, Any]],
    output_dir: Path,
    timestamp: str,
) -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(results),
            "downloaded": sum(
                result.get("status") in ("downloaded", "verified") for result in results
            ),
            "failed": sum(result.get("status") == "failed" for result in results),
            "skipped": sum(result.get("status") == "skipped" for result in results),
        },
        "tracks": results,
    }
    write_json_atomic(output_dir / f"download_report_{timestamp}.json", report)


def _write_csv(
    results: list[dict[str, Any]],
    output_dir: Path,
    timestamp: str,
) -> None:
    destination = output_dir / f"download_report_{timestamp}.csv"
    with destination.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_m3u(
    results: list[dict[str, Any]],
    output_dir: Path,
    timestamp: str,
) -> None:
    destination = output_dir / f"playlist_{timestamp}.m3u8"
    with destination.open("w", encoding="utf-8") as file:
        file.write("#EXTM3U\n")
        for result in results:
            if result.get("status") not in ("downloaded", "verified"):
                continue
            file_path = result.get("file_path")
            if not file_path:
                continue
            try:
                relative_path = "./" + str(Path(file_path).relative_to(output_dir)).replace(
                    "\\", "/"
                )
            except ValueError:
                relative_path = file_path
            file.write(
                f"#EXTINF:{result.get('duration_seconds', -1)},"
                f"{result.get('artist', '')} - {result.get('song', '')}\n"
            )
            file.write(f"{relative_path}\n")
