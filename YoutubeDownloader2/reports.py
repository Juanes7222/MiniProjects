"""
Report export (JSON / CSV / M3U8) and --update-json rewriter.
"""

from __future__ import annotations

import csv, json, shutil, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def export_report(results: List[dict], output_dir: Path, formats: List[str]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for fmt in formats:
        if fmt == "json":   _write_json(results, output_dir, ts)
        elif fmt == "csv":  _write_csv(results, output_dir, ts)
        elif fmt == "m3u":  _write_m3u(results, output_dir, ts)


def update_json_file(path: Path, results: List[dict]) -> None:
    annotated: dict = {}
    for r in results:
        artist, song = r.get("artist", ""), r.get("song", "")
        annotated.setdefault(artist, {})
        entry = {"status": r.get("status", "unknown")}
        if r.get("status") == "downloaded":
            entry["file"] = r.get("file_path", "")
        elif r.get("status") == "failed":
            entry["reason"] = r.get("reason", "Unknown error")
        annotated[artist][song] = entry
    _atomic_write(path, annotated)


_CSV_FIELDS = [
    "artist", "song", "status", "source", "url", "matched_title",
    "fuzzy_score", "duration_seconds", "file_path", "file_size_bytes",
    "md5", "musicbrainz_enriched", "album", "year", "genre",
    "composite_score", "fingerprint_verified", "fingerprint_confidence",
    "fingerprint_matched_title", "silence_ratio", "duration_verified",
]


def _write_json(results, output_dir, ts):
    dl   = sum(1 for r in results if r.get("status") == "downloaded")
    fail = sum(1 for r in results if r.get("status") == "failed")
    skip = sum(1 for r in results if r.get("status") == "skipped")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"total": len(results), "downloaded": dl, "failed": fail, "skipped": skip},
        "tracks": results,
    }
    dest = output_dir / f"download_report_{ts}.json"
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def _write_csv(results, output_dir, ts):
    dest = output_dir / f"download_report_{ts}.csv"
    with dest.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_m3u(results, output_dir, ts):
    dest = output_dir / f"playlist_{ts}.m3u8"
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("#EXTM3U\n")
        for r in results:
            if r.get("status") != "downloaded" or not r.get("file_path"):
                continue
            fp = r["file_path"]
            duration = r.get("duration_seconds", -1)
            try:
                rel = "./" + str(Path(fp).relative_to(output_dir)).replace("\\", "/")
            except ValueError:
                rel = fp
            fh.write(f'#EXTINF:{duration},{r.get("artist","")} - {r.get("song","")}\n')
            fh.write(f"{rel}\n")


def _atomic_write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        shutil.move(tmp, str(path))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise