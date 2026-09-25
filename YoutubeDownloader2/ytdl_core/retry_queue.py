from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .json_io import write_json_atomic
from .result import DownloadResult

RETRY_QUEUE_FILENAME = "retry_queue.json"


class RetryQueueError(RuntimeError):
    pass


def queue_path(output_dir: Path) -> Path:
    return Path(output_dir) / RETRY_QUEUE_FILENAME


def read_retry_queue(output_dir: Path) -> tuple[dict[str, list[str]], str | None]:
    path = queue_path(output_dir)
    try:
        with path.open("r", encoding="utf-8") as file:
            document: Any = json.load(file)
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as error:
        warning = _quarantine(path, error)
        return {}, warning

    if not isinstance(document, dict):
        warning = _quarantine(path, ValueError("root must be an object"))
        return {}, warning

    songs: dict[str, list[str]] = {}
    for item in _normalized_items(document):
        songs.setdefault(item["artist"], [])
        if item["song"] not in songs[item["artist"]]:
            songs[item["artist"]].append(item["song"])
    return songs, None


def load_retry_queue(output_dir: Path) -> dict[str, list[str]]:
    path = queue_path(output_dir)
    try:
        with path.open("r", encoding="utf-8") as file:
            document: Any = json.load(file)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    songs: dict[str, list[str]] = {}
    for item in _normalized_items(document):
        songs.setdefault(item["artist"], [])
        if item["song"] not in songs[item["artist"]]:
            songs[item["artist"]].append(item["song"])
    return songs


def load_retry_details(output_dir: Path) -> list[dict[str, Any]]:
    path = queue_path(output_dir)
    try:
        with path.open("r", encoding="utf-8") as file:
            document: Any = json.load(file)
    except (FileNotFoundError, OSError, ValueError):
        return []
    if not isinstance(document, dict):
        return []
    return _normalized_items(document)


def record_retry_queue(
    output_dir: Path,
    results: list[DownloadResult],
) -> dict[str, int | str]:
    path = queue_path(output_dir)
    document, warning = _read_document(path)
    items = _normalized_items(document)

    added = 0
    requeued = 0
    removed = 0
    for result in results:
        matching_index = next(
            (
                index
                for index, item in enumerate(items)
                if item["artist"] == result.artist and item["song"] == result.song
            ),
            None,
        )
        if result.status == "failed":
            existing = dict(items[matching_index]) if matching_index is not None else {}
            attempts = int(existing.get("attempts", 0)) + 1
            item = {
                "artist": result.artist,
                "song": result.song,
                "reason": result.reason or "Unknown error",
                "attempts": attempts,
                "last_attempt": datetime.now(timezone.utc).isoformat(),
            }
            if matching_index is None:
                items.append(item)
                added += 1
            else:
                items[matching_index] = item
                requeued += 1
        elif result.status in ("downloaded", "verified") and matching_index is not None:
            items.pop(matching_index)
            removed += 1

    output_document = dict(document)
    output_document.update(
        {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "items": items,
        }
    )
    write_json_atomic(path, output_document)
    stats: dict[str, int | str] = {
        "pending": len(items),
        "added": added,
        "requeued": requeued,
        "removed": removed,
    }
    if warning:
        stats["warning"] = warning
    return stats


def _read_document(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        with path.open("r", encoding="utf-8") as file:
            document: Any = json.load(file)
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as error:
        return {}, _quarantine(path, error)
    if not isinstance(document, dict):
        return {}, _quarantine(path, ValueError("root must be an object"))
    return document, None


def _normalized_items(document: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = document.get("items", [])
    if isinstance(raw_items, dict):
        values = list(raw_items.values())
    elif isinstance(raw_items, list):
        values = raw_items
    else:
        return []

    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_item in values:
        if not isinstance(raw_item, dict):
            continue
        artist = str(raw_item.get("artist") or "").strip()
        song = str(raw_item.get("song") or "").strip()
        identity = (artist, song)
        if not artist or not song or identity in seen:
            continue
        seen.add(identity)
        try:
            attempts = int(raw_item.get("attempts") or 1)
        except (TypeError, ValueError):
            attempts = 1
        items.append(
            {
                "artist": artist,
                "song": song,
                "reason": str(raw_item.get("reason") or "Unknown error"),
                "attempts": attempts,
                "last_attempt": str(raw_item.get("last_attempt") or ""),
            }
        )
    return items


def _quarantine(path: Path, error: Exception) -> str:
    quarantine_path = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
    try:
        path.replace(quarantine_path)
    except OSError:
        raise RetryQueueError(f"Could not quarantine invalid retry queue: {error}") from error
    return f"Invalid retry queue moved to {quarantine_path}"
