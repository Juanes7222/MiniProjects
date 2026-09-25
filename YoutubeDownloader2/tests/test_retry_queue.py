from __future__ import annotations

import json

from ytdl_core.retry_queue import (
    load_retry_queue,
    queue_path,
    read_retry_queue,
    record_retry_queue,
)
from ytdl_core.result import DownloadResult


def test_failed_results_are_saved_for_retry(tmp_path) -> None:
    results = [
        DownloadResult(artist="Artist", song="Song", reason="Network error"),
        DownloadResult(artist="Other", song="Track", status="downloaded"),
    ]

    stats = record_retry_queue(tmp_path, results)

    assert stats == {"pending": 1, "added": 1, "requeued": 0, "removed": 0}
    assert load_retry_queue(tmp_path) == {"Artist": ["Song"]}
    document = json.loads(queue_path(tmp_path).read_text(encoding="utf-8"))
    assert document["items"][0]["attempts"] == 1


def test_repeated_failures_increment_attempts(tmp_path) -> None:
    failed = [DownloadResult(artist="Artist", song="Song", reason="Timeout")]

    record_retry_queue(tmp_path, failed)
    record_retry_queue(tmp_path, failed)

    document = json.loads(queue_path(tmp_path).read_text(encoding="utf-8"))
    assert document["items"][0]["attempts"] == 2


def test_success_removes_item_from_queue(tmp_path) -> None:
    record_retry_queue(
        tmp_path,
        [DownloadResult(artist="Artist", song="Song", reason="Timeout")],
    )

    stats = record_retry_queue(
        tmp_path,
        [DownloadResult(artist="Artist", song="Song", status="downloaded")],
    )

    assert stats == {"pending": 0, "added": 0, "requeued": 0, "removed": 1}
    assert load_retry_queue(tmp_path) == {}


def test_malformed_queue_is_quarantined(tmp_path) -> None:
    queue_path(tmp_path).write_text("not-json", encoding="utf-8")

    songs, warning = read_retry_queue(tmp_path)

    assert songs == {}
    assert warning
    assert list(tmp_path.glob("retry_queue.json.corrupt-*"))


def test_load_retry_queue_does_not_modify_files(tmp_path) -> None:
    queue_path(tmp_path).write_text("not-json", encoding="utf-8")

    assert load_retry_queue(tmp_path) == {}
    assert queue_path(tmp_path).exists()


def test_invalid_attempts_value_is_tolerated(tmp_path) -> None:
    queue_path(tmp_path).write_text(
        '{"items": [{"artist": "Artist", "song": "Song", "attempts": "many"}]}',
        encoding="utf-8",
    )

    assert load_retry_queue(tmp_path) == {"Artist": ["Song"]}


def test_separator_in_names_does_not_collide(tmp_path) -> None:
    results = [
        DownloadResult(artist="A::B", song="C", reason="First"),
        DownloadResult(artist="A", song="B::C", reason="Second"),
    ]

    record_retry_queue(tmp_path, results)

    assert load_retry_queue(tmp_path) == {
        "A::B": ["C"],
        "A": ["B::C"],
    }
