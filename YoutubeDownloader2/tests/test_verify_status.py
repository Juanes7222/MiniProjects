"""A verify pass must not invalidate what the state file already established.

``--verify`` rewrites ``status`` from ``downloaded`` to ``verified``. Two places
asked "is this file present and good?" by testing for the literal string
``"downloaded"``, and both broke the next time anything read the file:

* ``ChannelTrust.from_state`` skipped every entry, so the learned channel model
  went from 10 channels to 0 -- and the channel provenance a batch had spent
  whole runs accumulating became invisible;
* ``--skip-existing`` stopped skipping, so the following batch re-downloaded the
  whole library.

Measured on a real library: ``--verify`` took skip-existing from 88-of-91 songs
skipped to 0-of-91, and emptied the trust model. Both now go through
``has_local_file``.
"""
from __future__ import annotations


from ytdl_core.channels import ChannelTrust
from ytdl_core.config import Config
from ytdl_core.state import PRESENT_STATUSES, has_local_file


def _entry(status: str, **extra):
    return {"status": status, "channel": "Some Channel", "url": "https://x/y", **extra}


# --- the rule ---------------------------------------------------------------


def test_both_spellings_count_as_present():
    assert has_local_file(_entry("downloaded")) is True
    assert has_local_file(_entry("verified")) is True


def test_a_failure_is_not_present():
    for status in ("failed", "skipped", "", None, "pending"):
        assert has_local_file(_entry(status)) is False, status


def test_a_malformed_entry_is_not_present():
    for value in (None, "downloaded", 42, []):
        assert has_local_file(value) is False, value


def test_the_present_set_is_explicit_and_small():
    """A catch-all would let a failed entry back in."""
    assert PRESENT_STATUSES == frozenset({"downloaded", "verified"})


# --- the trust model --------------------------------------------------------


def test_channel_trust_survives_a_verify_pass():
    downloaded = {"downloads": {"A::S": _entry("downloaded")}}
    verified = {"downloads": {"A::S": _entry("verified")}}

    before = ChannelTrust.from_state(downloaded, Config())
    after = ChannelTrust.from_state(verified, Config())

    assert len(before) == 1, "control: a downloaded entry builds the model"
    assert len(after) == 1, "a verified entry must build it too"
    assert before.top_channels() == after.top_channels()


def test_channel_trust_still_ignores_failures():
    state = {"downloads": {"A::S": _entry("failed"), "A::T": _entry("skipped")}}
    assert len(ChannelTrust.from_state(state, Config())) == 0


def test_a_verified_entry_earns_the_verified_multiplier():
    """The point of verifying at all: verified downloads count triple."""
    plain = {"downloads": {"A::S": _entry("downloaded", fingerprint_verified=False)}}
    verified = {"downloads": {"A::S": _entry("verified", fingerprint_verified=True)}}

    plain_weight = ChannelTrust.from_state(plain, Config()).top_channels()[0][1]
    verified_weight = ChannelTrust.from_state(verified, Config()).top_channels()[0][1]

    assert verified_weight == plain_weight * Config().TRUST_VERIFIED_MULTIPLIER


# --- skip-existing ----------------------------------------------------------


def test_skip_existing_survives_a_verify_pass(tmp_path):
    """The end-to-end consequence: a verified file is still skipped."""
    import threading

    from ytdl_core.core import MusicDownloader
    from ytdl_core.events import DownloaderEvents
    from ytdl_core.utils import compute_file_hash

    audio = tmp_path / "Artist" / "Song.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"\x00" * 1000)
    digest = compute_file_hash(audio)

    class Recorder(DownloaderEvents):
        def __init__(self):
            self.skipped: list[tuple] = []
            self.mismatched: list[str] = []

        def on_skip_existing(self, artist, song, path, md5_ok):
            self.skipped.append((artist, song, md5_ok))

        def on_md5_mismatch(self, artist, song):
            self.mismatched.append(f"{artist} -- {song}")

    def run(status: str):
        events = Recorder()
        dl = MusicDownloader(events=events, skip_fingerprint=True)
        state = {
            "downloads": {
                "Artist::Song": {
                    "status": status,
                    "md5": digest,
                    "file_path": str(audio),
                }
            }
        }
        result = dl._process_song(
            "Artist", "Song", tmp_path, "mp3", "192", True,
            state, threading.Lock(), threading.Event(),
            set(), threading.Lock(), {"Artist": 1},
        )
        return events, result

    for status in ("downloaded", "verified"):
        events, result = run(status)
        assert result.status == "skipped", f"status={status} did not skip"
        assert events.skipped == [("Artist", "Song", True)], f"status={status}: {events.skipped}"
        assert not events.mismatched, f"status={status}: {events.mismatched}"

    # And a file whose contents changed still re-downloads rather than skipping.
    audio.write_bytes(b"\xff" * 1000)
    events, result = run("verified")
    assert result.status != "skipped", "a modified file must not be skipped"
    assert events.mismatched == ["Artist -- Song"]