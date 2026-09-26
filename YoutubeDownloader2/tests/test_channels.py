from __future__ import annotations

from ytdl_core.channels import ChannelTrust
from ytdl_core.config import Config


def _state(entries: list[dict]) -> dict:
    return {"downloads": {entry["key"]: entry for entry in entries}}


def _entry(key: str, channel: str, **extra) -> dict:
    return {
        "key": key,
        "status": "downloaded",
        "url": f"https://www.youtube.com/watch?v={key}",
        "channel": channel,
        **extra,
    }


class TestConstruction:
    def test_empty_state_yields_no_trust(self) -> None:
        trust = ChannelTrust.from_state({"downloads": {}})
        assert len(trust) == 0
        assert trust.bonus_for("Anything") == 0

    def test_missing_state_is_safe(self) -> None:
        assert len(ChannelTrust.from_state(None)) == 0
        assert len(ChannelTrust.from_state({})) == 0

    def test_malformed_downloads_is_safe(self) -> None:
        trust = ChannelTrust.from_state({"downloads": "not-a-dict"})
        assert len(trust) == 0

    def test_only_downloaded_entries_count(self) -> None:
        state = _state(
            [
                _entry("A::One", "Label Channel"),
                {"key": "A::Two", "status": "failed", "channel": "Label Channel"},
            ]
        )
        trust = ChannelTrust.from_state(state)
        assert trust.weight("Label Channel") == 1

    def test_entries_without_channel_are_skipped(self) -> None:
        state = _state([{"key": "A::One", "status": "downloaded", "url": "x"}])
        assert len(ChannelTrust.from_state(state)) == 0

    def test_verified_downloads_weigh_more(self) -> None:
        state = _state(
            [
                _entry("A::One", "Topic Channel", fingerprint_verified=True),
            ]
        )
        trust = ChannelTrust.from_state(state)
        assert trust.weight("Topic Channel") == Config().TRUST_VERIFIED_MULTIPLIER


class TestBonus:
    def test_unknown_channel_gets_nothing(self) -> None:
        trust = ChannelTrust.from_state(_state([_entry("A::One", "Known Channel")]))
        assert trust.bonus_for("Random Uploader") == 0

    def test_single_observation_is_not_trusted(self) -> None:
        """One mistaken download must not promote its uploader."""
        trust = ChannelTrust.from_state(_state([_entry("A::One", "Mario Alberto Rdz")]))
        assert trust.weight("Mario Alberto Rdz") == 1
        assert trust.bonus_for("Mario Alberto Rdz", "A") == 0
        assert trust.is_known("Mario Alberto Rdz") is True

    def test_two_observations_start_earning_trust(self) -> None:
        trust = ChannelTrust.from_state(
            _state([_entry("A::One", "Label Channel"), _entry("A::Two", "Label Channel")])
        )
        assert trust.bonus_for("Label Channel", "A") > 0

    def test_two_verified_observations_earn_full_trust(self) -> None:
        trust = ChannelTrust.from_state(
            _state(
                [
                    _entry("A::One", "Label Channel", fingerprint_verified=True),
                    _entry("A::Two", "Label Channel", fingerprint_verified=True),
                ]
            )
        )
        assert trust.bonus_for("Label Channel") == Config().TRUSTED_CHANNEL_BONUS

    def test_repeated_channel_outranks_single_use(self) -> None:
        entries = [_entry(f"A::S{i}", "Label Channel") for i in range(3)]
        entries.append(_entry("B::Other", "Random Uploader"))
        trust = ChannelTrust.from_state(_state(entries))
        assert trust.bonus_for("Label Channel", "A") > trust.bonus_for("Random Uploader", "B")

    def test_same_artist_channel_outranks_other_artist_channel(self) -> None:
        entries = [_entry("A::S1", "Shared Channel")]
        entries += [_entry(f"A::S{i}", "Shared Channel") for i in range(2, 5)]
        entries.append(_entry("Z::Other", "Other Artist Channel"))
        trust = ChannelTrust.from_state(_state(entries))
        assert trust.bonus_for("Shared Channel", "A") > trust.bonus_for("Other Artist Channel", "Z")

    def test_bonus_is_capped(self) -> None:
        entries = [_entry(f"A::S{i}", "Label Channel", fingerprint_verified=True) for i in range(50)]
        trust = ChannelTrust.from_state(_state(entries))
        assert trust.bonus_for("Label Channel", "A") == Config().TRUST_MAX_BONUS

    def test_matching_is_case_and_punctuation_insensitive(self) -> None:
        trust = ChannelTrust.from_state(
            _state([_entry("A::One", "Discos Fuentes Edimusica"), _entry("A::Two", "Discos Fuentes Edimusica")])
        )
        assert trust.bonus_for("discos fuentes edimusica", "A") > 0
        assert trust.bonus_for("DISCOS FUENTES EDIMUSICA!", "A") > 0

    def test_is_known(self) -> None:
        trust = ChannelTrust.from_state(_state([_entry("A::One", "Label Channel")]))
        assert trust.is_known("label channel") is True
        assert trust.is_known("nobody") is False


class TestChannelLookup:
    def test_only_channels_with_a_url_are_returned(self) -> None:
        entries = [
            _entry("A::One", "No URL Channel"),
            _entry("A::Two", "Url Channel", channel_url="https://youtube.com/channel/UC1"),
        ]
        trust = ChannelTrust.from_state(_state(entries))
        found = trust.channels_for_artist("A")
        assert [c["name"] for c in found] == ["Url Channel"]

    def test_non_youtube_channels_are_excluded(self) -> None:
        """Channel-scoped search is YouTube-only; SoundCloud handles 404."""
        entries = [
            _entry("A::One", "SC", channel_url="https://soundcloud.com/someone"),
            _entry("A::Two", "YT", channel_url="https://www.youtube.com/channel/UC1"),
        ]
        trust = ChannelTrust.from_state(_state(entries))
        assert [c["name"] for c in trust.channels_for_artist("A")] == ["YT"]

    def test_scoped_to_the_requested_artist(self) -> None:
        entries = [
            _entry("A::One", "A Channel", channel_url="https://youtube.com/channel/UC1"),
            _entry("B::One", "B Channel", channel_url="https://youtube.com/channel/UC2"),
        ]
        trust = ChannelTrust.from_state(_state(entries))
        assert [c["name"] for c in trust.channels_for_artist("A")] == ["A Channel"]

    def test_respects_the_limit(self) -> None:
        entries = [
            _entry(f"A::S{i}", f"Channel {i}", channel_url=f"https://youtube.com/channel/UC{i}")
            for i in range(5)
        ]
        trust = ChannelTrust.from_state(_state(entries))
        assert len(trust.channels_for_artist("A", limit=2)) == 2

    def test_unknown_artist_returns_nothing(self) -> None:
        trust = ChannelTrust.from_state(
            _state([_entry("A::One", "A Channel", channel_url="https://youtube.com/channel/UC1")])
        )
        assert trust.channels_for_artist("Nobody") == []


class TestScoringIntegration:
    def test_trusted_channel_lifts_the_candidate(self) -> None:
        from ytdl_core.scorer import score_youtube_result

        config = Config()
        base = {"title": "Song - Artist", "channel": "Label Channel", "duration": 200}
        other = {**base, "channel": "Random Uploader"}

        trust = ChannelTrust.from_state(
            _state([_entry("Artist::One", "Label Channel"), _entry("Artist::Two", "Label Channel")])
        )
        trusted_score, trusted_bd = score_youtube_result(base, "Artist", "Song", None, config, trust)
        plain_score, plain_bd = score_youtube_result(base, "Artist", "Song", None, config)

        assert trusted_score > plain_score
        assert trusted_bd["trusted_channel"] > 0
        assert "trusted_channel" not in plain_bd
        assert score_youtube_result(other, "Artist", "Song", None, config, trust)[0] == plain_score

    def test_untrusted_channel_scores_identically(self) -> None:
        from ytdl_core.scorer import score_youtube_result

        config = Config()
        entry = {"title": "Song - Artist", "channel": "One Off Uploader", "duration": 200}
        trust = ChannelTrust.from_state(
            _state([_entry("Artist::One", "One Off Uploader")])
        )
        assert score_youtube_result(entry, "Artist", "Song", None, config, trust)[0] == (
            score_youtube_result(entry, "Artist", "Song", None, config)[0]
        )
