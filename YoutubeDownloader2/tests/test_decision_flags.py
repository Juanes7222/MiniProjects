"""The decision-model flags, which were five provider-prefixed flags and are now three.

Most of what follows is about the merge rather than about Decider. The old names
are still accepted, so what has to be pinned is that they land on the *same*
destination as the new one -- a deprecation that quietly kept a separate value
would be worse than a hard break, because the flag would look like it worked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ytdl_core.cli.arg_parser import parse_args
from ytdl_core.config import Config
from ytdl_core.decider_server import DEFAULT_PACKAGE

SOURCE = ["--retry", "--output", "downloads"]


def _profile(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "profiles.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestMergedDecisionFlags:
    def test_threshold_and_runs_have_one_name_each(self):
        args = parse_args([*SOURCE, "--decision-threshold", "0.7", "--decision-runs", "4"])
        assert args.decision_threshold == 0.7
        assert args.decision_runs == 4

    def test_the_defaults_come_from_the_shared_config(self):
        args = parse_args(SOURCE)
        assert args.decision_threshold == Config.DECISION_DEFAULT_THRESHOLD
        assert args.decision_runs == Config.DECISION_DEFAULT_RUNS
        assert args.decision_timeout == Config.DECISION_TIMEOUT_SECONDS

    def test_timeout_is_named_for_the_decision_not_the_server(self):
        # --kev-timeout was always generic in behaviour: it applied to whichever
        # model was in use. Only the name was wrong.
        args = parse_args([*SOURCE, "--decision-timeout", "90"])
        assert args.decision_timeout == 90

    @pytest.mark.parametrize(
        ("old", "value", "dest"),
        [
            ("--jev-runs", "3", "decision_runs"),
            ("--kev-runs", "3", "decision_runs"),
            ("--jev-threshold", "0.8", "decision_threshold"),
            ("--kev-threshold", "0.8", "decision_threshold"),
            ("--kev-timeout", "120", "decision_timeout"),
        ],
    )
    def test_the_old_names_reach_the_same_destination(self, old, value, dest, capsys):
        args = parse_args([*SOURCE, old, value])
        assert getattr(args, dest) == type(getattr(parse_args(SOURCE), dest))(value)
        # And it says so, rather than letting the spelling rot unnoticed.
        assert "deprecated" in capsys.readouterr().err

    def test_the_old_names_are_not_left_behind_as_separate_attributes(self):
        args = parse_args([*SOURCE, "--kev-runs", "3"])
        assert not hasattr(args, "kev_runs")
        assert not hasattr(args, "jev_runs")

    @pytest.mark.parametrize(
        "argv",
        [
            ["--decision-runs", "0"],
            ["--decision-runs", "999"],
            ["--decision-threshold", "0"],
            ["--decision-threshold", "1.5"],
            ["--decision-timeout", "0"],
        ],
    )
    def test_the_merged_values_are_validated_under_their_new_names(self, argv):
        with pytest.raises(SystemExit):
            parse_args([*SOURCE, *argv])

    def test_the_old_names_are_validated_too(self):
        with pytest.raises(SystemExit):
            parse_args([*SOURCE, "--kev-runs", "0"])


class TestProviderSelection:
    def test_the_three_providers_are_mutually_exclusive(self):
        for pair in (("--jev", "--kev"), ("--kev", "--decider"), ("--jev", "--decider")):
            with pytest.raises(SystemExit):
                parse_args([*SOURCE, *pair])

    def test_decider_is_off_by_default(self):
        args = parse_args(SOURCE)
        assert args.decider is False
        assert args.jev is False
        assert args.kev is False

    def test_decider_has_its_own_port_and_directory(self):
        args = parse_args([*SOURCE, "--decider"])
        assert args.decider_url == "http://127.0.0.1:8010"
        assert args.decider_port == 8010
        assert args.decider_dir.parts[-1] == "decider"
        # It must not collide with Kev's, or both would try to bind it.
        assert args.decider_port != parse_args([*SOURCE, "--kev"]).kev_port

    def test_decider_pins_its_checkpoint_and_its_package(self):
        args = parse_args([*SOURCE, "--decider"])
        assert args.decider_run == "Mapika/decider-4b@v2"
        assert args.decider_package == DEFAULT_PACKAGE
        assert "[serve]" in args.decider_package

    def test_the_decider_model_name_is_discovered_by_default(self):
        # There is no stable string to hardcode: decider.serve names the
        # checkpoint after the version in its own config. A default here would
        # 404 against every checkpoint but one.
        assert parse_args([*SOURCE, "--decider"]).decider_model is None

    def test_an_explicit_decider_model_is_kept(self):
        args = parse_args([*SOURCE, "--decider", "--decider-model", "decider-v2.1"])
        assert args.decider_model == "decider-v2.1"

    @pytest.mark.parametrize("flag", ["--jev", "--kev", "--decider"])
    def test_no_provider_may_share_a_single_url(self, flag):
        with pytest.raises(SystemExit):
            parse_args(["--url", "https://youtu.be/x", flag])

    @pytest.mark.parametrize("flag", ["--jev", "--kev", "--decider"])
    @pytest.mark.parametrize("other", ["--verify", "--repair", "--review", "--dry-run"])
    def test_no_provider_may_run_alone_against_stored_state(self, flag, other):
        with pytest.raises(SystemExit):
            parse_args([*SOURCE, flag, other])

    @pytest.mark.parametrize("port_flag", ["--kev-port", "--decider-port"])
    def test_ports_are_range_checked(self, port_flag):
        with pytest.raises(SystemExit):
            parse_args([*SOURCE, port_flag, "70000"])

    @pytest.mark.parametrize(
        "timeout_flag", ["--kev-startup-timeout", "--decider-startup-timeout"]
    )
    def test_startup_timeouts_must_be_positive(self, timeout_flag):
        with pytest.raises(SystemExit):
            parse_args([*SOURCE, timeout_flag, "0"])


class TestProfiles:
    # A profile is not a source, so every call below supplies --retry. Without it
    # the parser stops at "exactly one of --file, --data, --url, or --retry is
    # required" and the flag under test is never reached.

    def test_a_profile_can_set_the_merged_flags(self, tmp_path):
        path = _profile(
            tmp_path,
            "[profiles.gpu]\n"
            "decider = true\n"
            "decision_threshold = 0.72\n"
            "decision_runs = 2\n",
        )
        args = parse_args(["--retry", "--profile", "gpu", "--profiles-file", str(path)])
        assert args.decider is True
        assert args.decision_threshold == 0.72
        assert args.decision_runs == 2

    def test_an_old_profile_key_is_mapped_and_reported(self, tmp_path, capsys):
        path = _profile(tmp_path, "[profiles.old]\nkev_runs = 5\n")
        args = parse_args(["--retry", "--profile", "old", "--profiles-file", str(path)])
        assert args.decision_runs == 5
        assert "deprecated" in capsys.readouterr().err

    def test_setting_both_a_key_and_its_replacement_is_an_error(self, tmp_path, capsys):
        # Silently letting one win would make the profile's behaviour depend on
        # dict ordering rather than on what it says.
        path = _profile(tmp_path, "[profiles.both]\nkev_runs = 5\ndecision_runs = 2\n")
        with pytest.raises(SystemExit):
            parse_args(["--retry", "--profile", "both", "--profiles-file", str(path)])
        assert "keep only 'decision_runs'" in capsys.readouterr().err

    def test_a_profile_can_select_the_decider_server(self, tmp_path):
        path = _profile(
            tmp_path,
            "[profiles.local]\ndecider = true\ndecider_port = 9001\n"
            "decider_run = 'Mapika/decider-2b@v11'\n",
        )
        args = parse_args(["--retry", "--profile", "local", "--profiles-file", str(path)])
        assert args.decider_port == 9001
        assert args.decider_run == "Mapika/decider-2b@v11"

    def test_the_run_count_is_still_type_checked_under_its_old_name(self, tmp_path, capsys):
        path = _profile(tmp_path, '[profiles.bad]\nkev_runs = "three"\n')
        with pytest.raises(SystemExit):
            parse_args(["--retry", "--profile", "bad", "--profiles-file", str(path)])
        assert "must be an integer" in capsys.readouterr().err

    def test_the_calibration_log_is_off_unless_asked_for(self):
        """It writes a line per song; nobody should pay that by accident."""
        assert parse_args(SOURCE).decision_calibration_log == ""
        assert Config.DECISION_CALIBRATION_LOG == ""

    def test_the_calibration_log_can_be_set_from_a_profile(self, tmp_path):
        path = _profile(
            tmp_path,
            "[profiles.tune]\ndecider = true\ndecision_calibration_log = 'calib.jsonl'\n",
        )
        args = parse_args(["--retry", "--profile", "tune", "--profiles-file", str(path)])
        assert Path(args.decision_calibration_log) == Path("calib.jsonl")

    def test_the_calibration_log_is_type_checked(self, tmp_path, capsys):
        path = _profile(tmp_path, "[profiles.bad]\ndecision_calibration_log = 7\n")
        with pytest.raises(SystemExit):
            parse_args(["--retry", "--profile", "bad", "--profiles-file", str(path)])
        assert "must be a path" in capsys.readouterr().err


class TestDeciderGraphFlags:
    """The narrow CUDA-graph grid is the default, the wide one is reachable, and
    the kernel flags never probe once the operator has said what they want.
    """

    def test_the_default_grid_is_the_narrow_serial_one(self):
        args = parse_args(SOURCE)
        assert args.decider_graphs_b == "1"
        assert all(t <= 1024 for t in map(int, args.decider_graphs_t.split(",")))

    def test_the_grid_flags_reach_the_parser(self):
        args = parse_args(SOURCE + ["--decider-graphs-t", "64,128", "--decider-graphs-b", "1,2"])
        assert args.decider_graphs_t == "64,128"
        assert args.decider_graphs_b == "1,2"

    def test_the_kernel_flags_are_tri_state(self):
        args = parse_args(SOURCE)
        assert args.decider_kernels is None
        args = parse_args(SOURCE + ["--decider-kernels"])
        assert args.decider_kernels is True
        args = parse_args(SOURCE + ["--no-decider-kernels"])
        assert args.decider_kernels is False

    def test_the_kernel_flags_conflict_rather_than_last_wins(self):
        with pytest.raises(SystemExit):
            parse_args(SOURCE + ["--decider-kernels", "--no-decider-kernels"])

    def test_a_profile_can_carry_the_grid(self, tmp_path):
        path = _profile(
            tmp_path,
            "[profiles.wide]\ndecider = true\n"
            "decider_graphs_t = ''\ndecider_graphs_b = ''\n",
        )
        args = parse_args(["--retry", "--profile", "wide", "--profiles-file", str(path)])
        # Empty = leave decider.serve's own default, i.e. the wide grid.
        assert args.decider_graphs_t == ""
        assert args.decider_graphs_b == ""
