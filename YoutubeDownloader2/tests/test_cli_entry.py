from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

from ytdl_core.cli_entry import main


def test_cli_package_import_does_not_load_rich() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ytdl_core.cli; raise SystemExit('rich' in sys.modules)",
        ],
        check=False,
    )

    assert completed.returncode == 0


def test_cli_entry_delegates_to_cli() -> None:
    with patch("ytdl_core.cli.main", return_value=None) as cli_main:
        assert main() is None

    cli_main.assert_called_once_with()
