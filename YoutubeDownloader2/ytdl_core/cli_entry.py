from __future__ import annotations


def main() -> None:
    try:
        from .cli import main as cli_main

        cli_main()
    except ModuleNotFoundError as error:
        if error.name == "rich":
            raise SystemExit("CLI dependencies are missing. Install ytdl-core[cli].") from error
        raise


if __name__ == "__main__":
    main()
