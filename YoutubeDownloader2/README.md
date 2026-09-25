# ytdl-core

Batch audio downloader with composite scoring, AcoustID fingerprint verification, and MusicBrainz enrichment.

## Features

* Batch audio downloading capabilities.
* Composite scoring mechanism to ensure high accuracy when selecting tracks.
* AcoustID fingerprint verification for robust audio identification.
* MusicBrainz integration for metadata enrichment and curation.
* Extensible and user-friendly Command Line Interface.

## Requirements

* Python 3.11 or higher.
* FFmpeg (required by yt-dlp and pydub for audio processing).

## Installation

You can install the package locally using `pip`:

```bash
# Basic installation
pip install .

# Installation with CLI and Developer dependencies
pip install ".[cli,dev]"
```

## Usage

After installing with the `cli` dependencies, you can use the command-line interface:

```bash
ytdl --help
```

## Profiles

Copy `profiles.example.toml` to `profiles.toml` and run:

```bash
ytdl --profile high-quality --file songs.json
```

Explicit command-line options always override profile values.

## Retry queue

Failed songs are saved to `retry_queue.json` inside the output directory. Retry
only those songs with:

```bash
ytdl --output ./downloads --retry
```

Successful retries are removed from the queue automatically. The final summary
shows pending retries, unverified files, and the exact command for the next
recommended action.

## License

This project is licensed under the MIT License.
