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

## License

This project is licensed under the MIT License.
