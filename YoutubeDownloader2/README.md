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

## How a batch runs

A batch is a **stage pipeline**. Each stage gets its own pool of threads, sized
against the resource it actually contends for, and hands work on through bounded
queues:

| Stage | Work | Sized against |
| --- | --- | --- |
| search | multi-source search, ranking, decision model | network |
| verify | 90-second partial download + AcoustID lookup | the AcoustID request budget |
| download | full download with candidate fallback | network |
| post | duration/silence checks, tagging, checksums | CPU |

Because the queues are bounded, a slow stage pushes back on the stage feeding it
instead of letting the whole batch queue up behind it. Results are reported as
they finish but returned in the order the songs were requested, so a report for
a large batch is reproducible.

```bash
ytdl --file songs.json --musicbrainz --workers 8
ytdl --file songs.json --stage-workers 12 8 12 6   # search verify download post
ytdl --file songs.json --no-pipeline                # one worker, song at a time
```

## Rate limiting

Remote services publish per-process budgets, and they are enforced by token
buckets that block **only when a request would actually exceed the budget** —
not with a fixed sleep charged to every song. Backoff uses full jitter, so
workers that fail together do not retry together, and `Retry-After` is honoured.

| Service | Budget |
| --- | --- |
| AcoustID | 3 requests/second (published limit) |
| MusicBrainz | 1 request/second |
| YouTube / other search | configurable |

If a service throttles us repeatedly, a circuit breaker opens and widens its
cooldown, letting exactly one probe through once it expires.

`--delay MIN MAX` still adds an extra pause per song if you want it; it is off
by default.

## Decision model

`--kev` runs a local [Kev](https://github.com/jaredpalmer/kev) server on CUDA and
picks the candidate with it; `--jev` uses the hosted TypeSafe model instead. They
are mutually exclusive.

```bash
ytdl --file songs.json --kev                     # auto-start a local Kev server
ytdl --file songs.json --kev --decision-questions 20   # faster, fewer questions
ytdl --file songs.json --kev --decision-in-flight 8    # let the server batch more
```

The server's health, CUDA graph usage, batching and prefix-cache statistics are
reported at startup from `/v1/models`, so a silent slowdown shows up there
rather than as a thousand identical timeouts.

`--decision-questions` is the main speed dial: every question is prefilled as its
own sequence, so the cost of an evaluation scales with the question count
(candidates × dimensions, plus one tie-break). Lower it to go faster, raise it to
arbitrate more candidates.

`--kev-fused` / `--no-kev-fused` override the fused-kernel probe. By default the
fused Qwen3.5 kernels are installed if missing and then *proven* by running a real
evaluation under a timeout, because some Triton/platform combinations import
cleanly and then never return.

## Downloads

```bash
ytdl --file songs.json --fragment-concurrency 8
ytdl --file songs.json --use-aria2c
```

Both were measured on this project and **neither is a speed-up as configured**:

- `--fragment-concurrency` is **inert**. With the configured YouTube player
  clients, yt-dlp is offered audio as a single progressive HTTPS file (measured:
  format 251, webm/opus, no fragments), so there is nothing to fetch
  concurrently. It applies if a format ever arrives segmented.
- `--use-aria2c` was **75% slower** on a ~450 KB/s link (22.3 s vs 12.7 s for the
  same file, consistent across repeats). Splitting one connection's worth of
  bandwidth across eight costs more in connection setup than it recovers. It is
  the right tool where a single connection is throttled or latency is high, and
  the wrong one on a slow pipe — so it is off by default.

The knobs exist because they cost nothing when unused and are the first thing to
reach for when the situation changes. Neither is presented here as a win.

## Caching

Candidate lists, MusicBrainz/iTunes metadata, cover art and AcoustID verdicts are
cached under `~/.cache/ytdl`, each with its own TTL. Re-running a batch, retrying
failures, and re-verifying a library all reuse them. `--no-cache` bypasses this,
and `YTDL_NO_CACHE=1` disables it permanently.

Set `YTDL_CACHE_DIR` to move the cache elsewhere.

## Albums

An artist's songs are usually one album. Once any of them resolves to a MusicBrainz
release, the rest are answered from that release's tracklist — two throttled
calls for a whole album instead of one per song, and an exact tracklist rather
than a search-ranked guess.

## License

This project is licensed under the MIT License.
