"""
MusicDownloader — the library-facing core class.

This module contains NO Rich / CLI code. All user-facing output is delegated
to the DownloaderEvents instance supplied at construction time.
"""

from __future__ import annotations

import concurrent.futures
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import acoustid
import mutagen
import musicbrainzngs
import yt_dlp
from pydub import AudioSegment
from pydub.silence import detect_silence
from rapidfuzz import fuzz
from yt_dlp.utils import DownloadError, ExtractorError, download_range_func

from config import Config
from events import DownloaderEvents
from metadata import embed_metadata, fetch_musicbrainz
from reports import export_report, update_json_file
from result import DownloadResult
from search import build_search_query, score_youtube_result, search_source, select_best_result
from state import load_state, save_state
from utils import apply_delay, compute_md5, format_duration, sanitize_filename


class MusicDownloader:
    def __init__(
        self,
        config: Optional[Config] = None,
        events: Optional[DownloaderEvents] = None,
        acoustid_key: Optional[str] = None,
        skip_fingerprint: bool = False,
        no_silence_check: bool = False,
        score_threshold: Optional[int] = None,
        sources: Optional[list[str]] = None,
        workers: int = 2,
        delay: tuple[float, float] = (2.0, 5.0),
        max_results: int = 5,
        fuzzy_threshold: int = 65,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        musicbrainz: bool = False,
        cookies_browser: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self.config = config or Config()
        self.events = events or DownloaderEvents()
        self.acoustid_key = acoustid_key
        self.skip_fingerprint = skip_fingerprint
        self.no_silence_check = no_silence_check
        self.score_threshold = (
            score_threshold if score_threshold is not None
            else self.config.SCORE_THRESHOLD_REJECT
        )
        self.sources = sources or list(self.config.DEFAULT_SOURCES)
        self.workers = max(1, min(workers, self.config.MAX_WORKERS))
        self.delay = delay
        self.max_results = max_results
        self.fuzzy_threshold = fuzzy_threshold
        self.max_duration = max_duration or self.config.MAX_DURATION_SECONDS
        self.min_duration = min_duration or self.config.MIN_DURATION_SECONDS
        self.musicbrainz = musicbrainz
        self.cookies_browser = cookies_browser
        self.proxy = proxy
        self.fpcalc_available: bool = shutil.which("fpcalc") is not None
        self._fp_semaphore = threading.Semaphore(2)


    def download(
        self,
        artist: str,
        song: str,
        output_dir: Path,
        fmt: str = "mp3",
        quality: str = "192",
        skip_existing: bool = False,
    ) -> DownloadResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(output_dir)
        state_lock = threading.Lock()
        stop_event = threading.Event()
        return self._process_song(
            artist=artist, song=song, output_dir=output_dir,
            fmt=fmt, quality=quality, skip_existing=skip_existing,
            state=state, state_lock=state_lock, stop_event=stop_event,
            seen_artists=set(), seen_artists_lock=threading.Lock(),
            all_pairs=[(artist, song)],
        )

    def download_batch(
        self,
        songs: dict[str, list[str]],
        output_dir: Path,
        fmt: str = "mp3",
        quality: str = "192",
        skip_existing: bool = False,
        report_formats: Optional[list[str]] = None,
        update_json_path: Optional[Path] = None,
    ) -> list[DownloadResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = [(artist, song) for artist, lst in songs.items() for song in lst if lst]
        total = len(pairs)
        self.events.on_session_start(total)

        state = load_state(output_dir)
        state_lock = threading.Lock()
        stop_event = threading.Event()
        seen_artists: set[str] = set()
        seen_artists_lock = threading.Lock()
        all_results: list[DownloadResult] = []
        results_lock = threading.Lock()
        start = time.monotonic()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(
                        self._process_song,
                        artist=artist, song=song, output_dir=output_dir,
                        fmt=fmt, quality=quality, skip_existing=skip_existing,
                        state=state, state_lock=state_lock, stop_event=stop_event,
                        seen_artists=seen_artists, seen_artists_lock=seen_artists_lock,
                        all_pairs=pairs,
                    ): (artist, song)
                    for artist, song in pairs
                    if not stop_event.is_set()
                }
                for fut in concurrent.futures.as_completed(futures):
                    artist, song = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = DownloadResult(
                            artist=artist, song=song, status="failed",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    with results_lock:
                        all_results.append(result)
                    self.events.on_result(result)

        except KeyboardInterrupt:
            stop_event.set()
            elapsed = time.monotonic() - start
            self.events.on_interrupted(len(all_results), total, elapsed)
            save_state(state, output_dir)
            self.events.on_session_complete(all_results, elapsed)
            return all_results

        elapsed = time.monotonic() - start
        self.events.on_session_complete(all_results, elapsed)

        if report_formats:
            export_report([r.to_dict() for r in all_results], output_dir, report_formats)
        if update_json_path:
            update_json_file(update_json_path, [r.to_dict() for r in all_results])

        return all_results


    def _process_song(
        self,
        artist: str, song: str, output_dir: Path,
        fmt: str, quality: str, skip_existing: bool,
        state: dict, state_lock: threading.Lock,
        stop_event: threading.Event,
        seen_artists: set, seen_artists_lock: threading.Lock,
        all_pairs: list[tuple[str, str]],
    ) -> DownloadResult:
        with seen_artists_lock:
            if artist not in seen_artists:
                seen_artists.add(artist)
                count = sum(1 for a, _ in all_pairs if a == artist)
                self.events.on_artist_start(artist, count)

        result = DownloadResult(artist=artist, song=song)
        key = f"{artist}::{song}"

        if stop_event.is_set():
            result.status = "skipped"
            result.reason = "Interrupted"
            return result

        safe_artist = sanitize_filename(artist)
        safe_song = sanitize_filename(song)
        expected_file = output_dir / safe_artist / f"{safe_song}.{fmt}"

        with state_lock:
            existing = state.get("downloads", {}).get(key)

        if skip_existing and existing and existing.get("status") == "downloaded":
            stored_md5 = existing.get("md5")
            if expected_file.exists():
                if stored_md5:
                    if compute_md5(expected_file) == stored_md5:
                        self.events.on_skip_existing(artist, song, expected_file, True)
                        result.status = "skipped"
                        result.file_path = expected_file
                        result.md5 = stored_md5
                        return result
                    self.events.on_md5_mismatch(artist, song)
                else:
                    self.events.on_skip_existing(artist, song, expected_file, False)
                    result.status = "skipped"
                    result.file_path = expected_file
                    return result

        apply_delay(self.delay[0], self.delay[1])

        search_opts = {
            "max_results": self.max_results,
            "cookies_browser": self.cookies_browser,
            "proxy": self.proxy,
        }

        best_result: Optional[dict] = None
        ranked_candidates: list[tuple[dict, int, dict]] = []
        chosen_source: Optional[str] = None

        for source in self.sources:
            if stop_event.is_set():
                break
            self.events.on_search_start(artist, song, source)
            raw = search_source(build_search_query(artist, song, source), source, search_opts)
            if not raw:
                self.events.on_no_results(artist, song, source)
                continue
            found, ranked = select_best_result(
                results=raw, artist=artist, song=song,
                mb_duration_seconds=None, config=self.config,
                console=None, console_lock=None,
                min_duration=self.min_duration, max_duration=self.max_duration,
                score_threshold=self.score_threshold,
            )
            self.events.on_candidates_scored(artist, song, ranked)
            if found:
                best_result = found
                ranked_candidates = ranked
                chosen_source = source
                break

        if best_result is None:
            self.events.on_search_failed(artist, song, self.sources)
            result.reason = "No valid result found after all sources"
            self._persist(state, state_lock, key, "failed", None, None, None, output_dir)
            return result

        url = best_result.get("webpage_url") or best_result.get("url", "")
        matched_title = best_result.get("title") or ""
        duration_s = int(best_result.get("duration") or 0)
        thumbnail_url = best_result.get("thumbnail")
        composite_score = best_result.get("_composite_score", 0)
        score_breakdown = best_result.get("_score_breakdown", {})

        result.source = chosen_source
        result.url = url
        result.matched_title = matched_title
        result.fuzzy_score = int(fuzz.token_sort_ratio(
            f"{artist} {song}".lower(), matched_title.lower()
        ))
        result.duration_seconds = duration_s
        result.composite_score = composite_score
        result.score_breakdown = score_breakdown

        fp_verified = False
        fp_confidence = 0.0
        fp_matched_title: Optional[str] = None
        fp_label = "disabled"

        needs_fp = (
            bool(self.acoustid_key)
            and not self.skip_fingerprint
            and self.fpcalc_available
            and composite_score < self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT
        )

        if self.acoustid_key and composite_score >= self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT:
            fp_label = f"skipped -- score {composite_score} >= threshold"
        elif self.acoustid_key and not self.fpcalc_available:
            fp_label = "disabled -- fpcalc not found"
        elif self.skip_fingerprint:
            fp_label = "disabled -- --skip-fingerprint"

        if needs_fp:
            partial_path: Optional[Path] = None
            self.events.on_fingerprint_start(artist, song, self.config.PARTIAL_DOWNLOAD_SECONDS)
            try:
                with self._fp_semaphore:
                    partial_path = self._download_partial(url, output_dir)
                    if partial_path is None:
                        self.events.on_fingerprint_partial_failed(artist, song)
                        fp_label = "partial download failed"
                    else:
                        is_match, conf, fp_title = self._verify_fingerprint(partial_path, artist, song)
                        time.sleep(0.35)
                        fp_confidence = conf
                        fp_matched_title = fp_title
                        self.events.on_fingerprint_result(artist, song, is_match, conf, fp_title)

                        if is_match:
                            fp_verified = True
                            fp_label = f"verified {conf:.0%} conf."
                        elif conf > 0.4:
                            fp_label = f"low confidence ({fp_title})"
                            self.events.on_fingerprint_low_confidence(artist, song, fp_title)
                            for cand_r, cand_s, _bd in ranked_candidates[1:]:
                                if cand_s < self.score_threshold:
                                    break
                                next_url = cand_r.get("webpage_url") or cand_r.get("url", "")
                                next_partial: Optional[Path] = None
                                try:
                                    next_partial = self._download_partial(next_url, output_dir)
                                    if next_partial:
                                        n_ok, n_conf, n_title = self._verify_fingerprint(
                                            next_partial, artist, song
                                        )
                                        time.sleep(0.35)
                                        if n_ok:
                                            best_result = cand_r
                                            url = next_url
                                            matched_title = cand_r.get("title") or ""
                                            duration_s = int(cand_r.get("duration") or 0)
                                            fp_verified = True
                                            fp_confidence = n_conf
                                            fp_matched_title = n_title
                                            fp_label = f"verified next candidate {n_conf:.0%}"
                                            result.url = url
                                            result.matched_title = matched_title
                                            result.duration_seconds = duration_s
                                            result.composite_score = cand_r.get("_composite_score", 0)
                                            result.score_breakdown = cand_r.get("_score_breakdown", {})
                                            break
                                finally:
                                    if next_partial and next_partial.exists():
                                        next_partial.unlink(missing_ok=True)
                                break
                        else:
                            self.events.on_fingerprint_no_match(artist, song)
                            fp_label = "no AcoustID match"
            finally:
                if partial_path and partial_path.exists():
                    partial_path.unlink(missing_ok=True)

        sc = result.composite_score
        score_label = (
            "high confidence" if sc >= self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT
            else "moderate" if sc >= self.config.SCORE_THRESHOLD_REJECT
            else "low"
        )
        self.events.on_verification_status(artist, song, sc, score_label, fp_label)
        result.fingerprint_verified = fp_verified
        result.fingerprint_confidence = fp_confidence
        result.fingerprint_matched_title = fp_matched_title

        (output_dir / safe_artist).mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / safe_artist / f"{safe_song}.%(ext)s")
        self.events.on_download_start(artist, song, url)

        def _progress_hook(d: dict) -> None:
            if d.get("status") == "downloading":
                total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded_b = d.get("downloaded_bytes") or 0
                pct = (downloaded_b / total_b * 100.0) if total_b else 0.0
                self.events.on_download_progress(
                    artist, song, pct, d.get("speed") or 0.0, downloaded_b, total_b
                )

        ydl_opts: Any = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [_progress_hook],
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                "preferredcodec": fmt, "preferredquality": quality}],
            "noplaylist": True,
        }
        if self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)
        if self.proxy:
            ydl_opts["proxy"] = self.proxy

        downloaded_file: Optional[Path] = None
        last_error = ""

        for attempt in range(1, self.config.RETRY_ATTEMPTS + 1):
            if stop_event.is_set():
                break
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                for ext in (fmt, "mp3", "m4a", "opus", "webm", "ogg"):
                    candidate = output_dir / safe_artist / f"{safe_song}.{ext}"
                    if candidate.exists():
                        downloaded_file = candidate
                        break
                if downloaded_file and downloaded_file.exists():
                    break
            except DownloadError as exc:
                last_error = f"DownloadError: {exc}"
            except ExtractorError as exc:
                last_error = f"ExtractorError: {exc}"
            except OSError as exc:
                if exc.errno == 28:
                    self.events.on_disk_full()
                    stop_event.set()
                    save_state(state, output_dir)
                    result.reason = "Disk full"
                    return result
                last_error = f"OSError: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.config.RETRY_ATTEMPTS:
                wait = self.config.RETRY_BACKOFF_BASE ** attempt
                self.events.on_download_retry(
                    artist, song, attempt, self.config.RETRY_ATTEMPTS, last_error, wait
                )
                time.sleep(wait)

        if downloaded_file is None or not downloaded_file.exists():
            self.events.on_download_failed(artist, song, last_error)
            result.reason = last_error
            self._persist(state, state_lock, key, "failed", url, None, None, output_dir)
            return result

        dur_ok, actual_dur = self._verify_duration(downloaded_file, duration_s)
        result.duration_verified = dur_ok
        self.events.on_duration_check(artist, song, duration_s, actual_dur, dur_ok)

        if not dur_ok and duration_s > 0:
            discrepancy = abs(actual_dur - duration_s) / max(duration_s, 1)
            if discrepancy > 0.40:
                downloaded_file.unlink(missing_ok=True)
                result.reason = f"Duration discrepancy {discrepancy:.0%}"
                self._persist(state, state_lock, key, "failed", url, None, None, output_dir)
                return result

        silence_ratio = 0.0
        if not self.no_silence_check:
            is_excessive, silence_ratio = self._has_excessive_silence(downloaded_file)
            result.silence_ratio = silence_ratio
            self.events.on_silence_check(artist, song, silence_ratio, is_excessive)
            if is_excessive:
                self.events.on_silence_rejected(artist, song, silence_ratio)
                downloaded_file.unlink(missing_ok=True)
                result.reason = f"Excessive silence ({silence_ratio:.1%})"
                self._persist(state, state_lock, key, "failed", url, None, None, output_dir)
                return result

        self.events.on_post_check_summary(artist, song, dur_ok, actual_dur, silence_ratio)

        mb_data: Optional[dict] = None
        mb_enriched = False
        if self.musicbrainz:
            mb_data = fetch_musicbrainz(artist, song)
            mb_enriched = mb_data is not None
            self.events.on_musicbrainz_result(artist, song, mb_enriched, mb_data or {})

        extra = {
            "source_url": url,
            "album": mb_data.get("album") if mb_data else None,
            "year": mb_data.get("year") if mb_data else None,
            "genre": mb_data.get("genre") if mb_data else None,
            "track_num": mb_data.get("track_num") if mb_data else None,
            "mb_id": mb_data.get("mb_id") if mb_data else None,
            "cover_url": mb_data.get("cover_url") if mb_data else None,
        }

        embed_ok = embed_metadata(
            downloaded_file, song, artist, extra, thumbnail_url, fmt,
            lambda msg: self.events.on_warn(msg),
        )

        if not embed_ok:
            self.events.on_metadata_error(artist, song, downloaded_file.name)
            downloaded_file.unlink(missing_ok=True)
            result.reason = "Metadata integrity check failed"
            self._persist(state, state_lock, key, "failed", url, None, None, output_dir)
            return result

        md5 = compute_md5(downloaded_file)
        file_size = downloaded_file.stat().st_size

        result.status = "downloaded"
        result.file_path = downloaded_file
        result.file_size_bytes = file_size
        result.md5 = md5
        result.musicbrainz_enriched = mb_enriched
        result.album = extra.get("album")
        result.year = extra.get("year")
        result.genre = extra.get("genre")
        result.silence_ratio = silence_ratio
        result.duration_verified = dur_ok

        self._persist(
            state, state_lock, key, "downloaded",
            url, str(downloaded_file), md5, output_dir,
        )
        return result


    def _download_partial(self, url: str, output_dir: Path) -> Optional[Path]:
        token = uuid4().hex[:8]
        expected = output_dir / f"_partial_{token}.mp3"
        ydl_opts: Any = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / f"_partial_{token}.%(ext)s"),
            "quiet": True, "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                "preferredcodec": "mp3", "preferredquality": "128"}],
            "download_ranges": download_range_func(
                [], [(0, self.config.PARTIAL_DOWNLOAD_SECONDS)]
            ),
            "force_keyframes_at_cuts": True,
        }
        if self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)
        if self.proxy:
            ydl_opts["proxy"] = self.proxy
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if expected.exists():
                return expected
            for candidate in output_dir.glob(f"_partial_{token}.*"):
                return candidate
        except Exception:
            pass
        return None

    def _verify_fingerprint(
        self, partial_path: Path, artist: str, song: str
    ) -> tuple[bool, float, str]:
        if not self.acoustid_key:
            return False, 0.0, "no_key"
        try:
            results = list(acoustid.match(
                self.acoustid_key, str(partial_path), meta="recordings"
            ))
            best_conf = 0.0
            best_title = ""
            for score, _rec_id, title, a in results:
                if score < self.config.FINGERPRINT_MIN_CONFIDENCE:
                    continue
                a_sim = fuzz.token_sort_ratio(artist.lower(), (a or "").lower())
                t_sim = fuzz.token_sort_ratio(song.lower(), (title or "").lower())
                if a_sim > 75 and t_sim > 75:
                    return True, score, title or ""
                if score > best_conf:
                    best_conf = score
                    best_title = f"{a} -- {title}"
            return False, best_conf, best_title
        except Exception as exc:
            self.events.on_fingerprint_error(artist, song, str(exc))
            return False, 0.0, "fingerprint_error"

    def _has_excessive_silence(self, file_path: Path) -> tuple[bool, float]:
        try:
            audio = AudioSegment.from_file(str(file_path))
            silent_ranges = detect_silence(
                audio,
                min_silence_len=self.config.SILENCE_MIN_DURATION_MS,
                silence_thresh=self.config.SILENCE_THRESHOLD_DB,
            )
            total_ms = sum(e - s for s, e in silent_ranges)
            ratio = total_ms / len(audio) if len(audio) > 0 else 0.0
            return ratio > self.config.EXCESSIVE_SILENCE_RATIO, ratio
        except Exception:
            return False, 0.0

    def _verify_duration(
        self, path: Path, expected: int, tolerance: float = 0.20
    ) -> tuple[bool, int]:
        try:
            info = mutagen.File(str(path)) # type: ignore
            if info is None or info.info is None:
                return False, 0
            actual = int(info.info.length)
            if expected == 0:
                return True, actual
            ratio = abs(actual - expected) / max(expected, 1)
            return ratio <= tolerance, actual
        except Exception:
            return False, 0


    @staticmethod
    def _persist(
        state: dict, lock: threading.Lock, key: str, status: str,
        url: Optional[str], file_path: Optional[str], md5: Optional[str],
        output_dir: Path,
    ) -> None:
        with lock:
            state.setdefault("downloads", {})[key] = {
                "status": status, "url": url, "file_path": file_path,
                "md5": md5,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state, output_dir)