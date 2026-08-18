"""
MusicDownloader — the library-facing core class.

This module contains NO Rich / CLI code. All user-facing output is delegated
to the DownloaderEvents instance supplied at construction time.
"""

from __future__ import annotations

import concurrent.futures
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

import yt_dlp
from rapidfuzz import fuzz

from .config import Config
from .downloader import download_partial, execute_download
from .events import DownloaderEvents
from .fingerprint import AcoustIDCircuitBreaker, verify_fingerprint
from .metadata import fetch_musicbrainz
from .post_checks import check_duration, check_silence, embed_and_verify, enrich_musicbrainz
from .reports import export_report, update_json_file
from .result import DownloadResult
from .search import search_all_sources, select_best_result
from .state import load_state, save_state
from .utils import apply_delay, compute_md5, sanitize_filename
from .verifier import verify_library as _verify_library
from .ytdlp_options import build_ytdlp_base_opts, make_progress_hook, resolve_downloaded_file


class MusicDownloader:
    def __init__(self, config=None, events=None, acoustid_key=None, force_fingerprint=False,
                 skip_fingerprint=False, no_silence_check=False, score_threshold=None,
                 sources=None, workers=2, delay=(2.0, 5.0), max_results=5, fuzzy_threshold=65,
                 max_duration=None, min_duration=None, musicbrainz=False, cookies_browser=None,
                 cookies_file=None, proxy=None):
        self.config = config or Config()
        self.events = events or DownloaderEvents()
        self.acoustid_key = acoustid_key
        self.force_fingerprint = force_fingerprint
        self.skip_fingerprint = skip_fingerprint
        self.no_silence_check = no_silence_check
        self.score_threshold = score_threshold if score_threshold is not None else self.config.SCORE_THRESHOLD_REJECT
        self.sources = sources or list(self.config.DEFAULT_SOURCES)
        self.workers = max(1, min(workers, self.config.MAX_WORKERS))
        self.delay = delay
        self.max_results = max_results
        self.fuzzy_threshold = fuzzy_threshold
        self.max_duration = max_duration or self.config.MAX_DURATION_SECONDS
        self.min_duration = min_duration or self.config.MIN_DURATION_SECONDS
        self.musicbrainz = musicbrainz
        self.cookies_browser = cookies_browser
        self.cookies_file = cookies_file
        self.proxy = proxy
        self.fpcalc_available = shutil.which("fpcalc") is not None
        self._fp_semaphore = threading.Semaphore(2)
        self._circuit_breaker = AcoustIDCircuitBreaker(cooldown_seconds=60.0)
        self._selection_lock = threading.Lock()

    # Public API

    def download(self, artist, song, output_dir, fmt="mp3", quality="192", skip_existing=False):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(output_dir)
        return self._process_song(artist, song, output_dir, fmt, quality, skip_existing,
                                  state, threading.Lock(), threading.Event(), set(), threading.Lock(), [(artist, song)])

    def download_batch(self, songs, output_dir, fmt="mp3", quality="192", skip_existing=False,
                       report_formats=None, update_json_path=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = [(a, s) for a, lst in songs.items() for s in lst if lst]
        self.events.on_session_start(len(pairs))
        state = load_state(output_dir)
        state_lock = threading.Lock()
        stop = threading.Event()
        seen, seen_lock = set(), threading.Lock()
        all_results = []
        results_lock = threading.Lock()
        start = time.monotonic()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = {pool.submit(self._process_song, a, s, output_dir, fmt, quality,
                                    skip_existing, state, state_lock, stop, seen, seen_lock, pairs): (a, s)
                        for a, s in pairs if not stop.is_set()}
                for f in concurrent.futures.as_completed(futs):
                    a, s = futs[f]
                    try:
                        r = f.result()
                    except Exception as e:
                        r = DownloadResult(
                            artist=a, song=s, status="failed", reason=f"{type(e).__name__}: {e}")
                    with results_lock:
                        all_results.append(r)
                    self.events.on_result(r)
        except KeyboardInterrupt:
            stop.set()
            self.events.on_interrupted(
                len(all_results), len(pairs), time.monotonic() - start)
            save_state(state, output_dir)
            self.events.on_session_complete(
                all_results, time.monotonic() - start)
            return all_results
        elapsed = time.monotonic() - start
        self.events.on_session_complete(all_results, elapsed)
        if report_formats:
            export_report([r.to_dict()
                          for r in all_results], output_dir, report_formats)
        if update_json_path:
            update_json_file(update_json_path, [
                             r.to_dict() for r in all_results])
        return all_results

    def download_url(self, url, output_dir, fmt="mp3", quality="192", max_downloads=None,
                     skip_existing=False, match_title=None, reject_title=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        match_re = re.compile(
            match_title, re.IGNORECASE) if match_title else None
        reject_re = re.compile(
            reject_title, re.IGNORECASE) if reject_title else None
        scan_opts = build_ytdlp_base_opts(output_dir, fmt, quality, quiet=False, no_warnings=False,
                                          progress_hook=None, skip_existing=False, max_downloads=max_downloads,
                                          cookies_browser=self.cookies_browser, cookies_file=self.cookies_file,
                                          proxy=self.proxy, enable_remote_components=True, youtube_player_clients=["web"],
                                          noplaylist=False, for_scan=True)
        try:
            with yt_dlp.YoutubeDL(scan_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            self.events.on_download_failed("URL", url, str(exc))
            return
        if not info:
            return
        entries = list(self._iter_entries(info))
        urls = []
        for e in entries:
            if not e:
                continue
            title = e.get("title", "Unknown")
            uploader = e.get("uploader", "Unknown")
            if match_re and not match_re.search(title):
                self.events.on_warn(
                    f"[yellow]  Skipped (no match): {title}[/yellow]")
                continue
            if reject_re and reject_re.search(title):
                self.events.on_warn(
                    f"[yellow]  Skipped (rejected): {title}[/yellow]")
                continue
            if e.get("is_live"):
                self.events.on_warn(
                    f"[yellow]  Skipped (live): {title}[/yellow]")
                continue
            dur = e.get("duration")
            if dur is not None:
                if self.min_duration and dur < self.min_duration:
                    self.events.on_warn(
                        f"[yellow]  Skipped (too short): {title}[/yellow]")
                    continue
                if self.max_duration and dur > self.max_duration:
                    self.events.on_warn(
                        f"[yellow]  Skipped (too long): {title}[/yellow]")
                    continue
            iu = e.get("webpage_url") or e.get("url")
            if not iu and e.get("id"):
                iu = f"https://www.youtube.com/watch?v={e['id']}"
            if iu and iu != url and "search?" not in iu:
                urls.append((uploader, title, iu))
        if not urls:
            self.events.on_warn("[yellow]No suitable URLs found.[/yellow]")
            return
        self.events.on_session_start(len(urls))
        all_results = []
        lock = threading.Lock()
        stop = threading.Event()
        start = time.monotonic()

        def _dl_one(ia, it, iu):
            if stop.is_set():
                return
            r = DownloadResult(artist=ia, song=it)
            hook = make_progress_hook(self.events, ia, it)
            opts = build_ytdlp_base_opts(
                output_dir, fmt, quality, quiet=True, no_warnings=True, progress_hook=hook,
                skip_existing=skip_existing, cookies_browser=self.cookies_browser,
                cookies_file=self.cookies_file, proxy=self.proxy, enable_remote_components=True,
                youtube_player_clients=["web"], noplaylist=True, for_scan=False,
            )
            sf = output_dir / \
                sanitize_filename(ia) / f"{sanitize_filename(it)}.{fmt}"
            if skip_existing and sf.exists():
                self.events.on_skip_existing(ia, it, sf, True)
                r.status = "skipped"
                r.reason = "File exists"
                r.file_path = sf
                with lock:
                    all_results.append(r)
                    self.events.on_result(r)
                    return
            self.events.on_download_start(ia, it, iu)
            try:
                with yt_dlp.YoutubeDL(opts) as y:
                    d = y.extract_info(iu, download=True)
                    if not d:
                        raise RuntimeError("no info_dict")
                    f = resolve_downloaded_file(
                        Path(y.prepare_filename(d)), fmt)
                    if f is None:
                        raise FileNotFoundError("no output file")
                    r.status = "downloaded"
                    r.file_path = f
                    r.duration_seconds = d.get("duration")
            except Exception as exc:
                self.events.on_download_failed(ia, it, str(exc))
                r.status = "failed"
                r.reason = str(exc)
            with lock:
                all_results.append(r)
                self.events.on_result(r)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                fs = [pool.submit(_dl_one, a, t, u) for a, t, u in urls]
                for f in concurrent.futures.as_completed(fs):
                    try:
                        f.result()
                    except:
                        pass
        except KeyboardInterrupt:
            stop.set()
        self.events.on_session_complete(all_results, time.monotonic() - start)

    def verify_library(self, songs, output_dir, fmt="mp3"):
        return _verify_library(songs, Path(output_dir), fmt, self.workers, self.acoustid_key,
                               self.config, self._circuit_breaker, self._fp_semaphore,
                               self.musicbrainz, self.events, self._persist,
                               load_state(Path(output_dir)), threading.Lock())

    # Internal: single-song pipeline

    def _process_song(self, artist, song, output_dir, fmt, quality, skip_existing,
                      state, state_lock, stop_event, seen, seen_lock, all_pairs):
        with seen_lock:
            if artist not in seen:
                seen.add(artist)
                self.events.on_artist_start(artist, sum(
                    1 for a, _ in all_pairs if a == artist))
        result = DownloadResult(artist=artist, song=song)
        key = f"{artist}::{song}"
        if stop_event.is_set():
            result.status = "skipped"
            result.reason = "Interrupted"
            return result
        safe_a, safe_s = sanitize_filename(artist), sanitize_filename(song)
        expected = output_dir / safe_a / f"{safe_s}.{fmt}"
        with state_lock:
            existing = state.get("downloads", {}).get(key)
        if skip_existing and existing and existing.get("status") == "downloaded":
            md5s = existing.get("md5")
            if expected.exists():
                if md5s:
                    if compute_md5(expected) == md5s:
                        self.events.on_skip_existing(
                            artist, song, expected, True)
                        result.status = "skipped"
                        result.file_path = expected
                        result.md5 = md5s
                        return result
                    self.events.on_md5_mismatch(artist, song)
                else:
                    self.events.on_skip_existing(artist, song, expected, False)
                    result.status = "skipped"
                    result.file_path = expected
                    return result
        apply_delay(self.delay[0], self.delay[1])
        mb_dur = None
        if self.musicbrainz:
            try:
                mb = fetch_musicbrainz(artist, song)
            except Exception as e:
                self.events.on_warn(
                    f"[yellow]MusicBrainz failed: {e}[/yellow]")
                mb = None
            if mb:
                self.events.on_musicbrainz_result(artist, song, True, mb)
                mb_dur = mb.get("duration_seconds")
        best, ranked, src = self._search_and_select(
            artist, song, output_dir, state, state_lock, key, result, stop_event, mb_dur)
        if best is None:
            return result
        url = best.get("webpage_url") or best.get("url", "")
        dur_s = int(best.get("duration") or 0)
        result.source = src
        result.url = url
        result.matched_title = best.get("title") or ""
        result.fuzzy_score = int(fuzz.token_sort_ratio(
            f"{artist} {song}".lower(), (best.get("title") or "").lower()))
        result.duration_seconds = dur_s
        result.composite_score = best.get("_composite_score", 0)
        result.score_breakdown = best.get("_score_breakdown", {})
        fp_ok, fp_conf, fp_title, fp_label = self._fingerprint_check(
            artist, song, url, output_dir, best, ranked, result)
        sc = result.composite_score
        self.events.on_verification_status(artist, song, sc, "high confidence" if sc >=
                                           self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT else "moderate" if sc >= self.config.SCORE_THRESHOLD_REJECT else "low", fp_label)
        result.fingerprint_verified = fp_ok
        result.fingerprint_confidence = fp_conf
        result.fingerprint_matched_title = fp_title
        (output_dir / safe_a).mkdir(parents=True, exist_ok=True)
        self.events.on_download_start(artist, song, url)
        dl_file, err = execute_download(url, output_dir, fmt, quality, artist, song, self.events, self.config,
                                        stop_event, state, state_lock, self.cookies_browser, self.cookies_file, self.proxy)
        if dl_file is None:
            self.events.on_download_failed(artist, song, err)
            result.reason = err
            self._persist(state, state_lock, key, "failed",
                          url, None, None, output_dir)
            return result
        return self._post_download_checks(dl_file, artist, song, url, best.get("thumbnail"), fmt, dur_s, state, state_lock, key, result, output_dir)

    def _search_and_select(self, artist, song, output_dir, state, state_lock, key, result, stop_event, mb_dur):
        opts = {"max_results": self.max_results, "cookies_browser": self.cookies_browser,
                "cookies_file": self.cookies_file, "proxy": self.proxy}
        best, ranked, src = None, [], None
        if stop_event.is_set():
            result.status = "skipped"
            return best, ranked, src
        self.events.on_search_start(artist, song, "parallel sources")
        raw = search_all_sources(artist, song, self.sources, opts)
        if raw:
            found, ranked = select_best_result(
                raw, artist, song, mb_dur, self.config, None, None, self.min_duration, self.max_duration, self.score_threshold)
            has_sel = hasattr(self.events, "selector_fn") and callable(
                self.events.selector_fn)
            has_con = hasattr(self.events, "confirm_fn") and callable(
                self.events.confirm_fn)
            if not ranked:
                self.events.on_search_failed(artist, song, self.sources)
            else:
                self.events.on_candidates_scored(artist, song, ranked)
            if ranked:
                if has_sel:
                    with self._selection_lock:
                        if stop_event.is_set():
                            result.status = "skipped"
                            result.reason = "Interrupted"
                            return best, ranked, src
                        chosen = self.events.selector_fn(artist, song, ranked)
                    if chosen is None:
                        result.status = "skipped"
                        result.reason = "User skipped"
                        return best, ranked, src
                    best, src = chosen, chosen.get("_source", "unknown")
                elif has_con:
                    if found is None:
                        self.events.on_search_failed(
                            artist, song, self.sources)
                        result.reason = "No valid result"
                        self._persist(state, state_lock, key,
                                      "failed", None, None, None, output_dir)
                        return best, ranked, src
                    if not self.events.confirm_fn(artist, song, found):
                        result.status = "skipped"
                        result.reason = "User skipped"
                        return best, ranked, src
                    best, src = found, found.get("_source", "unknown")
                else:
                    if found:
                        best, src = found, found.get("_source", "unknown")
        else:
            for s in self.sources:
                self.events.on_no_results(artist, song, s)
        if best is None:
            self.events.on_search_failed(artist, song, self.sources)
            result.reason = "No valid result"
            self._persist(state, state_lock, key, "failed",
                          None, None, None, output_dir)
        return best, ranked, src

    def _fingerprint_check(self, artist, song, url, output_dir, best, ranked, result):
        fp_ok, fp_conf, fp_title, fp_label = False, 0.0, None, "disabled"
        sc = result.composite_score
        needs = self.force_fingerprint or (bool(
            self.acoustid_key) and not self.skip_fingerprint and self.fpcalc_available and sc < self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT)
        if self.acoustid_key and sc >= self.config.SCORE_THRESHOLD_SKIP_FINGERPRINT:
            fp_label = f"skipped -- score {sc} >= threshold"
        elif self.acoustid_key and not self.fpcalc_available:
            fp_label = "disabled -- fpcalc not found"
        elif self.skip_fingerprint:
            fp_label = "disabled -- --skip-fingerprint"
        if needs:
            self.events.on_fingerprint_start(
                artist, song, self.config.PARTIAL_DOWNLOAD_SECONDS)
            pp = None
            try:
                with self._fp_semaphore:
                    pp = download_partial(
                        url, output_dir, self.events, self.cookies_browser, self.cookies_file, self.proxy)
                    if pp is None:
                        self.events.on_fingerprint_partial_failed(artist, song)
                        fp_label = "partial download failed"
                    else:
                        is_m, conf, t = verify_fingerprint(pp, artist, song, self.acoustid_key, self.config, self._circuit_breaker,
                                                           on_warn=self.events.on_warn, on_info=self.events.on_info, on_fingerprint_error=self.events.on_fingerprint_error)
                        time.sleep(0.35)
                        fp_conf = conf
                        fp_title = t
                        self.events.on_fingerprint_result(
                            artist, song, is_m, conf, t)
                        if is_m:
                            fp_ok, fp_label = True, f"verified {conf:.0%} conf."
                        elif conf > 0.4:
                            fp_label = f"low confidence ({t})"
                            self.events.on_fingerprint_low_confidence(
                                artist, song, t)
                            fp_ok, fp_conf, fp_title, fp_label = self._try_next_fp(
                                ranked, artist, song, output_dir, result)
                        else:
                            self.events.on_fingerprint_no_match(artist, song)
                            fp_label = "no AcoustID match"
            finally:
                if pp and pp.exists():
                    pp.unlink(missing_ok=True)
        return fp_ok, fp_conf, fp_title, fp_label

    def _try_next_fp(self, ranked, artist, song, output_dir, result):
        for cr, cs, _ in ranked[1:]:
            if cs < self.score_threshold:
                break
            nu = cr.get("webpage_url") or cr.get("url", "")
            np_ = None
            try:
                np_ = download_partial(
                    nu, output_dir, self.events, self.cookies_browser, self.cookies_file, self.proxy)
                if np_:
                    ok, c, t = verify_fingerprint(np_, artist, song, self.acoustid_key, self.config, self._circuit_breaker,
                                                  on_warn=self.events.on_warn, on_info=self.events.on_info, on_fingerprint_error=self.events.on_fingerprint_error)
                    time.sleep(0.35)
                    if ok:
                        result.url, result.matched_title = nu, cr.get(
                            "title") or ""
                        result.duration_seconds = int(cr.get("duration") or 0)
                        result.composite_score = cr.get("_composite_score", 0)
                        result.score_breakdown = cr.get("_score_breakdown", {})
                        return True, c, t, f"verified next candidate {c:.0%}"
            finally:
                if np_ and np_.exists():
                    np_.unlink(missing_ok=True)
            break
        return False, 0.0, None, "low confidence (no alternate match)"

    def _post_download_checks(self, f, artist, song, url, thumb, fmt, dur_s, state, state_lock, key, result, out):
        ok, actual, fail = check_duration(f, dur_s, artist, song, self.events)
        if fail:
            result.reason = fail
            self._persist(state, state_lock, key,
                          "failed", url, None, None, out)
            return result
        sil = 0.0
        if not self.no_silence_check:
            sil, exc, sf = check_silence(
                f, artist, song, self.config, self.events)
            result.silence_ratio = sil
            if sf:
                result.reason = sf
                self._persist(state, state_lock, key,
                              "failed", url, None, None, out)
                return result
        self.events.on_post_check_summary(artist, song, ok, actual, sil)
        mb, mb_e = enrich_musicbrainz(
            artist, song, self.musicbrainz, self.events)
        if not embed_and_verify(f, song, artist, url, thumb, fmt, mb, self.events):
            result.reason = "Metadata integrity check failed"
            self._persist(state, state_lock, key,
                          "failed", url, None, None, out)
            return result
        md5 = compute_md5(f)
        sz = f.stat().st_size
        result.status = "downloaded"
        result.file_path = f
        result.file_size_bytes = sz
        result.md5 = md5
        result.musicbrainz_enriched = mb_e
        result.album = mb.get("album") if mb else None
        result.year = mb.get("year") if mb else None
        result.genre = mb.get("genre") if mb else None
        result.silence_ratio = sil
        result.duration_verified = ok
        self._persist(state, state_lock, key,
                      "downloaded", url, str(f), md5, out)
        return result

    @staticmethod
    def _iter_entries(data):
        entries = data.get("entries")
        if entries:
            for e in entries:
                if e:
                    yield from MusicDownloader._iter_entries(e)
            return
        yield data

    @staticmethod
    def _persist(state, lock, key, status, url, file_path, md5, out, fingerprint_verified=False, preserve_timestamp=False):
        from datetime import datetime, timezone
        with lock:
            dl = state.setdefault("downloads", {})
            ex = dl.get(key)
            ts = ex.get("timestamp") if (
                preserve_timestamp and ex and "timestamp" in ex) else datetime.now(timezone.utc).isoformat()
            dl[key] = {"status": status, "url": url, "file_path": file_path,
                       "md5": md5, "fingerprint_verified": fingerprint_verified, "timestamp": ts}
            save_state(state, out)
