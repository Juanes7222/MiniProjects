#!/usr/bin/env python3
"""
Sermon WhatsApp message generator and sender.

Parses sermon audio filenames, searches YouTube and Spotify for links,
builds a formatted WhatsApp message, and optionally sends it via WhatsApp Web.

Filename format: numero_predica_titulo..._predicador_fecha.ext
"""

import argparse
import sys
import re
import json
import pyperclip
import requests
from difflib import SequenceMatcher
from pathlib import Path
from models import SermonMetadata, SermonLinks
try:
    from whatsapp import WhatsAppWebSender
except ImportError:
    WhatsAppWebSender = None

from constants import (AUDIO_EXTENSIONS, YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID,
                      ARROW_EMOJI, DEFAULT_CHROME_PROFILE, DEFAULT_WHATSAPP_CONTACT,
                       LINKS_CACHE_FILE)
from whatsapp import WhatsAppWebSender



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predica_whatsapp",
        description=(
            "Generate and send WhatsApp messages for sermon recordings. "
            "Searches YouTube and Spotify automatically, then falls back to manual input."
        ),
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file",
        metavar="PATH",
        type=Path,
        help="Single audio file to process.",
    )
    source.add_argument(
        "--folder",
        metavar="PATH",
        type=Path,
        help="Folder of audio files; all audio files inside will b  e processed in order.",
    )

    parser.add_argument(
        "--send",
        action="store_true",
        help="Send via WhatsApp Web after reviewing each message. Requires --contact.",
    )
    parser.add_argument(
        "--contact",
        metavar="NAME",
        default=DEFAULT_WHATSAPP_CONTACT,
        help="Exact WhatsApp contact or group name to send to (required with --send).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        metavar="0.0-1.0",
        help="Minimum similarity score to accept a search result (default: 0.55).",
    )
    parser.add_argument(
        "--chrome-profile",
        metavar="DIR",
        default=DEFAULT_CHROME_PROFILE,
        dest="chrome_profile",
        help="Chrome user data directory used to persist the WhatsApp Web session.",
    )

    return parser


def parse_audio_filename(filepath: Path) -> SermonMetadata:
    """
    Splits the filename stem into sermon metadata.
    Format: numero_predica_titulo..._predicador_fecha
    The literal segment 'predica' after the number is treated as a fixed separator.
    """
    PATTERN = r"^(?P<number>\d+)_(?P<title>[^_]+)_(?P<preacher>[^_]+)_(?P<date>\d{2}-\d{2}-\d{4})(?:_(?P<description>[^.]+))?(?P<extension>\.\w+)\]?$"
    match = re.match(PATTERN, filepath.name)

    if not match:
        raise ValueError(
            f"Cannot parse '{filepath.name}'. "
            "Expected format: numero-predica_titulo_predicador_fecha"
        )
        
    number      = match.group("number")
    date        = match.group("date")
    preacher    = match.group("preacher")
    title_parts = match.group("title")

    if not title_parts:
        raise ValueError(f"Could not extract a title from '{filepath.name}'.")

    return SermonMetadata(
        number=number,
        title=title_parts.title(),
        preacher=preacher.title(),
        date=date,
    )


def find_audio_files(folder: Path) -> list[Path]:
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )


def _string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def search_youtube_video(title: str, preacher: str, threshold: float) -> str | None:
    """
    Queries the configured channel for a video whose title (before the first '|')
    matches the sermon title. Returns a short youtu.be URL or None.
    """
    
    cache = _lookup_cached_link(title, "youtube")
    if cache:
        print(f"  [YouTube] Found in local cache.")
        return cache
    
    if not YOUTUBE_API_KEY or not YOUTUBE_CHANNEL_ID:
        return None

    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "key":        YOUTUBE_API_KEY,
                "channelId":  YOUTUBE_CHANNEL_ID,
                "q":          f"{title} {preacher}",
                "part":       "snippet",
                "type":       "video",
                "maxResults": 5,
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [YouTube] Request error: {e}")
        return None

    best_id    = None
    best_score = 0.0

    for item in response.json().get("items", []):
        candidate = item["snippet"]["title"].split("|")[0].strip()
        score = _string_similarity(title, candidate)
        if score > best_score:
            best_score = score
            best_id    = item["id"]["videoId"]

    if best_score >= threshold and best_id:
        print(f"  [YouTube] Match found (score {best_score:.2f})")
        _save_to_links_cache(title, "youtube", f"https://youtu.be/{best_id}")
        return f"https://youtu.be/{best_id}"

    print(f"  [YouTube] No match above threshold (best: {best_score:.2f})")
    return None


def _load_links_cache() -> dict:
    if not LINKS_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(LINKS_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_to_links_cache(title: str, platform: str, url: str) -> None:
    cache = _load_links_cache()
    key   = title.lower().strip()
    if key not in cache:
        cache[key] = {}
    cache[key][platform] = url
    LINKS_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _lookup_cached_link(title: str, platform: str) -> str | None:
    cache = _load_links_cache()
    return cache.get(title.lower().strip(), {}).get(platform)


def search_spotify_episode(title: str) -> str | None:
    """
    Checks local cache first. Falls back to manual input and caches the result.
    Spotify API is unavailable without extended quota mode.
    """
    cached = _lookup_cached_link(title, "spotify")
    if cached:
        print(f"  [Spotify] Found in local cache.")
        return cached
    return None


def resolve_sermon_links(meta: SermonMetadata, threshold: float) -> SermonLinks:
    print(f"\nSearching: '{meta.title}' by {meta.preacher}")

    youtube_url = search_youtube_video(meta.title, meta.preacher, threshold)
    if not youtube_url:
        youtube_url = _prompt_for_missing_link("YouTube", meta.title)

    spotify_url = search_spotify_episode(meta.title)
    if not spotify_url:
        spotify_url = _prompt_for_missing_link("Spotify", meta.title)

    return SermonLinks(youtube=youtube_url, spotify=spotify_url)


def _prompt_for_missing_link(platform: str, title: str) -> str | None:
    raw = input(f"  Paste {platform} link manually (Enter to skip): ").strip()
    if raw:
        _save_to_links_cache(title, platform.lower(), raw)
        print(f"  Saved to local cache for future use.")
    return raw or None

def build_whatsapp_message(title: str, links: SermonLinks) -> str:
    lines = [f"*TEMA: {title.upper()}*", ""]

    if links.youtube:
        lines.append(f"{ARROW_EMOJI} *Video de la prédica:*  {links.youtube}")
    if links.spotify:
        lines.append(f"{ARROW_EMOJI} *Escucha esta prédica en Spotify:* {links.spotify}")

    return "\n".join(lines)

def attempt_whatsapp_send(
    contact: str,
    audio_path: Path,
    message: str,
    chrome_profile: str,
) -> None:
    if WhatsAppWebSender is None:
        print("Error: selenium is not installed. Run: pip install selenium")
        return

    sender = None
    try:
        sender = WhatsAppWebSender(chrome_profile)
        sender.send_sermon(contact, audio_path, message)
    except Exception as e:
        print(f"WhatsApp Web automation error: {e}")
    finally:
        if sender:
            sender.close()


def process_single_sermon(audio_path: Path, args: argparse.Namespace) -> None:
    try:
        meta = parse_audio_filename(audio_path)
    except ValueError as e:
        print(f"Skipping '{audio_path.name}': {e}")
        return

    links   = resolve_sermon_links(meta, args.threshold)
    message = build_whatsapp_message(meta.title, links)

    print("\n--- WhatsApp message preview ---")
    print(message)
    print("--------------------------------")

    pyperclip.copy(message)
    print("Message copied to clipboard.")

    if not args.send:
        return

    if not args.contact:
        print("Error: --contact is required when using --send.")
        return

    answer = input("Send this via WhatsApp Web? [y/N]: ").strip().lower()
    if answer == "y":
        attempt_whatsapp_send(args.contact, audio_path, message, args.chrome_profile)


def process_folder(folder: Path, args: argparse.Namespace) -> None:
    audio_files = find_audio_files(folder)

    if not audio_files:
        print(f"No audio files found in '{folder}'.")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio file(s) in '{folder}'.")

    for index, audio_path in enumerate(audio_files, start=1):
        print(f"\n{'=' * 50}")
        print(f"[{index}/{len(audio_files)}] {audio_path.name}")
        process_single_sermon(audio_path, args)


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    if args.file:
        if not args.file.is_file():
            parser.error(f"'{args.file}' is not a valid file.")
        process_single_sermon(args.file, args)

    elif args.folder:
        if not args.folder.is_dir():
            parser.error(f"'{args.folder}' is not a valid directory.")
        process_folder(args.folder, args)


if __name__ == "__main__":
    main()