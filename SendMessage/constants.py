from dotenv import load_dotenv
from pathlib import Path
import os
import json

load_dotenv()

LINKS_CACHE_FILE = Path(".sermon_links_cache.json")
AUDIO_EXTENSIONS       = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
SPOTIFY_EPISODE_LIMIT  = 50
WHATSAPP_LOAD_TIMEOUT  = 60
ARROW_EMOJI            = "\U0001f449\U0001f3fb"

YOUTUBE_API_KEY: str        = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_CHANNEL_ID: str     = os.getenv("YOUTUBE_CHANNEL_ID", "")
SPOTIFY_CLIENT_ID: str      = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET: str  = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_SHOW_ID: str        = os.getenv("SPOTIFY_SHOW_ID", "")
DEFAULT_CHROME_PROFILE: str = os.getenv(
    "CHROME_PROFILE_DIR", str(Path.home() / ".whatsapp_chrome_profile")
)
DEFAULT_WHATSAPP_CONTACT: str = os.getenv("WHATSAPP_CONTACT", "")

if any(var == "" for var in [YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_SHOW_ID]):
    raise ValueError("One or more required environment variables are missing. Please check your .env file.")