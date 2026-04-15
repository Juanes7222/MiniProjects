"""
ytdl_core — Python library for batch audio downloading with verification.

Quick start
-----------
from ytdl_core import MusicDownloader, Config
from ytdl_core.events import DownloaderEvents
from pathlib import Path

# Minimal usage
dl = MusicDownloader(quality="320", musicbrainz=True)
result = dl.download("Radiohead", "Creep", Path("./music"))
print(result.status, result.file_path)

# Batch with callbacks
class MyEvents(DownloaderEvents):
    def on_result(self, r):
        print(f"{r.artist} -- {r.song}: {r.status} (score={r.composite_score})")

dl = MusicDownloader(
    acoustid_key="YOUR_KEY",
    quality="320",
    musicbrainz=True,
    events=MyEvents(),
)
results = dl.download_batch(
    songs={"Radiohead": ["Creep", "Karma Police"], "Portishead": ["Glory Box"]},
    output_dir=Path("./music"),
    report_formats=["json", "csv"],
)
"""

from .config import Config
from .core import MusicDownloader
from .events import DownloaderEvents
from .result import DownloadResult

__all__ = [
    "MusicDownloader",
    "Config",
    "DownloadResult",
    "DownloaderEvents",
]
__version__ = "2.0.0"