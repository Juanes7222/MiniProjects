import argparse
import pandas as pd
import yt_dlp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from pathlib import Path

console = Console()

class RichProgressLogger:
    def __init__(self, progress, task_id):
        self.progress = progress
        self.task_id = task_id
        
    def __call__(self, d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            if total:
                self.progress.update(self.task_id, completed=downloaded, total=total)
        elif d['status'] == 'finished':
            self.progress.update(self.task_id, completed=100, total=100)

def download_audio_mp3(url, output_path="downloads"):
    """Download audio from URL and convert to MP3"""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Downloading audio...", total=100)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'progress_hooks': [RichProgressLogger(progress, task)],
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                console.print(f"[green]✓[/green] Audio downloaded: [bold]{title}[/bold]")
        except Exception as e:
            console.print(f"[red]✗[/red] Error downloading audio: {str(e)}")

def download_video_mp4(url, output_path="downloads"):
    """Download video from URL in MP4 format"""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Downloading video...", total=100)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'progress_hooks': [RichProgressLogger(progress, task)],
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                console.print(f"[green]✓[/green] Video downloaded: [bold]{title}[/bold]")
        except Exception as e:
            console.print(f"[red]✗[/red] Error downloading video: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Download videos or audio from YouTube and other platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -u "https://youtube.com/watch?v=..." -o ~/Downloads -t audio
  %(prog)s -f urls.csv -o ./videos -t video
  %(prog)s -u "https://youtu.be/..." "https://youtu.be/..." -t video
        """
    )
    
    parser.add_argument(
        '-u', '--urls',
        nargs='+',
        help='One or more URLs to download'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='CSV file containing URLs (column: "YouTube URL")'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='downloads',
        help='Output directory for downloads (default: downloads)'
    )
    
    parser.add_argument(
        '-t', '--type',
        choices=['audio', 'video'],
        default='video',
        help='Download type: audio (MP3) or video (MP4) (default: video)'
    )
    
    args = parser.parse_args()
    
    # Collect URLs
    urls = []
    if args.urls:
        urls.extend(args.urls)
    
    if args.file:
        try:
            df = pd.read_csv(args.file)
            if "YouTube URL" in df.columns:
                urls.extend(df["YouTube URL"].tolist())
            else:
                console.print("[red]Error:[/red] CSV file must have a 'YouTube URL' column")
                return
        except Exception as e:
            console.print(f"[red]Error reading CSV file:[/red] {str(e)}")
            return
    
    if not urls:
        parser.print_help()
        console.print("\n[yellow]Warning:[/yellow] No URLs provided. Use -u or -f option.")
        return
    
    # Display summary
    table = Table(title="Download Summary", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Type", args.type.upper())
    table.add_row("Output Path", args.output)
    table.add_row("Total URLs", str(len(urls)))
    
    console.print(Panel(table, border_style="blue"))
    console.print()
    
    # Download
    download_func = download_audio_mp3 if args.type == 'audio' else download_video_mp4
    
    for i, url in enumerate(urls, 1):
        console.print(f"\n[bold yellow]Processing {i}/{len(urls)}[/bold yellow]")
        download_func(url, args.output)
    
    console.print(f"\n[bold green]✓ All downloads completed![/bold green]")

if __name__ == "__main__":
    main()