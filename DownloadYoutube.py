import os
import yt_dlp    

def descargar_audio_mp3_2(url, output_path="descargas"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])



videos = ["https://youtu.be/CSd-I2LUGv0", ]

for video in videos:
    descargar_audio_mp3_2(video, r"C:\Users\juanb\Downloads\Musica")