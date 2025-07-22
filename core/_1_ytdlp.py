import os,sys
import glob
import re
import subprocess
from functools import partial
from core.utils import *

def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.strip('. ')
    return filename if filename else 'media'

def update_ytdlp():
    from yt_dlp import YoutubeDL
    return YoutubeDL

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    os.makedirs(save_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best' if resolution == 'best' else f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'proxy': load_key("proxy.http"),
        'noplaylist': True,
        'writethumbnail': True,
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
    }

    cookies_path = load_key("youtube.cookies_path")
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)

    YoutubeDL = update_ytdlp()
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

def find_media_file(media_type=None, save_path='output'):
    """
    Finds the single source media file in the specified path, intelligently ignoring
    intermediate outputs. If media_type is specified ('video' or 'audio'), it
    will only search for that type. Otherwise, it will search for any media.
    Returns a single file path string.
    """
    video_formats = load_key("allowed_video_formats")
    audio_formats = load_key("allowed_audio_formats")
    
    formats_to_check = []
    if media_type == 'video':
        formats_to_check = video_formats
    elif media_type == 'audio':
        formats_to_check = audio_formats
    elif media_type is None:
        formats_to_check = video_formats + audio_formats
    else:
        raise ValueError("Invalid media_type specified. Must be 'video', 'audio', or None.")

    all_files_in_path = glob.glob(os.path.join(save_path, "*"))
    
    # Filter for the correct media type(s)
    candidates = [f for f in all_files_in_path if os.path.splitext(f)[1][1:].lower() in formats_to_check]
    
    # The original, robust filtering logic: ignore known intermediate files.
    source_candidates = [f for f in candidates if not os.path.basename(f).startswith("output_")]

    if sys.platform.startswith('win'):
        source_candidates = [f.replace("\\", "/") for f in source_candidates]

    if len(source_candidates) == 0:
        type_str = media_type if media_type else "media"
        raise ValueError(f"No source {type_str} files found in '{save_path}' after filtering.")
    if len(source_candidates) > 1:
        type_str = media_type if media_type else "media"
        raise ValueError(f"Multiple source {type_str} files found after filtering: {source_candidates}. Please ensure only one original media file exists.")
    
    return source_candidates[0]

# Alias for backward compatibility.
find_video_files = partial(find_media_file, media_type="video")

if __name__ == '__main__':
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")