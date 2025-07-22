import os
import subprocess
from rich.console import Console
from core._1_ytdlp import find_video_files
from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import normalize_audio_volume

console = Console()

DUB_VIDEO_OUTPUT = "output/output_dub.mp4"
DUB_AUDIO_INPUT = 'output/dub.mp3'
SUBTITLED_VIDEO_INPUT = "output/output_sub.mp4"

def merge_video_audio():
    """Merges the dubbed audio with the video based on user settings."""
    
    video_output_mode = load_key("video_output_mode")
    burn_subtitles = load_key("burn_subtitles")

    # If no video output is needed, just skip.
    if video_output_mode == "none":
        rprint("[bold yellow]Skipping video generation as per 'video_output_mode' setting.[/bold yellow]")
        return

    # --- Determine the video source ---
    if burn_subtitles and os.path.exists(SUBTITLED_VIDEO_INPUT):
        video_source = SUBTITLED_VIDEO_INPUT
        rprint(f"[bold green]Using subtitled video as source: {video_source}[/bold green]")
    else:
        video_source = find_video_files()
        if not video_source:
            rprint("[bold red]Error: No source video file found for merging.[/bold red]")
            return
        rprint(f"[bold green]Using original video as source: {video_source}[/bold green]")

    # --- Normalize the dubbed audio ---
    normalized_dub_audio = 'output/normalized_dub.wav'
    if not os.path.exists(DUB_AUDIO_INPUT):
        rprint(f"[bold red]Error: Dubbed audio file not found: {DUB_AUDIO_INPUT}[/bold red]")
        return
    normalize_audio_volume(DUB_AUDIO_INPUT, normalized_dub_audio)

    # --- Build and execute ffmpeg command ---
    cmd = [
        'ffmpeg', '-y', '-i', video_source, '-i', normalized_dub_audio,
    ]

    # Video stream handling: copy without re-encoding
    cmd.extend(['-c:v', 'copy'])

    # Audio stream handling based on the selected mode
    if video_output_mode == "replace_audio":
        # Map video stream from first input and audio stream from second input
        cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        rprint("[bold green]Mode: Replacing original audio with dubbed audio.[/bold green]")

    elif video_output_mode == "add_audio_track":
        # Map video stream and all original audio streams, plus the new dubbed audio stream
        cmd.extend(['-map', '0', '-map', '-0:a', '-map', '0:a?', '-map', '1:a:0'])
        cmd.extend([
            '-c:a', 'aac', '-b:a', '192k', # For the new dubbed track
            '-metadata:s:a:0', 'language=eng', # Example: Tag original audio as English
            '-metadata:s:a:1', 'language=tra', # Example: Tag new audio as Translated
            '-disposition:a:1', 'default' # Make the new track the default
        ])
        rprint("[bold green]Mode: Adding dubbed audio as a new, default track.[/bold green]")

    cmd.append(DUB_VIDEO_OUTPUT)

    try:
        subprocess.run(cmd, check=True)
        rprint(f"[bold green]Successfully generated final video: {DUB_VIDEO_OUTPUT}[/bold green]")
    except subprocess.CalledProcessError as e:
        rprint(f"[bold red]Error during ffmpeg execution: {e}[/bold red]")

if __name__ == '__main__':
    merge_video_audio()