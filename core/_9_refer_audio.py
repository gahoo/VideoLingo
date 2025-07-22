import os
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from core.utils import *
from core.utils.models import *
import pandas as pd
import soundfile as sf
console = Console()
from core.asr_backend.demucs_vl import demucs_audio
from core.utils.models import *

def time_to_samples(time_str, sr):
    """Unified time conversion function"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',') if ',' in s else (s, '0')
    seconds = int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
    return int(seconds * sr)

def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Extracts an audio segment, ensuring it is at least 3 seconds long."""
    min_duration_seconds = 3
    min_duration_samples = int(min_duration_seconds * sr)

    # Get initial start and end samples
    start_samples = time_to_samples(start_time, sr)
    end_samples = time_to_samples(end_time, sr)

    duration_samples = end_samples - start_samples

    # If the audio is shorter than the minimum duration, extend it
    if duration_samples < min_duration_samples:
        shortfall = min_duration_samples - duration_samples
        # Extend symmetrically from both ends
        half_extension = shortfall // 2
        
        extended_start = start_samples - half_extension
        extended_end = end_samples + half_extension + (shortfall % 2) # Add remainder to one side

        # Boundary checks to ensure we don't go out of the audio's limits
        final_start = max(0, extended_start)
        final_end = min(len(audio_data), extended_end)

        # If we were clipped at the start, try to compensate by extending more at the end
        if extended_start < 0:
            final_end = min(len(audio_data), final_end - extended_start) # extended_start is negative
        
        # If we were clipped at the end, try to compensate by extending more at the start
        if extended_end > len(audio_data):
            final_start = max(0, final_start - (extended_end - len(audio_data)))
    else:
        final_start = start_samples
        final_end = end_samples

    sf.write(out_file, audio_data[final_start:final_end], sr)

def extract_refer_audio_main():
    demucs_audio() #!!! in case demucs not run
    if os.path.exists(os.path.join(_AUDIO_SEGS_DIR, '1.wav')):
        rprint(Panel("Audio segments already exist, skipping extraction", title="Info", border_style="blue"))
        return

    # Create output directory
    os.makedirs(_AUDIO_REFERS_DIR, exist_ok=True)
    
    # Read task file and audio data
    df = pd.read_excel(_8_1_AUDIO_TASK)
    data, sr = sf.read(_VOCAL_AUDIO_FILE)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Extracting audio segments...", total=len(df))
        
        for _, row in df.iterrows():
            out_file = os.path.join(_AUDIO_REFERS_DIR, f"{row['number']}.wav")
            extract_audio(data, sr, row['start_time'], row['end_time'], out_file)
            progress.update(task, advance=1)
            
    rprint(Panel(f"Audio segments saved to {_AUDIO_REFERS_DIR}", title="Success", border_style="green"))

if __name__ == "__main__":
    extract_refer_audio_main()