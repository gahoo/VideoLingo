import os
import io
import json
import time
import requests
import librosa
import soundfile as sf
from rich import print as rprint
from core.utils import *
from core.utils.models import *

OUTPUT_LOG_DIR = "output/log"

def transcribe_audio_fastapi(raw_audio_path: str, vocal_audio_path: str, start: float = None, end: float = None):
    """
    Transcribes audio using a WhisperX FastAPI server.
    This involves a two-step process:
    1. Submit the audio file and get a task identifier.
    2. Use the identifier to fetch the transcription result.
    Reads configuration from config.yaml under 'whisper'.
    """
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    vocal_basename = os.path.basename(vocal_audio_path).rsplit('.', 1)[0]
    LOG_FILE = f"{OUTPUT_LOG_DIR}/whisperx_fastapi_{vocal_basename}_{start}_{end}.json"

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
        
    # Load configs from config.yaml, with defaults from the whisperx script
    lang = load_key("whisper.language")
    base_url = load_key("whisper.whisperx_fastapi_url") # e.g., http://127.0.0.1:9000
    model = load_key("whisper.model")
    batch_size = load_key("whisper.batch_size")
    chunk_size = load_key("whisper.chunk_size")
    diarization = load_key("whisper.diarization")

    # Prepare audio slice
    y, sr = librosa.load(vocal_audio_path, sr=16000)
    audio_duration = len(y) / sr
    
    if start is None or end is None:
        start = 0
        end = audio_duration
        
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_slice = y[start_sample:end_sample]
    
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, y_slice, sr, format='WAV', subtype='PCM_16')
    audio_buffer.seek(0)
    
    files = {'file': ('audio_slice.wav', audio_buffer, 'audio/wav')}
    
    # Step 1: Submit transcription task
    post_url = f"{base_url}/speech-to-text"
    post_params = {
        "language": lang,
        "task": "transcribe",
        "model": model,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "alignment": "true",
        "diarization": diarization,
        "return_char_alignments": "true",
        "is_async": "false" # As per whisperx script, this should block until task is processable
    }
    
    start_time = time.time()
    rprint(f"[cyan]🎤 Submitting transcription task for audio slice ({start:.2f}s - {end:.2f}s) with language: <{lang}>...[/cyan]")
    
    try:
        response = requests.post(post_url, params=post_params, files=files)
        response.raise_for_status()
        task_response_json = response.json()
        identifier = task_response_json.get("identifier")
        if not identifier:
            rprint(f"[red]Error: Could not get task identifier from response: {task_response_json}[/red]")
            return {"segments": [], "text": ""}
    except requests.exceptions.RequestException as e:
        rprint(f"[red]Error during transcription task submission to {post_url}: {e}[/red]")
        rprint("[red]Please ensure your whisperX FastAPI server is running and the URL is correctly configured in config.yaml (whisper.whisperx_fastapi_url).[/red]")
        return {"segments": [], "text": ""}

    rprint(f"[cyan]Got task identifier: {identifier}. Fetching results...[/cyan]")

    # Step 2: Fetch transcription result
    get_url = f"{base_url}/task/{identifier}/json"
    get_params = {
        "karaoke_style": "false",
        "highlight_words": "false"
    }

    try:
        # Adding a small delay just in case, though is_async=false should make it synchronous
        time.sleep(1) 
        response = requests.get(get_url, params=get_params)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        rprint(f"[red]Error fetching transcription result from {get_url}: {e}[/red]")
        return {"segments": [], "text": ""}

    # Adjust timestamps to be absolute
    if start is not None and 'segments' in response_json:
        for segment in response_json['segments']:
            if 'start' in segment:
                segment['start'] += start
            if 'end' in segment:
                segment['end'] += start
            for word in segment.get('words', []):
                if 'start' in word:
                    word['start'] += start
                if 'end' in word:
                    word['end'] += start
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=4, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    rprint(f"[green]✓ Transcription completed in {elapsed_time:.2f} seconds[/green]")
    return response_json

if __name__ == "__main__":  
    # This is a module and not intended to be run directly.
    pass