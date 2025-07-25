from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.uvr import uvr_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_media_file
from core.utils.models import *

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. At this early stage, there's only one media file in the 'output' directory.
    # We call find_media_file without arguments to find it, regardless of type.
    rprint("[cyan]🔍 Finding unique source media file...[/cyan]")
    media_path = find_media_file()
    
    # 2. Standardize the media to the raw audio file for ASR.
    # This function handles both video (extracts audio) and audio (standardizes format).
    rprint(f"[cyan]🎧 Standardizing '{media_path}' for ASR...[/cyan]")
    convert_video_to_audio(media_path)

    # 2. Vocal separation
    separation_method = load_key("vocal_separation.method")
    if separation_method == "demucs":
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    elif separation_method == "uvr":
        uvr_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "fastapi":
        from core.asr_backend.whisperX_FastAPI import transcribe_audio_fastapi as ts
        rprint("[cyan]🎤 Transcribing audio with whisperX FastAPI...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")

    for start, end in segments:
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        all_results.append(result)
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()