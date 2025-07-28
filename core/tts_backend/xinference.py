# core/tts_backend/xinference.py

from pathlib import Path
from xinference_client import RESTfulClient as Client
from core.utils import *

@except_handler("Failed to generate audio using Xinference TTS")
def xinference_tts(text, save_as, number, task_df):
    """
    Xinference TTS (Text-to-Speech) interface.
    Supports preset voice and voice cloning.
    """
    base_url = load_key("xinference_tts.base_url") or 'http://127.0.0.1:9997'
    model_uid = load_key("xinference_tts.model_uid")
    
    if not model_uid or model_uid == 'your_model_uid':
        print("Error: 'model_uid' not configured for xinference_tts in config.yaml. You must launch a model in Xinference and provide its UID.")
        return False

    mode = load_key("xinference_tts.mode") or 'preset' # 'preset' or 'clone'
    
    try:
        client = Client(base_url)
        model = client.get_model(model_uid)
    except Exception as e:
        print(f"Error connecting to Xinference or getting model: {e}")
        return False

    save_path = Path(save_as)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == 'clone':
        # For clone mode, we need reference audio.
        # This part is similar to sf_cosyvoice2.py
        prompt_text = task_df.loc[task_df['number'] == number, 'origin'].values[0]
        
        current_dir = Path.cwd()
        ref_audio_path = current_dir / f"output/audio/refers/{number}.wav"
        
        if not ref_audio_path.exists():
            ref_audio_path = current_dir / "output/audio/refers/1.wav"
            if not ref_audio_path.exists():
                try:
                    from core._9_refer_audio import extract_refer_audio_main
                    print(f"Reference audio file not found, trying to extract: {ref_audio_path}")
                    extract_refer_audio_main()
                except Exception as e:
                    print(f"Failed to extract reference audio: {str(e)}")
                    raise

        with open(ref_audio_path, "rb") as f:
            prompt_speech_bytes = f.read()

        # Based on docs for FishSpeech and CosyVoice with xinference_client
        # The `speech` method returns bytes
        speech_bytes = model.speech(
            text,
            prompt_speech=prompt_speech_bytes,
            prompt_text=prompt_text
        )
        with open(save_path, "wb") as f:
            f.write(speech_bytes)

    else: # preset mode
        voice = load_key("xinference_tts.voice") or 'default' # For ChatTTS or CosyVoice-SFT
        
        # The doc shows model.speech(voice=..., input=...)
        # The `speech` method returns bytes
        resp_bytes = model.speech(
            input=text,
            voice=voice
        )
        with open(save_path, "wb") as f:
            f.write(resp_bytes)

    print(f"Audio successfully saved to: {save_path}")
    return True

if __name__ == '__main__':
    print("This script is intended to be imported, not run directly.")
    print("To test, call xinference_tts with appropriate arguments and a running Xinference service.")
