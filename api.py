

import os
import shutil
import uuid
import yaml
import logging
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup tasks directory
TASKS_DIR = Path("tasks").resolve()
TASKS_DIR.mkdir(exist_ok=True)

# --- Mock Core Imports ---
try:
    from core.utils.onekeycleanup import cleanup
    from core import (
        _1_ytdlp, _2_asr, _3_1_split_nlp, _3_2_split_meaning,
        _4_1_summarize, _4_2_translate, _5_split_sub, _6_gen_sub,
        _7_sub_into_vid, _8_1_audio_task, _8_2_dub_chunks, _9_refer_audio,
        _10_gen_audio, _11_merge_audio, _12_dub_to_vid
    )
except ImportError:
    logging.warning("Could not import core modules. Using mock functions.")
    class MockCoreModule:
        def __getattr__(self, name):
            def mock_func(*args, **kwargs):
                logging.info(f"Mock call to {self.__class__.__name__}.{name}")
                if name == 'find_video_files': return 'mock_video.mp4'
            return mock_func
    _1_ytdlp, _2_asr, _3_1_split_nlp, _3_2_split_meaning, _4_1_summarize, _4_2_translate, \
    _5_split_sub, _6_gen_sub, _7_sub_into_vid, _8_1_audio_task, _8_2_dub_chunks, _9_refer_audio, \
    _10_gen_audio, _11_merge_audio, _12_dub_to_vid = [MockCoreModule() for _ in range(15)]
    def cleanup(save_dir=None): pass

# --- Pydantic Models ---
class ProcessOptions(BaseModel):
    target_language: str = Field('简体中文')
    dubbing: bool = Field(False)
    burn_subtitles: bool = Field(True)
    demucs: bool = Field(True)
    whisper_model: str = Field('large-v3', alias="whisper.model")
    ytb_resolution: str = Field('1080')
    video_output_mode: Literal["replace_audio", "add_audio_track", "none"] = Field(
        "replace_audio", 
        description="How to handle the final video output when dubbing."
    )
    cache: bool = Field(True, description="Whether to use a cached result if available.")

    class Config:
        allow_population_by_field_name = True

class TaskResult(BaseModel):
    task_id: str
    status: Literal["completed", "failed", "cached"]
    message: Optional[str] = None
    completed_steps: Optional[List[str]] = None
    available_files: Optional[Dict[str, str]] = None
    error: Optional[str] = None

# --- Helper Functions ---
@contextmanager
def execution_context(workdir: str):
    original_dir = os.getcwd()
    try:
        os.chdir(workdir)
        yield
    finally:
        os.chdir(original_dir)

def find_file(directory: Path, patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        files = list(directory.rglob(pattern))
        if files:
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return files[0]
    return None

def find_available_files(history_dir: Path) -> Dict[str, str]:
    """Finds all generated files and maps them to a file type with robust logic."""
    file_map = {}

    # 1. Find all potential files first
    all_mp4s = list(history_dir.rglob("*.mp4"))
    all_srts = list(history_dir.rglob("*.srt"))
    all_audios = list(history_dir.rglob("*.mp3")) + list(history_dir.rglob("*.wav"))

    # 2. Categorize video files by excluding processed ones from originals
    processed_video_names = []
    for video_path in all_mp4s:
        if "_dub.mp4" in video_path.name:
            file_map["dubbed_video"] = video_path.name
            processed_video_names.append(video_path.name)
        elif "_sub.mp4" in video_path.name:
            file_map["subtitled_video"] = video_path.name
            processed_video_names.append(video_path.name)

    # Any mp4 that is not a processed video is an original candidate
    original_candidates = [p for p in all_mp4s if p.name not in processed_video_names]
    if original_candidates:
        file_map["original_video"] = original_candidates[0].name

    # 3. Categorize SRT files
    for srt_path in all_srts:
        if srt_path.name == "src.srt":
            file_map["transcript_srt"] = srt_path.name
        elif srt_path.name == "trans.srt":
            file_map["translated_srt"] = srt_path.name
            
    # 4. Categorize Audio files
    for audio_path in all_audios:
        if audio_path.stem == "dub": # Check for dub.mp3 or dub.wav
            file_map["dubbed_audio"] = audio_path.name

    return file_map

def generate_task_id(url: Optional[str], file: Optional[UploadFile], options: ProcessOptions) -> str:
    hasher = hashlib.sha256()
    if url:
        hasher.update(url.encode('utf-8'))
    elif file:
        while chunk := file.file.read(8192):
            hasher.update(chunk)
        file.file.seek(0)
    # Add options to the hash for true idempotency, excluding the cache parameter itself.
    options_dict = options.dict()
    options_dict.pop('cache', None) # Exclude cache from the hash
    sorted_options_str = str(sorted(options_dict.items()))
    hasher.update(sorted_options_str.encode('utf-8'))
    return hasher.hexdigest()

def save_config(config, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

# --- FastAPI Application ---
app = FastAPI(
    title="VideoLingo API (Sync)",
    description="A synchronous, idempotent API for video processing with downloadable artifacts.",
    version="1.2.0"
)

def get_media_type_from_filename(filename: str, config: dict) -> Literal['video', 'audio']:
    """Determines if a file is video or audio based on its extension."""
    extension = os.path.splitext(filename)[1][1:].lower()
    if extension in config.get("allowed_video_formats", []):
        return 'video'
    elif extension in config.get("allowed_audio_formats", []):
        return 'audio'
    raise ValueError(f"Unsupported file type for file: {filename}")

@app.post("/process/", response_model=TaskResult)
async def process_video_endpoint(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    options: ProcessOptions = Depends()
):
    if not url and not file:
        raise HTTPException(status_code=400, detail="Either 'url' or 'file' must be provided.")

    task_id = generate_task_id(url, file, options)
    workdir = TASKS_DIR / task_id
    history_dir = workdir / 'history'

    # --- Idempotency and Cache Check ---
    if options.cache and history_dir.exists():
        available_files = find_available_files(history_dir)
        if available_files:
            logging.info(f"Task {task_id} already completed. Returning cached result.")
            return TaskResult(
                task_id=task_id,
                status="cached",
                message="Task was already completed. Found cached results.",
                available_files=available_files
            )

    # --- Resume/Retry/Cache=False Logic ---
    if workdir.exists():
        logging.info(f"Clearing previous directory for task {task_id} before new run (cache={options.cache}).")
        shutil.rmtree(workdir)
    
    workdir.mkdir(parents=True)
    (workdir / 'output').mkdir()
    (workdir / 'input').mkdir()

    completed_steps = []
    try:
        # --- Setup and Config ---
        input_filename = None
        if file:
            input_filename = file.filename
            with open(workdir / "input" / input_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        with open('config.yaml', 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
        config['model_dir'] = str(Path.cwd() / config.get('model_dir', '_model_cache').lstrip('./\\'))
        opts_dict = options.dict(by_alias=True, exclude_unset=True)
        for k, v in opts_dict.items():
            p, c = k.split('.') if '.' in k else (k, None)
            if c: config.setdefault(p, {})[c] = v
            else: config[k] = v
        save_config(config, workdir / 'config.yaml')
        if os.path.exists('custom_terms.xlsx'): shutil.copy('custom_terms.xlsx', workdir)

        # --- Execute processing pipeline ---
        with execution_context(workdir):
            task_info = {}

            def process_input_file_step():
                if url:
                    _1_ytdlp.download_video_ytdlp(url, resolution=config.get('ytb_resolution', '1080'))
                    # After download, we need to find it to determine type
                    found_file = _1_ytdlp.find_media_file()
                    task_info['media_type'] = get_media_type_from_filename(found_file, config)
                else:
                    shutil.copy(workdir.parent / task_id / 'input' / input_filename, 'output')
                    task_info['media_type'] = get_media_type_from_filename(input_filename, config)

            # 1. Execute the first step to identify the media type
            logging.info(f"Task {task_id}: Executing step: Processing input file")
            process_input_file_step()
            completed_steps.append("Processing input file")

            # 2. Dynamically build the rest of the pipeline
            pipeline = [
                ("Transcribing", _2_asr.transcribe),
                ("Splitting sentences", lambda: (_3_1_split_nlp.split_by_spacy(), _3_2_split_meaning.split_sentences_by_meaning())),
                ("Translating", lambda: (_4_1_summarize.get_summary(), _4_2_translate.translate_all()))
            ]

            if config.get('dubbing', False):
                pipeline.append(("Aligning subtitles for dubbing", lambda: (_5_split_sub.split_for_sub_main(), _6_gen_sub.align_timestamp_main())))
                pipeline.extend([
                    ("Generating audio tasks", lambda: (_8_1_audio_task.gen_audio_task_main(), _8_2_dub_chunks.gen_dub_chunks())),
                    ("Extracting reference audio", _9_refer_audio.extract_refer_audio_main),
                    ("Generating audio", _10_gen_audio.gen_audio),
                    ("Merging full audio", _11_merge_audio.merge_full_audio),
                ])

            if task_info.get('media_type') == 'video':
                if not config.get('dubbing', False):
                    pipeline.append(("Aligning subtitles for burning", lambda: (_5_split_sub.split_for_sub_main(), _6_gen_sub.align_timestamp_main())))
                if config.get('burn_subtitles', True):
                    pipeline.append(("Merging subtitles", _7_sub_into_vid.merge_subtitles_to_video))
                if config.get('dubbing', False):
                    pipeline.append(("Merging dubbing to video", _12_dub_to_vid.merge_video_audio))

            # 3. Execute the constructed pipeline
            for name, step_func in pipeline:
                logging.info(f"Task {task_id}: Executing step: {name}")
                step_func()
                completed_steps.append(name)
            
            temp_dirs_to_clean = ['output/audio/refers', 'output/audio/segs', 'output/audio/tmp']
            for temp_dir in temp_dirs_to_clean:
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
            cleanup()


        # --- Finalize and Return Result ---
        available_files = find_available_files(history_dir)
        return TaskResult(
            task_id=task_id,
            status="completed",
            completed_steps=completed_steps,
            available_files=available_files
        )

    except Exception as e:
        logging.error(f"An error occurred during processing for task {task_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=TaskResult(
                task_id=task_id,
                status="failed",
                completed_steps=completed_steps,
                error=str(e)
            ).dict()
        )

@app.get("/data/{task_id}")
async def download_file_endpoint(
    task_id: str,
    file_type: Optional[Literal[
        "subtitled_video", "dubbed_video", "original_video", 
        "transcript_srt", "translated_srt", "dubbed_audio"
    ]] = Query(None, description="The type of file to download. If omitted, returns a list of available files.")
):
    workdir = TASKS_DIR / task_id
    history_dir = workdir / 'history'
    if not workdir.exists() or not history_dir.exists():
        raise HTTPException(status_code=404, detail="Task ID not found.")

    available_files = find_available_files(history_dir)

    # If file_type is not provided, return the list of available files.
    if not file_type:
        return JSONResponse(content={
            "task_id": task_id,
            "available_files": available_files
        })

    # If file_type is provided, proceed with file download.
    file_name = available_files.get(file_type)

    if not file_name:
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not available for this task.")
    
    # Find the file again to get its full path for serving
    file_path = find_file(history_dir, [f"**/{file_name}"])
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on server, though it was expected.")

    return FileResponse(path=file_path, filename=file_name)

if __name__ == "__main__":
    import uvicorn
    print("To run the API server, use the command:")
    print("uvicorn api_v3:app --reload")

