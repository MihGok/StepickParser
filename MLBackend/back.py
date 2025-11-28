import os
import shutil
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Any, Dict, List
import requests
from faster_whisper import WhisperModel
import time
import json

# --- Конфигурация ---
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

MAX_CONCURRENT_INFERENCES = int(os.environ.get("MAX_CONCURRENT_INFERENCES", "1"))
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

executor = ThreadPoolExecutor(max_workers=2)
app = FastAPI(title="ML Service")

STORAGE_ROOT = os.environ.get("STORAGE_ROOT", "/tmp/ml_service_storage")
os.makedirs(STORAGE_ROOT, exist_ok=True)


class ModelHolder:
    model: Optional[WhisperModel] = None

model_holder = ModelHolder()

@app.on_event("startup")
async def startup_event():
    try:
        model_holder.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Whisper model loaded:", MODEL_SIZE, DEVICE)
    except Exception as e:
        print(f"Failed to init model on device {DEVICE}: {e}. Trying CPU fallback.")
        try:
            model_holder.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            print("Whisper model loaded on CPU (int8)")
        except Exception as e2:
            print(f"Model init failed completely: {e2}")
            model_holder.model = None

class TranscribeRequest(BaseModel):
    video_url: HttpUrl
    language: Optional[str] = None
    sync: Optional[bool] = True
    # Это поле теперь будет работать как "Целевая длительность сегмента"
    max_segment_duration_seconds: int = 15 

# --- Вспомогательные функции ---

def download_to_path(url: str, out_path: str, chunk_size=1<<20) -> str:
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path

def format_time(seconds: float) -> str:
    if seconds is None:
        return "N/A"
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = (total_seconds // 3600)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def group_words_by_duration(all_words: List[Any], target_duration: int) -> List[Dict[str, Any]]:
    """
    Группирует отдельные слова в сегменты заданной длительности.
    Гарантирует, что сегменты не будут слишком короткими (хвосты склеиваются).
    """
    if not all_words:
        return []

    final_segments = []
    buffer_words = []
    
    # Безопасная инициализация
    group_start = all_words[0].start
    
    for word in all_words:
        # Пропускаем слова с битыми таймстемпами (защита от NoneType errors)
        if word.start is None or word.end is None:
            continue

        if not buffer_words:
            group_start = word.start
            
        buffer_words.append(word)
        current_dur = word.end - group_start
        
        # Проверяем условие сброса (достигли лимита)
        if current_dur >= target_duration:
            text = "".join([w.word for w in buffer_words]).strip()
            final_segments.append({
                "start": group_start,
                "end": word.end,
                "text": text
            })
            buffer_words = []
            
    # --- Обработка "хвоста" (остатков) ---
    if buffer_words:
        text = "".join([w.word for w in buffer_words]).strip()
        last_seg_start = group_start
        last_seg_end = buffer_words[-1].end
        
        # Если у нас уже есть сегменты и остаток очень маленький (менее 20% от цели),
        # лучше приклеить его к предыдущему, чтобы LLM не получила обрубок фразы.
        min_threshold = target_duration * 0.2 
        
        if final_segments and (last_seg_end - last_seg_start) < min_threshold:
            prev_seg = final_segments.pop()
            merged_text = prev_seg["text"] + " " + text
            final_segments.append({
                "start": prev_seg["start"],
                "end": last_seg_end,
                "text": merged_text.strip()
            })
        else:
            final_segments.append({
                "start": last_seg_start,
                "end": last_seg_end,
                "text": text
            })
            
    return final_segments

async def transcribe_video_file(video_path: str, language: Optional[str] = None, target_duration: int = 15) -> str:
    if model_holder.model is None:
        raise RuntimeError("Model not initialized")

    def blocking_transcribe(duration_param: int):
        # 1. Получаем все слова с таймстемпами (критично: word_timestamps=True)
        segments_generator, _ = model_holder.model.transcribe(
            video_path, 
            language=language, 
            beam_size=4,
            word_timestamps=True 
        )
        
        # Собираем все слова в один плоский список
        all_words = []
        for segment in segments_generator:
            if segment.words:
                all_words.extend(segment.words)
        
        if not all_words:
            return ""

        # 2. Группируем слова по нашей жесткой логике
        custom_segments = group_words_by_duration(all_words, duration_param)
        
        # 3. Форматируем вывод
        output_lines = []
        for seg in custom_segments:
            start_s = format_time(seg["start"])
            end_s = format_time(seg["end"])
            text = seg["text"]
            output_lines.append(f"[{start_s} -> {end_s}] {text}")
            
        return "\n".join(output_lines).strip()

    async with inference_semaphore:
        loop = asyncio.get_event_loop()
        # Если target_duration не передан, используем дефолт 15
        val_duration = target_duration if target_duration else 15
        transcript = await loop.run_in_executor(executor, blocking_transcribe, val_duration)
        return transcript

# --- Эндпоинты API ---

@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe(req: TranscribeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(STORAGE_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, "input.mp4")
    transcript_path = os.path.join(job_dir, "transcript.txt")
    metrics_path = os.path.join(job_dir, "metrics.json")
    
    start_download = time.time()
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, download_to_path, str(req.video_url), video_path)
    except Exception as e:
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")
    end_download = time.time()
    download_time = end_download - start_download

    if req.sync:
        start_transcribe = time.time()
        try:
            transcript = await transcribe_video_file(video_path, req.language, req.max_segment_duration_seconds)
            end_transcribe = time.time()
            transcribe_time = end_transcribe - start_transcribe
            
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript or "")
                
            metrics = {
                "download_time_seconds": round(download_time, 3),
                "transcription_time_seconds": round(transcribe_time, 3)
            }
            
            return {
                "job_id": job_id, "status": "completed", 
                "transcript": transcript, "metrics": metrics
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        finally:
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
    else:
        async def bg_task(initial_download_time: float, duration_param: int):
            start_transcribe = time.time()
            try:
                transcript = await transcribe_video_file(video_path, req.language, duration_param)
                end_transcribe = time.time()
                transcribe_time = end_transcribe - start_transcribe
                
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript or "")

                metrics = {
                    "download_time_seconds": round(initial_download_time, 3),
                    "transcription_time_seconds": round(transcribe_time, 3)
                }
                with open(metrics_path, "w", encoding="utf-8") as mf:
                    json.dump(metrics, mf)

            except Exception as e:
                with open(os.path.join(job_dir, "error.txt"), "w", encoding="utf-8") as ef:
                    ef.write(f"Error: {str(e)}")
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

        background_tasks.add_task(bg_task, download_time, req.max_segment_duration_seconds)
        return {"job_id": job_id, "status": "processing", "message": "Poll /job/{job_id}"}


@app.get("/job/{job_id}", response_model=Dict[str, Any])
async def job_status(job_id: str):
    job_dir = os.path.join(STORAGE_ROOT, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    
    transcript_path = os.path.join(job_dir, "transcript.txt")
    error_path = os.path.join(job_dir, "error.txt")
    metrics_path = os.path.join(job_dir, "metrics.json")

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
            
        metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    metrics = json.load(mf)
            except Exception:
                pass
        
        shutil.rmtree(job_dir)
        return {"job_id": job_id, "status": "completed", "transcript": transcript, "metrics": metrics, "cleanup": True}
    
    if os.path.exists(error_path):
        with open(error_path, "r", encoding="utf-8") as f:
            err = f.read()
        shutil.rmtree(job_dir)
        raise HTTPException(status_code=500, detail=err)

    return {"job_id": job_id, "status": "processing", "transcript": None}


@app.get("/health")
async def health():
    return {"ok": True, "model_loaded": model_holder.model is not None}
