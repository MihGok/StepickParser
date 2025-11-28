import os
import shutil
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Any, Dict
import requests
from faster_whisper import WhisperModel
import time # Добавлен для измерения времени
import json # Добавлен для сохранения и загрузки метрик

# --- Конфигурация: Установлена модель по умолчанию 'medium' ---
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

MAX_CONCURRENT_INFERENCES = int(os.environ.get("MAX_CONCURRENT_INFERENCES", "1"))
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

executor = ThreadPoolExecutor(max_workers=2)
app = FastAPI(title="ML Service")

STORAGE_ROOT = os.environ.get("STORAGE_ROOT", "/tmp/ml_service_storage")
os.makedirs(STORAGE_ROOT, exist_ok=True)


# --- Инициализация модели ---
class ModelHolder:
    """Держатель для модели Whisper."""
    model: Optional[WhisperModel] = None

model_holder = ModelHolder()

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске приложения с резервным вариантом на CPU."""
    try:
        # При загрузке консоль теперь будет выводить: Whisper model loaded: medium cuda
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

# --- Схемы данных ---
class TranscribeRequest(BaseModel):
    video_url: HttpUrl
    language: Optional[str] = None
    sync: Optional[bool] = True

# --- Вспомогательные функции ---

def download_to_path(url: str, out_path: str, chunk_size=1<<20) -> str:
    """Синхронно загружает файл по URL в указанный путь."""
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path

def format_time(seconds: float) -> str:
    """Форматирует секунды в строку HH:MM:SS.ms."""
    if seconds is None:
        return "N/A"
    
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = (total_seconds // 3600)
    
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


async def transcribe_video_file(video_path: str, language: Optional[str] = None) -> str:
    """Выполняет инференцию Whisper и форматирует результат с временными метками."""
    if model_holder.model is None:
        raise RuntimeError("Model not initialized or failed to load.")

    def blocking_transcribe():
        """Синхронная функция для инференции."""
        segments, _ = model_holder.model.transcribe(
            video_path, 
            language=language, 
            beam_size=5,
            word_timestamps=False # Используем только метки сегментов
        )
        
        output_lines = []
        for s in segments:
            text = s.text.strip()
            if text:
                start_time = format_time(s.start)
                end_time = format_time(s.end)
                # Формат: [HH:MM:SS.ms -> HH:MM:SS.ms] Text
                output_lines.append(f"[{start_time} -> {end_time}] {text}")
                
        return "\n".join(output_lines).strip()

    async with inference_semaphore:
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(executor, blocking_transcribe)
        return transcript

# --- Эндпоинты API (логика очистки сохранена) ---

@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe(req: TranscribeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(STORAGE_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, "input.mp4")
    transcript_path = os.path.join(job_dir, "transcript.txt")
    metrics_path = os.path.join(job_dir, "metrics.json") # Путь для сохранения метрик
    
    # 1. Загрузка видео и измерение времени
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
        # --- Синхронный режим: ждем результата ---
        start_transcribe = time.time()
        try:
            transcript = await transcribe_video_file(video_path, req.language)
            end_transcribe = time.time()
            transcribe_time = end_transcribe - start_transcribe
            
            # Сохраняем результат
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript or "")
                
            # Формируем метрики для немедленного ответа
            metrics = {
                "download_time_seconds": round(download_time, 3),
                "transcription_time_seconds": round(transcribe_time, 3)
            }
            
            return {
                "job_id": job_id, 
                "status": "completed", 
                "transcript": transcript,
                "metrics": metrics # Добавляем метрики в ответ
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        finally:
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
    else:
        # --- Асинхронный режим: запускаем задачу в фоне ---
        async def bg_task(initial_download_time: float):
            """Фоновая задача для выполнения транскрипции и сохранения результата/ошибки."""
            start_transcribe = time.time()
            try:
                transcript = await transcribe_video_file(video_path, req.language)
                end_transcribe = time.time()
                transcribe_time = end_transcribe - start_transcribe
                
                # Сохраняем результат
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript or "")

                # Сохраняем метрики в отдельный файл
                metrics = {
                    "download_time_seconds": round(initial_download_time, 3),
                    "transcription_time_seconds": round(transcribe_time, 3)
                }
                with open(metrics_path, "w", encoding="utf-8") as mf:
                    json.dump(metrics, mf)

            except Exception as e:
                # Сохраняем ошибку в файл, если транскрипция не удалась
                with open(os.path.join(job_dir, "error.txt"), "w", encoding="utf-8") as ef:
                    ef.write(f"Error during transcription: {str(e)}")
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

        background_tasks.add_task(bg_task, download_time) # Передаем время загрузки в фоновую задачу
        return {"job_id": job_id, "status": "processing", "message": "Poll the /job/{job_id} endpoint for results."}


@app.get("/job/{job_id}", response_model=Dict[str, Any])
async def job_status(job_id: str):
    job_dir = os.path.join(STORAGE_ROOT, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found. It may have been cleaned up.")
    
    transcript_path = os.path.join(job_dir, "transcript.txt")
    error_path = os.path.join(job_dir, "error.txt")
    metrics_path = os.path.join(job_dir, "metrics.json") # Путь для чтения метрик

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
            
        metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    metrics = json.load(mf)
            except Exception as e:
                print(f"Warning: Failed to load metrics for job {job_id}: {e}")
        
        # Очистка после успешного получения результата
        shutil.rmtree(job_dir) 
        
        return {
            "job_id": job_id,
            "status": "completed",
            "transcript": transcript,
            "metrics": metrics, # Добавляем метрики в ответ
            "cleanup": True,
        }
    
    if os.path.exists(error_path):
        with open(error_path, "r", encoding="utf-8") as f:
            error_message = f.read()
            
        # Очистка после получения ошибки
        shutil.rmtree(job_dir)
        
        raise HTTPException(
            status_code=500,
            detail=f"Job failed. Error message: {error_message}",
        )

    return {
        "job_id": job_id,
        "status": "processing",
        "transcript": None,
        "message": "Transcription is still running. Please poll again later.",
    }


@app.get("/health")
async def health():
    return {"ok": True, "model_loaded": model_holder.model is not None}
