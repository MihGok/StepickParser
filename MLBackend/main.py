import shutil
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

# Импорты из предыдущего шага (файлы core и services должны быть созданы)
from core.model_manager import model_manager
from services.whisper_service import WhisperService
from services.llava_service import LlavaService
from services.clip_service import ClipService

app = FastAPI(title="Pure ML Backend")

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Endpoint 1: Whisper (Аудио/Видео -> Текст) ---
@app.post("/v1/transcribe")
async def transcribe_video(
    file: UploadFile = File(...), 
    language: Optional[str] = Form(None)
):
    """
    Принимает файл видео/аудио, прогоняет через Whisper.
    Whisper загружается, остальные выгружаются.
    """
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        # Сохраняем загруженный файл
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Получаем модель (Менеджер сам выгрузит CLIP/LLaVA если надо)
        whisper = model_manager.get_model("whisper", WhisperService)
        
        # Транскрибируем
        segments = whisper.transcribe(temp_path, language=language)
        
        # Формируем простой ответ (или можно вернуть полный JSON с таймкодами)
        full_text = " ".join([s.text for s in segments])
        segments_data = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        
        return {"text": full_text, "segments": segments_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- Endpoint 2: CLIP (Картинка -> Вектор) ---
@app.post("/v1/clip/embed")
async def get_clip_features(file: UploadFile = File(...)):
    """
    Принимает картинку, возвращает вектор (embeddings).
    CLIP загружается, Whisper/LLaVA выгружаются.
    """
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        clip = model_manager.get_model("clip", ClipService)
        features = clip.get_image_features(temp_path)
        
        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Endpoint 3: LLaVA (Картинка + Промпт -> Описание) ---
@app.post("/v1/llava/describe")
async def describe_image(
    file: UploadFile = File(...), 
    prompt: str = Form("Дай подробное описание этой картинки. Отдельно сосредоточься на компонентах" \
    " и их взаимодействии друг с другом.")
):
    """
    Принимает картинку и промпт.
    LLaVA загружается, остальные выгружаются.
    """
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        llava = model_manager.get_model("llava", LlavaService)
        response = llava.analyze(temp_path, prompt)
        
        # Очистка ответа от системных токенов
        clean_text = response.split("[/INST]")[-1].strip()
        
        return {"description": clean_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
