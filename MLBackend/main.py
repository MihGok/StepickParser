import shutil
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests


from core.model_manager import model_manager
from services.whisper_service import WhisperService
from services.llava_service import LlavaService
from services.clip_service import ClipService
import uvicorn

class TranscribeRequest(BaseModel):
    video_url: str
    language: Optional[str] = "ru"
    sync: bool = True
    max_segment_duration_seconds: int = 15


app = FastAPI(title="Pure ML Backend")

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/transcribe")
async def transcribe_video(request: TranscribeRequest):
    """
    Принимает JSON с URL, скачивает видео, прогоняет через Whisper.
    """
    filename = f"{uuid.uuid4()}_video.mp4"
    temp_path = os.path.join(TEMP_DIR, filename)
    
    print(f"Скачивание видео: {request.video_url}")
    
    try:
        with requests.get(request.video_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        
        print("✅ Видео скачано, запуск Whisper...")

        whisper = model_manager.get_model("whisper", WhisperService)

        segments = whisper.transcribe(temp_path, language=request.language)
        
        full_text = " ".join([s.text for s in segments])
        segments_data = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        
        return {"transcript": full_text, "segments": segments_data}
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/clip_embed")
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


@app.post("/llava_describe")
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
        clean_text = response.split("[/INST]")[-1].strip()
        
        return {"description": clean_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    print("🚀 Запуск сервера ML Backend...")
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)