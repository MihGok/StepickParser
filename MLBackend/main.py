import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import uvicorn

from .core.model_manager import model_manager
from .services.whisper_service import WhisperService
from .services.llava_service import LlavaService
from .services.clip_service import ClipService
from .services.text_encoder_service import TextEncoderService

app = FastAPI()
TEMP_DIR = "server_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class TextRequest(BaseModel):
    text: str
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"

class BatchTextRequest(BaseModel):
    texts: List[str]
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"

class TranscribeRequest(BaseModel):
    video_url: str

@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.mp4")
    try:
        with requests.get(req.video_url, stream=True) as r:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
        
        whisper = model_manager.get_model("whisper", WhisperService)
        segments = whisper.transcribe(path)
        return {
            "transcript": " ".join([s.text for s in segments]),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()} 
                for s in segments
            ]
        }
    finally:
        if os.path.exists(path): os.remove(path)

@app.post("/clip_embed")
async def clip_embed(file: UploadFile = File(...)):
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        clip = model_manager.get_model("clip", ClipService)
        return {"features": clip.get_image_features(path)[0]}
    finally:
        if os.path.exists(path): os.remove(path)

@app.post("/llava_describe")
async def llava_describe(file: UploadFile = File(...), prompt: str = Form(...)):
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        llava = model_manager.get_model("llava", LlavaService)
        return {"description": llava.analyze(path, prompt)}
    finally:
        if os.path.exists(path): os.remove(path)

@app.post("/text_embed_batch")
async def text_embed_batch(req: BatchTextRequest):
    encoder = model_manager.get_model(f"text_{req.model_name}", 
                                    lambda: TextEncoderService(req.model_name))
    return {"embeddings": encoder.encode_batch(req.texts)}

@app.post("/text_embed")
async def text_embed(req: TextRequest):
    encoder = model_manager.get_model(f"text_{req.model_name}", 
                                    lambda: TextEncoderService(req.model_name))
    return {"embedding": encoder.encode_single(req.text)}

if __name__ == "__main__":
    HOST = "0.0.0.0"  # Доступен из локальной сети и Docker
    PORT = 8000       # Порт, на который стучится Client
    WORKERS = 1       # ВАЖНО: 1 воркер, чтобы не дублировать модели в VRAM
    
    print(f"🚀 Запуск ML Backend на http://{HOST}:{PORT}")
    print(f"⚙️  Workers: {WORKERS} (Single process mode for GPU safety)")
    
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT, 
        workers=WORKERS
    )