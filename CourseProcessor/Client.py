import requests
from typing import List, Optional
import os

class Client:
    SERVER_URL = "http://localhost:8000"

    @classmethod
    def transcribe(cls, video_url: str, step_id: int) -> str:
        # Этот метод корректен, ЕСЛИ вы обновили сервер под прием JSON (video_url)
        if not video_url:
            return ""

        payload = {
            "video_url": video_url,
            "sync": True,
            "language": "ru",
            "max_segment_duration_seconds": 15 
        }

        print(f"   [Transcriber] Step {step_id}: Отправка видео...")
        session = requests.Session()
        session.trust_env = False 
        
        try:
            resp = session.post(cls.SERVER_URL + "/transcribe", json=payload, timeout=600)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("transcript", "")
            else:
                print(f"   [Transcriber] Ошибка: {resp.status_code}")
                return ""
        except Exception as e:
            print(f"   [Transcriber] Exception: {e}")
            return ""


    @classmethod
    def get_image_clip_vector(cls, image_path: str) -> Optional[List[float]]:
        if not os.path.exists(image_path):
            return None

        session = requests.Session()
        session.trust_env = False 
        
        try:
            with open(image_path, "rb") as img_file:
                # ИСПРАВЛЕНИЕ 1: Ключ должен быть "file", как аргумент в FastAPI
                files = {"file": img_file} 
                resp = session.post(cls.SERVER_URL + "/clip_embed", files=files, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                # ИСПРАВЛЕНИЕ 2: Сервер возвращает "features", а не "vector"
                vector = data.get("features", None)
                if vector:
                    print(f"   [CLIP] Вектор получен. Длина: {len(vector)}")
                    return vector
            
            print(f"   [CLIP] Ошибка или пустой ответ: {resp.status_code}")
            return None
        except Exception as e:
            print(f"   [CLIP] Exception: {e}")
            return None
        
    @classmethod
    def get_image_description(cls, image_path: str) -> Optional[str]:
        if not os.path.exists(image_path):
            return None
            
        session = requests.Session()
        session.trust_env = False

        try:
            with open(image_path, "rb") as img_file:
                # ИСПРАВЛЕНИЕ 3: Ключ должен быть "file"
                files = {"file": img_file}
                data_payload = {
                    "prompt": "Дай подробное описание..."
                }
                resp = session.post(cls.SERVER_URL + "/llava_describe", files=files, data=data_payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                # Здесь ключ "description" совпадает с сервером
                return data.get("description", None)
            
            print(f"   [LLaVA] Ошибка: {resp.status_code}")
            return None
        except Exception as e:
            print(f"   [LLaVA] Exception: {e}")
            return None
