import requests
from typing import List, Optional
import os

class Client:
    SERVER_URL = "http://localhost:8000"

    @classmethod
    def transcribe(cls, video_url: str, step_id: int) -> str:
        """
        Транскрибация видео через Whisper.
        
        Args:
            video_url: URL видео
            step_id: ID шага (для логирования)
            
        Returns:
            Текст транскрипта
        """
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
        """
        Получает CLIP вектор изображения.
        Используется ТОЛЬКО для дедупликации кадров.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            CLIP вектор (512d) или None
        """
        if not os.path.exists(image_path):
            return None

        session = requests.Session()
        session.trust_env = False 
        
        try:
            with open(image_path, "rb") as img_file:
                files = {"file": img_file}
                resp = session.post(cls.SERVER_URL + "/clip_embed", files=files, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
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
    def get_image_description(cls, image_path: str, 
                            prompt: str = "Дай подробное описание этого изображения на русском языке. "
                                        "Сосредоточься на ключевых элементах, их взаимодействии и контексте.") -> Optional[str]:
        """
        Получает текстовое описание изображения от LLaVA.
        
        Args:
            image_path: Путь к изображению
            prompt: Промпт для LLaVA
            
        Returns:
            Текстовое описание или None
        """
        if not os.path.exists(image_path):
            return None
            
        session = requests.Session()
        session.trust_env = False

        try:
            with open(image_path, "rb") as img_file:
                files = {"file": img_file}
                data_payload = {"prompt": prompt}
                resp = session.post(cls.SERVER_URL + "/llava_describe", files=files, data=data_payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                description = data.get("description", None)
                if description:
                    print(f"   [LLaVA] Описание получено: {description[:100]}...")
                return description
            
            print(f"   [LLaVA] Ошибка: {resp.status_code}")
            return None
        except Exception as e:
            print(f"   [LLaVA] Exception: {e}")
            return None

    @classmethod
    def get_text_embedding(cls, text: str, model_name: str = "paraphrase-multilingual-mpnet-base-v2") -> Optional[List[float]]:
        """
        НОВОЕ: Получает текстовый вектор для индексации в Qdrant.
        Используется для векторизации текста шагов и описаний LLaVA.
        
        Args:
            text: Текст для векторизации
            model_name: Имя модели sentence-transformers
            
        Returns:
            Вектор (768d) или None
        """
        if not text or len(text.strip()) < 10:
            return None
        
        session = requests.Session()
        session.trust_env = False
        
        try:
            payload = {
                "text": text,
                "model_name": model_name
            }
            resp = session.post(cls.SERVER_URL + "/text_embed", json=payload, timeout=60)
            
            if resp.status_code == 200:
                data = resp.json()
                embedding = data.get("embedding", None)
                if embedding:
                    print(f"   [TextEmbed] Вектор получен. Размерность: {data.get('dimension')}")
                    return embedding
            
            print(f"   [TextEmbed] Ошибка: {resp.status_code}")
            return None
        except Exception as e:
            print(f"   [TextEmbed] Exception: {e}")
            return None

    @classmethod
    def get_batch_text_embeddings(cls, texts: List[str], 
                                  model_name: str = "paraphrase-multilingual-mpnet-base-v2") -> Optional[List[List[float]]]:
        """
        НОВОЕ: Батчевая векторизация текстов.
        Эффективнее для индексации большого количества документов.
        
        Args:
            texts: Список текстов
            model_name: Имя модели
            
        Returns:
            Список векторов или None
        """
        if not texts:
            return None
        
        session = requests.Session()
        session.trust_env = False
        
        try:
            payload = {
                "texts": texts,
                "model_name": model_name
            }
            resp = session.post(cls.SERVER_URL + "/text_embed_batch", json=payload, timeout=300)
            
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", None)
                if embeddings:
                    print(f"   [TextEmbed Batch] Получено {data.get('count')} векторов. "
                          f"Размерность: {data.get('dimension')}")
                    return embeddings
            
            print(f"   [TextEmbed Batch] Ошибка: {resp.status_code}")
            return None
        except Exception as e:
            print(f"   [TextEmbed Batch] Exception: {e}")
            return None
