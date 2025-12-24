import requests
from typing import List, Optional
import os
from services.config import ProxyConfig, AppConfig

class Client:
    """
    Клиент для взаимодействия с ML Backend.
    ВАЖНО: ML Backend - локальный сервис, прокси НЕ используется.
    """
    
    SERVER_URL = AppConfig.ML_SERVER_URL
    
    @classmethod
    def _get_session(cls) -> requests.Session:
        """Создает сессию БЕЗ прокси для локального ML Backend"""
        return ProxyConfig.get_session_with_proxy(use_proxy=False)

    @classmethod
    def transcribe(cls, video_url: str, step_id: int) -> str:
        """Транскрибация видео через Whisper."""
        if not video_url:
            return ""

        payload = {
            "video_url": video_url,
            "sync": True,
            "language": "ru",
            "max_segment_duration_seconds": 15 
        }

        print(f"   [Transcriber] Step {step_id}: Отправка видео...")
        session = cls._get_session()
        
        try:
            resp = session.post(
                f"{cls.SERVER_URL}/transcribe",
                json=payload,
                timeout=600
            )
            if resp.status_code == 200:
                data = resp.json()
                transcript = data.get("transcript", "")
                if transcript:
                    print(f"   [Transcriber] Получено {len(transcript)} символов")
                return transcript
            else:
                print(f"   [Transcriber] Ошибка {resp.status_code}: {resp.text[:200]}")
                return ""
        except requests.exceptions.Timeout:
            print(f"   [Transcriber] Timeout после 600 секунд")
            return ""
        except Exception as e:
            print(f"   [Transcriber] Exception: {type(e).__name__}: {e}")
            return ""

    @classmethod
    def get_image_clip_vector(cls, image_path: str) -> Optional[List[float]]:
        """Получает CLIP вектор изображения (для дедупликации)."""
        if not os.path.exists(image_path):
            print(f"   [CLIP] Файл не найден: {image_path}")
            return None

        session = cls._get_session()
        
        try:
            with open(image_path, "rb") as img_file:
                files = {"file": img_file}
                resp = session.post(
                    f"{cls.SERVER_URL}/clip_embed",
                    files=files,
                    timeout=60
                )

            if resp.status_code == 200:
                data = resp.json()
                vector = data.get("features", None)
                if vector:
                    return vector
                else:
                    print(f"   [CLIP] Пустой вектор в ответе")
            else:
                print(f"   [CLIP] Ошибка {resp.status_code}: {resp.text[:200]}")
            
            return None
        except Exception as e:
            print(f"   [CLIP] Exception: {type(e).__name__}: {e}")
            return None
        
    @classmethod
    def get_image_description(cls, image_path: str, 
                            prompt: str = "Дай подробное описание этого изображения на русском языке. "
                                        "Сосредоточься на ключевых элементах, их взаимодействии и контексте.") -> Optional[str]:
        """Получает текстовое описание изображения от LLaVA."""
        if not os.path.exists(image_path):
            print(f"   [LLaVA] Файл не найден: {image_path}")
            return None
            
        session = cls._get_session()

        try:
            with open(image_path, "rb") as img_file:
                files = {"file": img_file}
                data_payload = {"prompt": prompt}
                resp = session.post(
                    f"{cls.SERVER_URL}/llava_describe",
                    files=files,
                    data=data_payload,
                    timeout=120
                )

            if resp.status_code == 200:
                data = resp.json()
                description = data.get("description", None)
                if description:
                    return description
                else:
                    print(f"   [LLaVA] Пустое описание в ответе")
            else:
                print(f"   [LLaVA] Ошибка {resp.status_code}: {resp.text[:200]}")
            
            return None
        except Exception as e:
            print(f"   [LLaVA] Exception: {type(e).__name__}: {e}")
            return None

    @classmethod
    def get_text_embedding(cls, text: str, model_name: str = "paraphrase-multilingual-mpnet-base-v2") -> Optional[List[float]]:
        """Получает текстовый вектор для индексации."""
        if not text or len(text.strip()) < 10:
            return None
        
        session = cls._get_session()
        
        try:
            payload = {
                "text": text,
                "model_name": model_name
            }
            resp = session.post(
                f"{cls.SERVER_URL}/text_embed",
                json=payload,
                timeout=60
            )
            
            if resp.status_code == 200:
                data = resp.json()
                embedding = data.get("embedding", None)
                dimension = data.get("dimension")
                if embedding:
                    return embedding
                else:
                    print(f"   [TextEmbed] Пустой вектор в ответе")
            else:
                print(f"   [TextEmbed] Ошибка {resp.status_code}: {resp.text[:200]}")
            
            return None
        except Exception as e:
            print(f"   [TextEmbed] Exception: {type(e).__name__}: {e}")
            return None

    @classmethod
    def get_batch_text_embeddings(cls, texts: List[str], 
                                  model_name: str = "paraphrase-multilingual-mpnet-base-v2") -> Optional[List[List[float]]]:
        """Батчевая векторизация текстов."""
        if not texts:
            return None
        
        session = cls._get_session()
        
        try:
            payload = {
                "texts": texts,
                "model_name": model_name
            }
            resp = session.post(
                f"{cls.SERVER_URL}/text_embed_batch",
                json=payload,
                timeout=300
            )
            
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", None)
                if embeddings:
                    return embeddings
                else:
                    print(f"   [TextEmbed Batch] Пустой список векторов")
            else:
                print(f"   [TextEmbed Batch] Ошибка {resp.status_code}: {resp.text[:200]}")
            
            return None
        except Exception as e:
            print(f"   [TextEmbed Batch] Exception: {type(e).__name__}: {e}")
            return None

    @classmethod
    def health_check(cls) -> bool:
        """
        Проверяет работу ML Backend через реальную генерацию эмбеддинга.
        Отправляет тестовый запрос 'Проверка связи' на эндпоинт /text_embed.
        """
        session = cls._get_session()
        try:
            payload = {
                "text": "Проверка связи",
                "model_name": "paraphrase-multilingual-mpnet-base-v2"
            }
            # Используем короткий таймаут, так как модель может грузиться
            resp = session.post(
                f"{cls.SERVER_URL}/text_embed", 
                json=payload, 
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                # Убеждаемся, что вернулся именно эмбеддинг
                return "embedding" in data and len(data["embedding"]) > 0
            
            return False
        except Exception as e:
            # print(f"Health check failed: {e}") # Раскомментировать для отладки
            return False
