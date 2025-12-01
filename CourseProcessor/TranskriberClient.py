import requests

class TranscriberClient:
    SERVER_URL = "http://localhost:8000/transcribe"

    @classmethod
    def transcribe(cls, video_url: str, step_id: int) -> str:
        """
        Отправляет URL видео на сервер для транскрибации.
        Отключает использование системных прокси для localhost.
        """
        if not video_url:
            return ""

        payload = {
            "video_url": video_url,
            "sync": True,
            "language": "ru",
            "max_segment_duration_seconds": 15 
        }

        print(f"   [Transcriber] Step {step_id}: Отправка видео (seg=15s)...")
        
        # Создаем сессию и отключаем учет переменных окружения (прокси)
        session = requests.Session()
        session.trust_env = False 
        
        try:
            # post через session с отключенными прокси
            resp = session.post(cls.SERVER_URL, json=payload, timeout=600)
            
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("transcript", "")
                print(f"   [Transcriber] Успех. Получено {len(text)} симв.")
                return text
            else:
                print(f"   [Transcriber] Ошибка сервера: {resp.status_code}")
                return ""
        except Exception as e:
            print(f"   [Transcriber] Ошибка соединения: {e}")
            return ""
