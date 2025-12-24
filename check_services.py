import time
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ваши импорты (сохранена структура)
from services.storage_service import StorageService
from CourseProcessor.client_api import Client
from services.LLM_Service.llm_service import GeminiService
from services.config import AppConfig
from qdrant_client import QdrantClient

load_dotenv()

# Модель для теста Gemini
class HealthCheckResponse(BaseModel):
    reply: str = Field(..., description="Просто верни слово 'OK'")

def check_all():
    print(f"\n{'='*60}")
    print("🚀 ДИАГНОСТИКА СЕРВИСОВ")
    print(f"{'='*60}\n")
    
    results = {}
    
    # --- 1. MinIO ---
    try:
        storage = StorageService()
        # Проверяем, существует ли бакет (инициализация это делает)
        results["MinIO"] = True
        print(f"✅ MinIO доступен (Bucket: {storage.bucket})")
    except Exception as e:
        results["MinIO"] = False
        print(f"❌ MinIO ошибка: {e}")
    
    print("-" * 30)

    # --- 2. ML Backend ---
    try:
        # Используем обновленный health_check клиента
        if Client.health_check():
            results["ML Backend"] = True
            print(f"✅ ML Backend доступен ({AppConfig.ML_SERVER_URL})")
        else:
            results["ML Backend"] = False
            print(f"❌ ML Backend недоступен (URL: {AppConfig.ML_SERVER_URL})")
    except Exception as e:
        results["ML Backend"] = False
        print(f"❌ ML Backend ошибка соединения: {e}")
    
    print("-" * 30)

    # --- 3. Qdrant ---
    try:
        qdrant = QdrantClient(
            host=AppConfig.QDRANT_HOST,
            port=AppConfig.QDRANT_PORT
        )
        col_info = qdrant.get_collections()
        results["Qdrant"] = True
        print(f"✅ Qdrant доступен. Коллекций: {len(col_info.collections)}")
    except Exception as e:
        results["Qdrant"] = False
        print(f"❌ Qdrant ошибка: {e}")
    
    print("-" * 30)

    # --- 4. Gemini API ---
    try:
        print("⏳ Gemini: Отправка тестового запроса...")
        gemini = GeminiService()
        
        start_t = time.time()
        # Реальный запрос к модели
        resp = gemini.generate(
            prompt="Say OK",
            response_schema=HealthCheckResponse,
            model_name="gemini-2.5-flash", # Быстрая модель для теста
            temperature=0.0
        )
        duration = time.time() - start_t
        
        if resp and resp.reply:
            results["Gemini API"] = True
            print(f"✅ Gemini API работает ({duration:.2f}s). Ответ: {resp.reply}")
        else:
            raise ValueError("Пустой ответ")
            
    except Exception as e:
        results["Gemini API"] = False
        print(f"❌ Gemini API ошибка: {e}")
    
    # --- ИТОГИ ---
    print("\n" + "="*60)
    total = len(results)
    working = sum(results.values())
    print(f"ИТОГ: Работает {working}/{total} сервисов")
    
    if working == total:
        print("✅ СИСТЕМА ГОТОВА К РАБОТЕ!")
        return True
    else:
        print("⚠️ ЕСТЬ ПРОБЛЕМЫ")
        return False

if __name__ == "__main__":
    check_all()
