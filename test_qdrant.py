import os
import sys
import random
import uuid
from dotenv import load_dotenv

# 1. Настройка окружения
# Добавляем текущую директорию в путь импорта, чтобы Python видел папки services/ и CourseProcessor/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ВАЖНО: Отключаем прокси для локальных адресов (фикс ошибки 502 Bad Gateway)
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

# Загружаем переменные из .env (если скрипт запускается отдельно)
load_dotenv()

# 2. Импорты из основного проекта
from services.config import AppConfig
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def test_qdrant_connection():
    print(f"\n{'='*50}")
    print("🧪 ТЕСТ ПОДКЛЮЧЕНИЯ К QDRANT")
    print(f"{'='*50}")

    # Используем настройки из services/config.py
    host = AppConfig.QDRANT_HOST
    port = AppConfig.QDRANT_PORT
    
    print(f"📍 Подключение к: {host}:{port}...")

    try:
        # Инициализация клиента (как в QdrantKnowledgeBaseIndexer)
        client = QdrantClient(host=host, port=port)
        
        # Настройки для теста
        collection_name = "test_manual_collection"
        vector_size = 768 # Размерность модели paraphrase-multilingual-mpnet-base-v2
        
        # 1. Создание коллекции
        # recreate_collection удалит старую, если она была, и создаст новую
        print(f"🛠️  Создание коллекции '{collection_name}'...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print("   ✅ Коллекция создана.")

        # 2. Генерация случайного вектора
        print("🎲 Генерация случайного вектора (dim=768)...")
        random_vector = [random.random() for _ in range(vector_size)]
        
        # 3. Вставка данных
        point_id = str(uuid.uuid4())
        payload = {
            "test_key": "test_value",
            "timestamp": "2023-01-01",
            "info": "Это тестовый вектор, сгенерированный скриптом"
        }
        
        print(f"📤 Отправка вектора (ID: {point_id})...")
        operation_info = client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=random_vector,
                    payload=payload
                )
            ]
        )
        print(f"   ✅ Статус операции: {operation_info.status}")

        # 4. Проверка (чтение)
        print("🔍 Проверка количества записей...")
        count_result = client.count(collection_name=collection_name)
        print(f"   📊 В коллекции сейчас элементов: {count_result.count}")
        
        if count_result.count > 0:
            print("\n🎉 УСПЕХ! Qdrant работает корректно и принимает данные.")
        else:
            print("\n⚠️ ВНИМАНИЕ! Коллекция создана, но пуста.")

    except Exception as e:
        print(f"\n❌ ОШИБКА ПОДКЛЮЧЕНИЯ ИЛИ ЗАПИСИ:")
        print(e)
        print("\nСоветы:")
        print("1. Убедитесь, что Qdrant запущен (docker ps)")
        print("2. Проверьте переменную no_proxy (должна включать localhost)")

if __name__ == "__main__":
    test_qdrant_connection()