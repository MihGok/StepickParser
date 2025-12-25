import os
import requests
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# Конфигурация
KNOWLEDGE_BASE_DIR = "knowledge_base"
ML_BACKEND_URL = "http://localhost:8000"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge_base"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
VECTOR_DIM = 768  # Размерность для mpnet-base-v2

# ВАЖНО: Отключаем прокси для локальных запросов
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0,::1'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0,::1'

# Создаем сессию БЕЗ прокси для локальных запросов
local_session = requests.Session()
local_session.trust_env = False  # Игнорировать системные прокси

def get_embedding(text: str) -> list:
    """Получить эмбеддинг через ML backend"""
    response = local_session.post(
        f"{ML_BACKEND_URL}/text_embed",
        json={"text": text, "model_name": MODEL_NAME}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Разбить текст на чанки с перекрытием"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks if chunks else [text]

def read_knowledge_base():
    """Прочитать все файлы content.txt из базы знаний"""
    documents = []
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    
    if not kb_path.exists():
        print(f"❌ Папка {KNOWLEDGE_BASE_DIR} не найдена!")
        return documents
    
    # Ищем все content.txt файлы
    content_files = list(kb_path.rglob("content.txt"))
    
    print(f"📚 Найдено файлов: {len(content_files)}")
    
    for file_path in content_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                continue
            
            # Определяем категорию и название урока
            parts = file_path.parts
            category = parts[-3] if len(parts) >= 3 else "unknown"
            lesson = parts[-2] if len(parts) >= 2 else "unknown"
            
            # Разбиваем на чанки
            chunks = chunk_text(content)
            
            for idx, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "category": category,
                    "lesson": lesson,
                    "file_path": str(file_path),
                    "chunk_id": idx,
                    "total_chunks": len(chunks)
                })
        
        except Exception as e:
            print(f"⚠️ Ошибка при чтении {file_path}: {e}")
    
    return documents

def create_collection(client: QdrantClient):
    """Создать коллекцию в Qdrant"""
    try:
        # Удаляем старую коллекцию если существует
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️ Старая коллекция '{COLLECTION_NAME}' удалена")
    except Exception as e:
        print(f"ℹ️ Коллекция не существовала: {e}")
    
    # Создаем новую коллекцию
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    print(f"✅ Коллекция '{COLLECTION_NAME}' создана")

def index_documents(documents: list):
    """Индексировать документы в Qdrant"""
    # ИСПРАВЛЕНО: Правильное подключение к Qdrant
    client = QdrantClient(
        host="localhost",
        port=6333,
        timeout=60,
        prefer_grpc=False  # Только REST API
    )
    
    # Создаем коллекцию
    create_collection(client)
    
    print(f"🔄 Начинаем индексацию {len(documents)} чанков...")
    
    points = []
    
    for idx, doc in enumerate(tqdm(documents, desc="Векторизация")):
        try:
            # Получаем эмбеддинг
            embedding = get_embedding(doc["text"])
            
            # Создаем точку для Qdrant
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": doc["text"],
                    "category": doc["category"],
                    "lesson": doc["lesson"],
                    "file_path": doc["file_path"],
                    "chunk_id": doc["chunk_id"],
                    "total_chunks": doc["total_chunks"]
                }
            )
            points.append(point)
            
            # Загружаем батчами по 100
            if len(points) >= 100:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []
        
        except Exception as e:
            print(f"⚠️ Ошибка при индексации документа {idx}: {e}")
    
    # Загружаем остатки
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    # Проверяем результат
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"\n✅ Индексация завершена!")
    print(f"📊 Всего векторов в коллекции: {collection_info.points_count}")

def main():
    print("=" * 60)
    print("🚀 Индексация базы знаний")
    print("=" * 60)
    
    # Проверяем доступность сервисов
    try:
        local_session.get(f"{ML_BACKEND_URL}/docs", timeout=2)
        print(f"✅ ML Backend доступен: {ML_BACKEND_URL}")
    except Exception as e:
        print(f"❌ ML Backend недоступен: {ML_BACKEND_URL}")
        print(f"   Ошибка: {e}")
        return
    
    try:
        # Пробуем через REST API напрямую
        response = local_session.get(f"{QDRANT_URL}/collections", timeout=5)
        if response.status_code == 200:
            print(f"✅ Qdrant доступен: {QDRANT_URL}")
        else:
            raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Qdrant недоступен: {QDRANT_URL}")
        print(f"   Ошибка: {e}")
        return
    
    # Читаем документы
    documents = read_knowledge_base()
    
    if not documents:
        print("❌ Документы не найдены!")
        return
    
    print(f"\n📝 Всего чанков для индексации: {len(documents)}")
    
    # Показываем статистику по категориям
    categories = {}
    for doc in documents:
        cat = doc["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Распределение по категориям:")
    for cat, count in categories.items():
        print(f"   {cat}: {count} чанков")
    
    # Индексируем
    print()
    index_documents(documents)
    
    print("\n🎉 Готово! Теперь можно запускать RAG сервер.")

if __name__ == "__main__":
    main()
