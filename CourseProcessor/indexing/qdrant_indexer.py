import os
import json
import uuid
import requests
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

class QdrantKnowledgeBaseIndexer:
    def __init__(self, 
                 knowledge_base_dir: str,
                 ml_server_url: str = "http://localhost:8000",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 text_model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        
        self.knowledge_base_dir = knowledge_base_dir
        self.ml_server_url = ml_server_url.rstrip('/')
        self.text_model_name = text_model_name
        
        # Конфигурация коллекций
        self.text_collection = "course_text_chunks"
        self.image_collection = "course_image_descriptions"
        
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.vector_size = 768 # Фиксировано для paraphrase-multilingual-mpnet-base-v2
        
        self._init_collections()

    def _init_collections(self):
        """Создает коллекции если их нет"""
        existing = [c.name for c in self.client.get_collections().collections]
        
        for name in [self.text_collection, self.image_collection]:
            if name not in existing:
                print(f"[Qdrant] Создание коллекции '{name}'...")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )

    def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Запрашивает вектора у ML Backend"""
        if not texts: return []
        try:
            resp = requests.post(
                f"{self.ml_server_url}/text_embed_batch",
                json={"texts": texts, "model_name": self.text_model_name}, timeout=300
            )
            if resp.status_code == 200:
                return resp.json().get("embeddings", [])
        except Exception as e:
            print(f"[Error] Embed batch failed: {e}")
        return []

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Разбивает текст на перекрывающиеся фрагменты"""
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # Ищем пробел, чтобы не резать слова
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start: end = start + chunk_size # Если слово длиннее чанка
            
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            
            start = end - overlap
            if start < 0: start = 0
            if end >= len(text): break
        return chunks

    def index_lessons(self):
        """Индексация текста с сохранением контекста"""
        print(f"\n[Indexer] Индексация ТЕКСТА в '{self.text_collection}'...")
        points = []
        total_chunks = 0
        
        for lesson_name in os.listdir(self.knowledge_base_dir):
            path = os.path.join(self.knowledge_base_dir, lesson_name, "content.txt")
            if not os.path.exists(path): continue
            
            with open(path, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            # Убираем секцию с описаниями кадров (она идет в image collection)
            text_only = full_text.split("FRAME DESCRIPTIONS:")[0].strip()
            
            chunks = self._chunk_text(text_only)
            embeddings = self._get_batch_embeddings(chunks)
            
            for idx, (txt, vec) in enumerate(zip(chunks, embeddings)):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "type": "lesson_chunk",
                        "lesson_name": lesson_name,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "text": txt
                    }
                ))
                
                if len(points) >= 50:
                    self.client.upsert(self.text_collection, points)
                    points = []
            
            total_chunks += len(chunks)
            print(f"   -> {lesson_name}: {len(chunks)} чанков")

        if points: self.client.upsert(self.text_collection, points)
        print(f"[Indexer] Всего текстовых чанков: {total_chunks}")

    def index_images(self):
        """Индексация описаний изображений"""
        print(f"\n[Indexer] Индексация ИЗОБРАЖЕНИЙ в '{self.image_collection}'...")
        points = []
        
        for lesson_name in os.listdir(self.knowledge_base_dir):
            meta_path = os.path.join(self.knowledge_base_dir, lesson_name, "frames_metadata.json")
            if not os.path.exists(meta_path): continue
            
            with open(meta_path, "r", encoding="utf-8") as f:
                frames = json.load(f)
            
            valid_frames = [fr for fr in frames if fr.get("description")]
            if not valid_frames: continue
            
            descriptions = [fr["description"] for fr in valid_frames]
            embeddings = self._get_batch_embeddings(descriptions)
            
            for fr, vec in zip(valid_frames, embeddings):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "type": "frame",
                        "lesson_name": lesson_name,
                        "s3_key": fr.get("frame_path"), # Здесь теперь ключ MinIO
                        "timestamp": fr.get("timestamp"),
                        "description": fr.get("description")
                    }
                ))
                
            if len(points) >= 50:
                self.client.upsert(self.image_collection, points)
                points = []
                
        if points: self.client.upsert(self.image_collection, points)
        print(f"[Indexer] Изображения проиндексированы.")

    def search_text(self, query: str, limit: int = 5):
        return self._search_generic(query, self.text_collection, limit)

    def search_images(self, query: str, limit: int = 5):
        return self._search_generic(query, self.image_collection, limit)

    def _search_generic(self, query: str, collection: str, limit: int):
        # Получаем вектор запроса
        resp = requests.post(f"{self.ml_server_url}/text_embed", 
                           json={"text": query, "model_name": self.text_model_name})
        query_vector = resp.json().get("embedding")
        
        results = self.client.search(collection_name=collection, query_vector=query_vector, limit=limit)
        return [{"score": r.score, **r.payload} for r in results]

    def get_lesson_context(self, lesson_name: str):
        """Возвращает все чанки урока для RAG"""
        results, _ = self.client.scroll(
            collection_name=self.text_collection,
            scroll_filter=Filter(must=[FieldCondition(key="lesson_name", match=MatchValue(value=lesson_name))]),
            limit=1000,
            with_payload=True
        )
        # Сортировка по порядку в тексте
        return sorted([p.payload for p in results], key=lambda x: x['chunk_index'])
