# hybrid_rag_server.py
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import google.generativeai as genai
import uvicorn
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
import traceback

# --- КОНФИГУРАЦИЯ ---
ML_BACKEND_URL = "http://localhost:8000"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge_base"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Allow overriding Gemini model via env; default to a performant model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
api_key = os.getenv("GEMINI_API_KEY")

# Настройка прокси (совместимая логика)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0,::1'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0,::1'

# Создаем сессию БЕЗ прокси для локальных запросов
local_session = requests.Session()
local_session.trust_env = False

app = FastAPI(title="Hybrid RAG Knowledge Base API")

# CORS (фронтенд на другом хосте)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# BM25 кэш
bm25_cache = {
    "index": None,
    "documents": [],
    "metadata": []
}

# --- МОДЕЛИ ДАННЫХ ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    category: Optional[str] = None
    use_hybrid: bool = True
    # Синхронизация с фронтендом: фронтенд присылает `use_entities`
    use_entities: bool = True
    # Для обратной совместимости (если кто-то всё ещё шлёт старое имя):
    use_entity_extraction: Optional[bool] = None
    # Новый параметр: порог схожести (0.0 - 1.0)
    similarity: float = 0.3

class RetrievedDocument(BaseModel):
    text: str
    score: float
    category: str
    lesson: str
    chunk_id: int
    search_method: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[RetrievedDocument]
    query: str
    extracted_entities: Optional[List[str]] = None

# --- Утилиты / core functions ---

def configure_client():
    """Настройка прокси и Gemini API"""
    proxy_url = os.getenv("GEMINI_PROXY", "http://127.0.0.1:12334")
    if proxy_url:
        no_proxy = "localhost,127.0.0.1,0.0.0.0,::1,localhost:8000,localhost:6333,127.0.0.1:8000,127.0.0.1:6333"
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['no_proxy'] = no_proxy
        os.environ['NO_PROXY'] = no_proxy
        print(f"[Gemini] Настроен прокси: {proxy_url}")
    if api_key:
        genai.configure(api_key=api_key, transport="rest")
        print("[Gemini] API ключ настроен")

def get_embedding(text: str) -> list:
    """Получить эмбеддинг через ML backend"""
    try:
        response = local_session.post(
            f"{ML_BACKEND_URL}/text_embed",
            json={"text": text, "model_name": MODEL_NAME},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка векторизации: {str(e)}")

def load_bm25_index() -> Tuple[Optional[BM25Okapi], List[str], List[dict]]:
    """Загрузить все документы из Qdrant и построить BM25 индекс (idempotent)"""
    if bm25_cache["index"] is not None and bm25_cache["documents"]:
        return bm25_cache["index"], bm25_cache["documents"], bm25_cache["metadata"]
    try:
        client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=30,
            prefer_grpc=False
        )
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        total_points = collection_info.points_count
        print(f"[BM25] Загружаем {total_points} документов для BM25 индекса...")
        documents = []
        metadata = []
        offset = None
        while True:
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True
            )
            points, next_offset = results
            if not points:
                break
            for point in points:
                # защитные проверки на наличие полей
                text = point.payload.get("text", "")
                cat = point.payload.get("category", "unknown")
                lesson = point.payload.get("lesson", "unknown")
                chunk_id = point.payload.get("chunk_id", -1)
                documents.append(text)
                metadata.append({
                    "id": point.id,
                    "category": cat,
                    "lesson": lesson,
                    "chunk_id": chunk_id
                })
            if next_offset is None:
                break
            offset = next_offset
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25_index = BM25Okapi(tokenized_docs) if documents else None
        bm25_cache["index"] = bm25_index
        bm25_cache["documents"] = documents
        bm25_cache["metadata"] = metadata
        print(f"[BM25] Индекс построен на {len(documents)} документах")
        return bm25_index, documents, metadata
    except Exception as e:
        print(f"[BM25] Ошибка загрузки: {e}")
        return None, [], []

def extract_entities_and_keywords(query: str) -> List[str]:
    """Извлечь ключевые термины и сущности из запроса через Gemini (если доступен)"""
    if not api_key:
        # Простая экстракция без API
        return [word for word in query.lower().split() if len(word) > 3]
    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        prompt = f"""Извлеки из запроса ключевые термины, концепции и именованные сущности для поиска в базе знаний.
Верни ТОЛЬКО список терминов через запятую, без объяснений.

Запрос: {query}

Термины:"""
        response = model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=200)
        )
        terms = [t.strip() for t in response.text.strip().split(',') if t.strip()]
        terms.extend([word for word in query.lower().split() if len(word) > 2])
        terms = list(dict.fromkeys(terms))  # сохранить порядок, убрать дубликаты
        print(f"[Entity] Извлечено терминов: {terms}")
        return terms
    except Exception as e:
        print(f"[Entity] Ошибка извлечения: {e}")
        return [word for word in query.lower().split() if len(word) > 2]

def bm25_search(query: str, top_k: int = 10, similarity: float = 0.0) -> List[dict]:
    """BM25 текстовый поиск. Применяем нормализацию и фильтр по similarity (0..1)"""
    bm25_index, documents, metadata = load_bm25_index()
    if bm25_index is None:
        return []
    query_tokens = query.lower().split()
    scores = bm25_index.get_scores(query_tokens)
    if len(scores) == 0:
        return []
    # Нормализуем по максимуму
    max_score = float(np.max(scores)) if np.max(scores) > 0 else 1.0
    normalized_scores = scores / max_score if max_score != 0 else scores
    # Выбираем индексы по top_k и фильтру по similarity
    sorted_idx = np.argsort(normalized_scores)[::-1]
    results = []
    count = 0
    for idx in sorted_idx:
        if count >= top_k:
            break
        ns = float(normalized_scores[idx])
        if ns < similarity:
            continue
        if scores[idx] > 0:
            results.append({
                "text": documents[idx],
                "score": float(scores[idx]),
                "normalized_score": ns,
                "category": metadata[idx]["category"],
                "lesson": metadata[idx]["lesson"],
                "chunk_id": metadata[idx]["chunk_id"],
                "search_method": "bm25"
            })
            count += 1
    return results

def vector_search(query: str, top_k: int = 10, category: Optional[str] = None, similarity: float = 0.3) -> List[dict]:
    """Векторный поиск в Qdrant с применением порога score_threshold=similarity"""
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=30, prefer_grpc=False)
        query_vector = get_embedding(query)
        query_filter = None
        if category:
            query_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=similarity  # напрямую используем similarity как порог для скорa
        )
        documents = []
        for point in results.points:
            documents.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "category": point.payload.get("category", "unknown"),
                "lesson": point.payload.get("lesson", "unknown"),
                "chunk_id": point.payload.get("chunk_id", -1),
                "search_method": "vector"
            })
        return documents
    except Exception as e:
        print(f"[Vector] Ошибка поиска: {e}")
        return []

def hybrid_search(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    use_entities: bool = True,
    similarity: float = 0.3
) -> Tuple[List[dict], List[str]]:
    """
    Гибридный поиск: векторный + BM25 с опциональной экстракцией сущностей
    Применяет similarity для фильтрации результатов каждого метода.
    Возвращает (results_list, extracted_terms)
    """
    extracted_terms = extract_entities_and_keywords(query) if use_entities else []
    vector_results = vector_search(query, top_k=top_k * 2, category=category, similarity=similarity)
    bm25_query = query
    if extracted_terms:
        bm25_query = query + " " + " ".join(extracted_terms[:5])
    bm25_results = bm25_search(bm25_query, top_k=top_k * 2, similarity=similarity)

    combined = {}
    # Нормализация vector_results по max в текущем сете
    if vector_results:
        max_vector_score = max((r["score"] for r in vector_results), default=0)
        for r in vector_results:
            key = r["text"][:200]
            normalized = (r["score"] / max_vector_score) if max_vector_score > 0 else 0
            combined.setdefault(key, {**r, "vector_score": normalized * 0.6, "bm25_score": 0, "search_method": r.get("search_method", "vector")})

    if bm25_results:
        max_bm25_score = max((r.get("normalized_score", 0) for r in bm25_results), default=0)
        # note: bm25_results already filtered by similarity and contain normalized_score
        for r in bm25_results:
            key = r["text"][:200]
            normalized = r.get("normalized_score", 0)
            if key in combined:
                combined[key]["bm25_score"] = normalized * 0.4
                combined[key]["search_method"] = "hybrid"
            else:
                combined.setdefault(key, {**r, "vector_score": 0, "bm25_score": normalized * 0.4, "search_method": "bm25"})

    final_results = []
    for v in combined.values():
        v["score"] = v.get("vector_score", 0) + v.get("bm25_score", 0)
        if v.get("search_method") == "hybrid":
            v["score"] *= 1.2
        final_results.append(v)
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:top_k], extracted_terms

def retrieve_documents(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    use_hybrid: bool = True,
    use_entities: bool = True,
    similarity: float = 0.3
) -> Tuple[List[dict], List[str]]:
    """Wrapper to choose retrieval method (passes similarity threshold)"""
    if use_hybrid:
        return hybrid_search(query, top_k=top_k, category=category, use_entities=use_entities, similarity=similarity)
    # vector only (apply similarity)
    return vector_search(query, top_k=top_k, category=category, similarity=similarity), []

def _ensure_citation_markers_in_answer(answer: str, num_sources: int) -> str:
    """
    Постобработка: если модель не вставила ссылок, добавляем [1], [2] в конце абзацев.
    Но стараемся не ломать уже вставленные метки.
    """
    # если уже есть хотя бы одна метка [1], считаем, что модель вставила ссылки
    if "[" in answer and any(f"[{i}]" in answer for i in range(1, num_sources+1)):
        return answer
    # Добавим в конец абзацев метки по очереди
    parts = [p.strip() for p in answer.split("\n\n") if p.strip()]
    if not parts:
        return answer
    out_parts = []
    idx = 1
    for p in parts:
        marker = f" [{idx}]" if idx <= num_sources else ""
        out_parts.append(p + marker)
        if idx < num_sources:
            idx += 1
    return "\n\n".join(out_parts)

def generate_answer(query: str, context_docs: List[dict], extracted_entities: List[str] = None) -> str:
    """
    Генерация ответа через Gemini с инструкцией вставлять ссылки вида [1], [2] внутри текста.
    Источники передаются в порядке context_docs (1..N).
    """
    num_sources = len(context_docs)
    if not api_key:
        # Вернём диагностический ответ, но с цитатами-метками, чтобы фронтенд мог ссылаться
        context_preview = "\n\n".join([
            f"[{i+1}] [{doc['category']} / {doc['lesson']}] (score: {doc.get('score', 0):.2f})\n{doc['text'][:500]}"
            for i, doc in enumerate(context_docs)
        ])
        ans = f"⚠️ Gemini API ключ не настроен. Найдено {len(context_docs)} документов:\n\n{context_preview}"
        # ensure markers
        return _ensure_citation_markers_in_answer(ans, num_sources)

    context = "\n\n---\n\n".join([
        f"Источник {i+1} [{doc['category']} / {doc['lesson']}]:\n{doc['text']}"
        for i, doc in enumerate(context_docs)
    ])
    entities_info = ""
    if extracted_entities:
        entities_info = f"\n\nКлючевые термины из запроса: {', '.join(extracted_entities[:10])}"

    # Инструкция: обязательно использовать цифровые ссылки [1], [2] соответствующие порядку источников
    prompt = f"""Ты — AI ассистент для базы знаний по программированию и ML.
Вопрос: {query}{entities_info}

Контекст (источники пронумерованы в порядке важности):
{context}

Инструкции для ответа:
- Ответь на вопрос, опираясь на информацию из контекста.
- В тексте **обязательно** проставляй ссылки на источники в формате [1], [2] и т.д., где цифра соответствует порядковому номеру источника из переданного контекста.
- Ссылки должны быть вставлены сразу после предложений/фраз, где используется информация из конкретного источника, например: '... это работает так [1].'
- Если информация комбинируется из нескольких источников, можно поставить несколько ссылок: [1][3].
- Если контекст не даёт полного ответа, честно скажи, что информации недостаточно и укажи источники, на которые опирался.
- Пиши на русском, будь понятным и кратким, но информативным.
- Не добавляй внешних URL-адресов в ответ — только цифровые ссылки [n].

Ответ:"""

    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                top_p=0.8,
                max_output_tokens=4096,
            )
        )
        text = response.text.strip()
        # Если модель не вставила метки — добавим постобработкой
        text = _ensure_citation_markers_in_answer(text, num_sources)
        return text
    except Exception as e:
        print(f"[Gemini] Ошибка генерации: {e}\n{traceback.format_exc()}")
        # fallback: показать небольшой превью контекста и гарантировать метки
        context_preview = "\n\n".join([
            f"[{i+1}] {doc['category']} / {doc['lesson']} (score: {doc.get('score',0):.2f})\n{doc['text'][:300]}"
            for i, doc in enumerate(context_docs)
        ])
        err_text = f"⚠️ Ошибка Gemini API: {str(e)}\n\nНайденные источники:\n\n{context_preview}"
        return _ensure_citation_markers_in_answer(err_text, num_sources)

# --- API endpoints ---

@app.get("/categories")
async def get_categories():
    if not bm25_cache["metadata"]:
        load_bm25_index()
    categories = sorted(list({m["category"] for m in bm25_cache["metadata"] if m.get("category")}))
    return {"categories": categories}

@app.get("/health")
async def health():
    """Проверка работоспособности внешних сервисов"""
    status = {
        "ml_backend": False,
        "qdrant": False,
        "gemini_api": bool(api_key),
        "bm25_index": bm25_cache["index"] is not None
    }
    try:
        response = local_session.get(f"{ML_BACKEND_URL}/docs", timeout=2)
        status["ml_backend"] = response.status_code == 200
    except:
        pass
    try:
        response = local_session.get(f"{QDRANT_URL}/collections", timeout=5)
        if response.status_code == 200:
            collections = response.json()
            names = [c["name"] for c in collections.get("result", {}).get("collections", [])]
            status["qdrant"] = COLLECTION_NAME in names
    except:
        pass
    return status

@app.post("/rebuild_bm25")
async def rebuild_bm25():
    """Перестроить BM25 индекс (endpoint)"""
    bm25_cache["index"] = None
    bm25_cache["documents"] = []
    bm25_cache["metadata"] = []
    load_bm25_index()
    return {"status": "ok", "documents_count": len(bm25_cache["documents"])}

@app.on_event("startup")
async def startup_event():
    """Инициализация при старте: настройка клиентов и перестройка индекса"""
    configure_client()

    # Стартовая проверка
    print("\n🔍 Проверка сервисов при старте:")
    try:
        health_status = await health()
        for service, st in health_status.items():
            emoji = "✅" if st else "❌"
            print(f"  {emoji} {service}: {st}")
    except Exception as e:
        print(f"[Startup] Ошибка health check: {e}")

    # Перестроить BM25 при старте (как ты просил)
    print("\n📊 Перестройка BM25 индекса при старте...")
    await rebuild_bm25()
    print("📊 BM25 готов.")

@app.post("/query", response_model=RAGResponse)
async def query_knowledge_base(req: QueryRequest):
    """
    Endpoint для поисковых запросов.
    Поддерживает поля, которые присылает фронтенд:
    - query, top_k, category, use_hybrid, use_entities, similarity
    (также поддерживается старое имя use_entity_extraction для совместимости)
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Пустой запрос")

    # поддержка двух возможных флагов (legacy)
    use_entities_flag = req.use_entities if req.use_entities is not None else bool(req.use_entity_extraction)
    similarity = float(req.similarity) if req.similarity is not None else 0.3
    # ensure bounds
    if similarity < 0.0: similarity = 0.0
    if similarity > 1.0: similarity = 1.0

    # выбор метода поиска
    documents, extracted_entities = retrieve_documents(
        req.query,
        top_k=req.top_k,
        category=req.category,
        use_hybrid=req.use_hybrid,
        use_entities=use_entities_flag,
        similarity=similarity
    )

    if not documents:
        return RAGResponse(
            answer="❌ Не найдено релевантных документов в базе знаний.",
            sources=[],
            query=req.query,
            extracted_entities=extracted_entities if extracted_entities else None
        )

    # Генерация ответа (включая инструкции вставлять [1], [2] в текст)
    answer = generate_answer(req.query, documents, extracted_entities)

    # Формируем источники для фронтенда — **полный текст** нужен для извлечения STEP ID / UPDATED
    sources = [
        RetrievedDocument(
            text=doc["text"],
            score=doc.get("score", 0.0),
            category=doc.get("category", "unknown"),
            lesson=doc.get("lesson", "unknown"),
            chunk_id=doc.get("chunk_id", -1),
            search_method=doc.get("search_method", "unknown")
        )
        for doc in documents
    ]

    return RAGResponse(
        answer=answer,
        sources=sources,
        query=req.query,
        extracted_entities=extracted_entities if extracted_entities else None
    )

@app.get("/")
async def root():
    return {
        "service": "Hybrid RAG Knowledge Base API",
        "features": [
            "Гибридный поиск (Vector + BM25)",
            "Извлечение сущностей через Gemini",
            "Перестройка BM25 при старте",
            "Вставка цифровых ссылок [1],[2] в текст ответа",
            "Порог схожести (similarity) поддержан в запросе"
        ],
        "endpoints": {
            "query": "POST /query - Задать вопрос",
            "health": "GET /health - Проверка сервисов",
            "rebuild_bm25": "POST /rebuild_bm25 - Перестроить BM25 индекс"
        }
    }

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8001))
    print("=" * 70)
    print("🚀 Hybrid RAG Knowledge Base Server")
    print("=" * 70)
    uvicorn.run(app, host=HOST, port=PORT)

