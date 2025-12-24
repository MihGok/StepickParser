import os
import sys
import shutil
from typing import List, Dict

# Добавляем пути импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CourseProcessor.CourseLoader import StepikCourseLoader
from CourseProcessor.CourseParser.CourseParser import CourseAnalyzer
# Импортируем индексатор для векторизации
from CourseProcessor.indexing.qdrant_indexer import QdrantKnowledgeBaseIndexer

# Импорт конфигурации
from services.config import AppConfig

# Сервисы
from services.LLM_Service.llm_service import GeminiService
from services.LLM_Service.schemas import CourseValidationResult
from services.LLM_Service.prompts import COURSE_FILTER_PROMPT_RU

# 1. Список запросов согласно заданию
TARGET_QUERIES = ["Python", "ML", "Мат статистика"]

def filter_courses_with_ai(query: str, raw_courses: List[Dict]) -> List[Dict]:
    """Фильтрация списка курсов через Gemini"""
    if not raw_courses: return []
    
    courses_text_list = []
    for c in raw_courses:
        courses_text_list.append(f"ID: {c['id']}, Title: {c['title']}")
    
    prompt = COURSE_FILTER_PROMPT_RU.format(
        query=query,
        courses_list="\n".join(courses_text_list)
    )
    
    print(f"\n[AI Filter] Анализирую {len(raw_courses)} курсов для запроса '{query}'...")
    try:
        llm = GeminiService()
        result: CourseValidationResult = llm.generate(
            prompt=prompt,
            response_schema=CourseValidationResult,
            temperature=0.1
        )
        
        valid_ids = set(result.relevant_ids)
        filtered = [c for c in raw_courses if c['id'] in valid_ids]
        
        print(f"[AI Filter] Одобрено: {len(filtered)} из {len(raw_courses)}")
        return filtered
    except Exception as e:
        print(f"[AI Filter Error] {e}")
        # Fallback: возвращаем первые 5, если AI сломался
        return raw_courses[:5]

def process_single_query(query: str, loader: StepikCourseLoader, courses_limit: int = 5):
    """Полный цикл обработки для одного поискового запроса"""
    print(f"\n{'#'*60}")
    print(f"🔍 ОБРАБОТКА ЗАПРОСА: '{query}'")
    print(f"{'#'*60}\n")

    # 1. Поиск курсов (ищем с запасом, чтобы было из чего выбирать AI)
    found_ids = loader.get_course_ids_by_query(query=query, limit=20)
    
    raw_courses = []
    for cid in found_ids:
        c_obj = loader.fetch_object_single('courses', cid)
        if c_obj:
            raw_courses.append({'id': c_obj['id'], 'title': c_obj['title']})
            
    if not raw_courses:
        print(f"[STOP] Курсы по запросу '{query}' не найдены.")
        return

    # 2. AI Фильтрация
    best_courses = filter_courses_with_ai(query, raw_courses)
    
    if not best_courses:
        print(f"[STOP] ИИ отклонил все курсы по запросу '{query}'.")
        return

    # 3. Берем ТОП-5 (или меньше, если столько нет)
    target_courses = best_courses[:courses_limit]
    print(f"\n[INFO] Будет загружено курсов: {len(target_courses)}")

    for idx, target_course in enumerate(target_courses, 1):
        print(f"\n--- Обработка курса {idx}/{len(target_courses)}: {target_course['title']} (ID: {target_course['id']}) ---")
        
        # 3.1 Загрузка контента
        full_course_obj = loader.fetch_object_single('courses', target_course['id'])
        loader.process_course(full_course_obj)
        
        # Определяем имя папки курса
        safe_title = loader._sanitize_filename(full_course_obj['title'])
        course_dir_name = f"Course_{target_course['id']}_{safe_title}"
        
        # Фикс имени папки (на случай если loader обрезал длинное имя)
        if not os.path.isdir(course_dir_name):
            possible = [d for d in os.listdir('.') if d.startswith(f"Course_{target_course['id']}")]
            if possible: course_dir_name = possible[0]
        
        # 3.2 Парсинг контента в текстовые файлы (БЕЗ индексации пока что)
        # CourseAnalyzer сохраняет результат в knowledge_base/{query}/...
        analyzer = CourseAnalyzer(course_dir_name, search_query=query)
        analyzer.parse()

    # 4. ИНДЕКСАЦИЯ (Векторизация) для текущего запроса
    # Это выполняется после обработки всех 5 курсов, чтобы собрать общую базу по теме
    print(f"\n>>> 🧠 ЗАПУСК ВЕКТОРИЗАЦИИ (QDRANT) ДЛЯ '{query}'...")
    
    kb_dir_for_query = os.path.join(AppConfig.KNOWLEDGE_BASE_DIR, query)
    
    if os.path.exists(kb_dir_for_query):
        indexer = QdrantKnowledgeBaseIndexer(knowledge_base_dir=kb_dir_for_query)
        
        # Индексация текстов (Вызывает text_embed_batch на бэкенде)
        indexer.index_lessons()
        
        # Индексация картинок (Описания и метаданные)
        indexer.index_images()
        
        print(f">>> ✅ Векторизация для '{query}' завершена.")
    else:
        print(f">>> ⚠️ Папка базы знаний не найдена: {kb_dir_for_query}")

def main():
    loader = StepikCourseLoader()
    
    # Цикл по всем запросам ("Python", "ML", "Мат статистика")
    for query in TARGET_QUERIES:
        try:
            process_single_query(query, loader, courses_limit=2)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Ошибка при обработке запроса '{query}': {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🚀 ВСЕ ЗАДАЧИ ЗАВЕРШЕНЫ")
    print("="*60)

if __name__ == '__main__':
    main()
