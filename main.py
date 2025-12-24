import os
import sys
from typing import List, Dict

# Добавляем пути импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from CourseProcessor.CourseLoader import StepikCourseLoader
from CourseProcessor.CourseParser.CourseParser import CourseAnalyzer

# Наши новые сервисы
from services.LLM_Service.llm_service import GeminiService
from services.LLM_Service.schemas import CourseValidationResult
from services.LLM_Service.prompts import COURSE_FILTER_PROMPT_RU

SEARCH_QUERY = "Deep Learning"

def filter_courses_with_ai(query: str, raw_courses: List[Dict]) -> List[Dict]:
    """Фильтрация списка курсов через Gemini"""
    if not raw_courses: return []
    
    # Формируем список для промпта
    courses_text_list = []
    for c in raw_courses:
        courses_text_list.append(f"ID: {c['id']}, Title: {c['title']}")
    
    prompt = COURSE_FILTER_PROMPT_RU.format(
        query=query,
        courses_list="\n".join(courses_text_list)
    )
    
    print(f"\n[AI Filter] Анализирую {len(raw_courses)} курсов...")
    try:
        llm = GeminiService()
        result: CourseValidationResult = llm.generate(
            prompt=prompt,
            response_schema=CourseValidationResult,
            temperature=0.1
        )
        
        # Оставляем только те курсы, чьи ID вернула модель
        valid_ids = set(result.relevant_ids)
        filtered = [c for c in raw_courses if c['id'] in valid_ids]
        
        print(f"[AI Filter] Одобрено: {len(filtered)} из {len(raw_courses)}")
        return filtered
    except Exception as e:
        print(f"[AI Filter Error] {e}")
        return raw_courses[:1] # Fallback: возвращаем хотя бы первый

def main(limit: int = 20):
    print(f"\n{'='*60}")
    print(f"🚀 ЗАПУСК SMART PIPELINE: '{SEARCH_QUERY}'")
    print(f"{'='*60}\n")
    
    loader = StepikCourseLoader()
    
    # 1. Поиск курсов (получаем объекты, а не просто ID)
    # Предполагаем, что loader.get_courses_by_query возвращает список словарей [{'id': 1, 'title': '...'}, ...]
    # Если в CourseLoader только get_course_ids_by_query, нужно немного адаптировать
    found_ids = loader.get_course_ids_by_query(query=SEARCH_QUERY, limit=limit)
    
    raw_courses = []
    for cid in found_ids:
        # Для фильтрации нам нужны названия. Делаем легкий запрос (или берем из кэша поиска если есть)
        c_obj = loader.fetch_object_single('courses', cid)
        if c_obj:
            raw_courses.append({'id': c_obj['id'], 'title': c_obj['title']})
            
    if not raw_courses:
        print("[STOP] Курсы не найдены.")
        return

    # 2. AI Фильтрация
    best_courses = filter_courses_with_ai(SEARCH_QUERY, raw_courses)
    
    if not best_courses:
        print("[STOP] ИИ отклонил все найденные курсы как нерелевантные.")
        return

    target_course = best_courses[0]
    print(f"\n[INFO] Выбран лучший курс: {target_course['title']} (ID: {target_course['id']})")
    
    # 3. Загрузка контента (StepikLoader)
    # Нужно получить полный объект для процессинга
    full_course_obj = loader.fetch_object_single('courses', target_course['id'])
    loader.process_course(full_course_obj)
    
    # Определяем папку
    safe_title = loader._sanitize_filename(full_course_obj['title'])
    course_dir_name = f"Course_{target_course['id']}_{safe_title}"
    
    # Фикс имени папки (если StepikLoader обрезал имя)
    if not os.path.isdir(course_dir_name):
        possible = [d for d in os.listdir('.') if d.startswith(f"Course_{target_course['id']}")]
        if possible: course_dir_name = possible[0]
    
    # 4. Парсинг и Создание Базы Знаний
    print("\n>>> ЗАПУСК ПАРСЕРА И ВАЛИДАЦИИ КОНТЕНТА...")
    analyzer = CourseAnalyzer(course_dir_name, search_query=SEARCH_QUERY)
    results = analyzer.parse()

    print("\n" + "="*60)
    print(f">>> ГОТОВО! В базу знаний добавлено {len(results)} элементов.")
    print("="*60)

if __name__ == '__main__':
    main()
