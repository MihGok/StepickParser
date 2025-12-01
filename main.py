from CourseProcessor.CourseLoader import StepikCoureLoader
from CourseProcessor.CourseParser.CourseParser import CourseAnalyzer
import os
import sys
SEARCH_QUERY = "Python"


def main():
    try:
        print(f"\n{'='*60}")
        print(f"🚀 ЗАПУСК АВТОМАТИЧЕСКОГО СБОРА: '{SEARCH_QUERY}'")
        print(f"{'='*60}\n")
        loader = StepikCoureLoader()
        found_ids = loader.get_course_ids_by_query(query=SEARCH_QUERY, limit=1)

        if not found_ids:
            print(f"[STOP] По запросу '{SEARCH_QUERY}' курсов не найдено.")
            return

        target_course_id = found_ids[0]
        print(f"\n[INFO] Выбран первый курс из списка: ID {target_course_id}")
        course_obj = loader.fetch_object_single('courses', target_course_id)
        if not course_obj:
            print(f"[ERROR] Не удалось получить данные о курсе {target_course_id}.")
            return

        course_title = course_obj.get('title', 'Без названия')
        print(f"[INFO] Название курса: {course_title}")
        print("\n>>> НАЧИНАЮ ЗАГРУЗКУ КОНТЕНТА...")
        loader.process_course(course_obj)

        safe_title = loader._sanitize_filename(course_title)
        course_dir_name = f"Course_{target_course_id}_{safe_title}"
        
        if not os.path.isdir(course_dir_name):
            possible = [d for d in os.listdir('.') if d.startswith(f"Course_{target_course_id}")]
            if possible:
                course_dir_name = possible[0]
                print(f"[FIX] Найдена альтернативная папка: {course_dir_name}")
            else:
                return

        print("\n" + "="*60)
        print(f">>> ЗАГРУЗКА ЗАВЕРШЕНА. ПАПКА: {course_dir_name}")
        print("="*60 + "\n")


        analyzer = CourseAnalyzer(course_dir_name, search_query=SEARCH_QUERY)
        results = analyzer.parse()

        print("\n" + "="*60)
        print(f">>> ГОТОВО! УСПЕШНО ОБРАБОТАНО.")
        print(f"Всего элементов (шагов): {len(results)}")
        print(f"База знаний: knowledge_base/{SEARCH_QUERY}/...")
        print("="*60)

    except KeyboardInterrupt:
        print("\n[STOP] Работа прервана пользователем.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Произошла критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()