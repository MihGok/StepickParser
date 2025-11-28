from CourseProcessor.CourseLoader import StepikCoureLoader
from CourseProcessor.CourseParser.CourseParser import CourseAnalyzer



if __name__ == '__main__':
    try:
        # client = StepikCoureLoader()
        # print("\n" + "="*80)
        # print("ТЕСТИРОВАНИЕ: Попытка доступа к курсу 4852")
        # print("="*80)
        
        # courses = client.fetch_objects('courses', [4852])
        # if courses:
        #     for c in courses:
        #         client.process_course(c)
        #     print('\nГотово!')
        # else:
        #     print('Курс не найден')
        dir = "Course_4852_Введение в Data Science и машинное обучение"
        cp = CourseAnalyzer(dir)
        result = cp.parse()
        print(f"Parsed {len(result)} step entries from course {dir}")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
