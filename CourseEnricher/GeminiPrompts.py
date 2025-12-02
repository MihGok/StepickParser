from typing import Optional, Dict, Any


class PromptTemplate:
    """Базовый класс для шаблонов промптов"""
    
    @staticmethod
    def format(template: str, **kwargs) -> str:
        """Форматирование промпта с параметрами"""
        return template.format(**kwargs)


class VideoAnalysisPrompts:
    """Промпты для анализа видео контента"""
    
    SCREENSHOT_TIMESTAMPS = """Ты анализируешь транскрипт образовательного видео. Твоя задача — определить временные метки, на которых появляются важные визуальные элементы, требующие сохранения в виде скриншотов.

Критерии для выбора временных меток:
1. Демонстрация кода, формул, диаграмм, графиков
2. Визуализация данных, схемы архитектуры
3. Важные слайды с ключевыми концепциями
4. Примеры работы интерфейсов, инструментов
5. Таблицы, сравнения, списки на экране

НЕ включай метки для:
- Простых разговорных фрагментов без визуального контента
- Вступлений и заключений
- Повторений уже показанного материала

{context_section}

Транскрипт с временными метками:
{transcript}

Верни JSON в формате:
{{
  "timestamps": ["HH:MM:SS.mmm", "HH:MM:SS.mmm", ...]
}}

Если визуальный контент не обнаружен, верни {{"timestamps": []}}."""

    @classmethod
    def screenshot_timestamps(cls, transcript: str, context: Optional[str] = None) -> str:
        """Промпт для анализа транскрипта"""
        context_section = f"Контекст: {context}\n" if context else ""
        return cls.SCREENSHOT_TIMESTAMPS.format(
            transcript=transcript,
            context_section=context_section
        )

    FULL_VIDEO_ANALYSIS = """Проанализируй транскрипт видео и предоставь полный анализ.

Транскрипт:
{transcript}

Верни JSON:
{{
  "transcript_summary": "краткое содержание в 2-3 предложениях",
  "screenshot_timestamps": ["HH:MM:SS.mmm", ...],
  "key_moments": [
    {{"timestamp": "HH:MM:SS.mmm", "description": "что происходит"}}
  ],
  "topics_covered": ["тема 1", "тема 2", ...]
}}"""

    @classmethod
    def full_video_analysis(cls, transcript: str) -> str:
        return cls.FULL_VIDEO_ANALYSIS.format(transcript=transcript)


class ConceptExtractionPrompts:
    """Промпты для извлечения концепций и терминов"""
    
    KEY_CONCEPTS = """Проанализируй следующий образовательный текст и извлеки до {max_concepts} ключевых концепций.

Для каждой концепции укажи:
- concept: краткое название концепции (1-5 слов)
- definition: чёткое определение в 1-2 предложениях

Сфокусируйся на ТЕХНИЧЕСКИХ терминах и ОСНОВНЫХ идеях. Не включай общие слова.

Текст:
{text}

Верни JSON:
{{
  "concepts": [
    {{"concept": "Название", "definition": "Определение"}},
    ...
  ]
}}"""

    @classmethod
    def key_concepts(cls, text: str, max_concepts: int = 10) -> str:
        return cls.KEY_CONCEPTS.format(text=text, max_concepts=max_concepts)

    TRANSLATE_TERM = """Переведи технический термин "{term}" с {source_lang} на {target_lang}.

{context_instruction}

Верни JSON:
{{
  "original": "{term}",
  "translated": "перевод термина",
  "context": "краткий контекст использования в IT/DS области (2-3 предложения)"
}}"""

    @classmethod
    def translate_term(
        cls,
        term: str,
        source_lang: str = "en",
        target_lang: str = "ru",
        with_context: bool = True
    ) -> str:
        context_instruction = (
            "Дополнительно объясни контекст использования термина в IT/Data Science."
            if with_context else ""
        )
        return cls.TRANSLATE_TERM.format(
            term=term,
            source_lang=source_lang,
            target_lang=target_lang,
            context_instruction=context_instruction
        )


class CodeAnalysisPrompts:
    """Промпты для анализа кода"""
    
    ENRICH_CODE = """Проанализируй следующий {language} код и предоставь детальный анализ.

Код:
```{language}
{code}
```

Верни JSON:
{{
  "explanation": "общее объяснение что делает код (2-3 предложения)",
  "key_points": ["важный момент 1", "важный момент 2", ...],
  "complexity": "beginner/intermediate/advanced",
  "language": "{language}",
  "potential_issues": ["потенциальная проблема 1", ...]
}}

В key_points включи:
- Используемые паттерны и концепции
- Важные библиотеки/функции
- Алгоритмическую сложность (если применимо)

В potential_issues укажи:
- Возможные баги или edge cases
- Предложения по оптимизации
- Проблемы читаемости"""

    @classmethod
    def enrich_code(cls, code: str, language: str = "python") -> str:
        return cls.ENRICH_CODE.format(code=code, language=language)

    CODE_REVIEW = """Проведи код-ревью следующего {language} кода.

Код:
```{language}
{code}
```

Верни JSON:
{{
  "analysis": {{
    "explanation": "...",
    "key_points": [...],
    "complexity": "...",
    "language": "{language}"
  }},
  "suggestions": ["улучшение 1", "улучшение 2", ...],
  "best_practices": ["применимая практика 1", ...],
  "security_concerns": ["проблема безопасности 1", ...]
}}"""

    @classmethod
    def code_review(cls, code: str, language: str = "python") -> str:
        return cls.CODE_REVIEW.format(code=code, language=language)


class QuestionGenerationPrompts:
    """Промпты для генерации вопросов"""
    
    EXTRACT_QA = """На основе следующего учебного материала сгенерируй {num_questions} вопросов для проверки понимания с ответами.

Требования к вопросам:
- Проверяют понимание КЛЮЧЕВЫХ концепций (не мелких деталей)
- Формулировка четкая и конкретная
- Ответы однозначные и проверяемые
- Разная сложность: от базовых до более глубоких

Текст:
{text}

Верни JSON:
{{
  "questions": [
    {{
      "question": "Вопрос?",
      "answer": "Развернутый ответ",
      "difficulty": "easy/medium/hard"
    }},
    ...
  ]
}}"""

    @classmethod
    def extract_qa(cls, text: str, num_questions: int = 5) -> str:
        return cls.EXTRACT_QA.format(text=text, num_questions=num_questions)


class DifficultyPrompts:
    """Промпты для оценки сложности контента"""
    
    CLASSIFY_DIFFICULTY = """Оцени сложность следующего учебного материала для студента изучающего {domain}.

Критерии оценки:
- Количество предварительных знаний
- Абстрактность концепций
- Технический жаргон
- Сложность примеров

Текст:
{text}

Верни JSON:
{{
  "level": "beginner/intermediate/advanced",
  "score": 0.0-1.0,
  "reasoning": "обоснование оценки (2-3 предложения)",
  "prerequisites": ["необходимое знание 1", "необходимое знание 2", ...]
}}

Где score:
- 0.0-0.3: beginner (базовые концепции, минимум терминов)
- 0.4-0.7: intermediate (требует предварительных знаний)
- 0.8-1.0: advanced (глубокие концепции, специализированные знания)"""

    @classmethod
    def classify_difficulty(cls, text: str, domain: str = "Data Science/ML") -> str:
        return cls.CLASSIFY_DIFFICULTY.format(text=text, domain=domain)


class SummaryPrompts:
    """Промпты для генерации резюме"""
    
    SUMMARY_TYPES = {
        "brief": "Создай краткое резюме в 2-3 предложениях, сфокусировавшись на главной идее.",
        "detailed": "Создай подробное резюме, охватывающее все основные моменты и ключевые детали.",
        "bullet_points": "Создай резюме в виде списка основных пунктов (используй дефисы для маркеров)."
    }
    
    GENERATE_SUMMARY = """Задача: {instruction}
Максимальная длина: {max_length} слов.

Текст:
{text}

Верни JSON:
{{
  "summary": "текст резюме",
  "summary_type": "{summary_type}",
  "word_count": примерное количество слов
}}"""

    @classmethod
    def generate_summary(
        cls,
        text: str,
        summary_type: str = "brief",
        max_length: int = 200
    ) -> str:
        instruction = cls.SUMMARY_TYPES.get(summary_type, cls.SUMMARY_TYPES["brief"])
        return cls.GENERATE_SUMMARY.format(
            instruction=instruction,
            max_length=max_length,
            text=text,
            summary_type=summary_type
        )


class LearningPathPrompts:
    """Промпты для генерации учебных планов"""
    
    GENERATE_PATH = """Создай пошаговый учебный план по теме "{topic}".

Параметры:
- Текущий уровень: {current_level}
- Целевой уровень: {target_level}
{goal_section}

Требования к плану:
- 5-7 последовательных шагов
- Реалистичные временные оценки
- Конкретные навыки на каждом этапе
- Разнообразие типов ресурсов (видео, статьи, практика)

Верни JSON:
{{
  "topic": "{topic}",
  "current_level": "{current_level}",
  "target_level": "{target_level}",
  "total_estimated_hours": сумма всех estimated_hours,
  "steps": [
    {{
      "step": 1,
      "title": "Название этапа",
      "description": "Детальное описание что изучать (2-3 предложения)",
      "estimated_hours": число,
      "resources": ["тип ресурса 1", "тип ресурса 2"],
      "skills_gained": ["навык 1", "навык 2"]
    }},
    ...
  ]
}}"""

    @classmethod
    def generate_path(
        cls,
        topic: str,
        current_level: str = "beginner",
        target_level: str = "intermediate",
        goal: Optional[str] = None
    ) -> str:
        goal_section = f"- Цель: {goal}" if goal else ""
        return cls.GENERATE_PATH.format(
            topic=topic,
            current_level=current_level,
            target_level=target_level,
            goal_section=goal_section
        )


class ContentEnrichmentPrompts:
    """Промпты для комплексного обогащения контента"""
    
    FULL_ENRICHMENT = """Проведи полное обогащение следующего учебного материала.

Материал:
{content}

Контекст: {context}

Выполни:
1. Создай краткое резюме (2-3 предложения)
2. Извлеки до 5 ключевых концепций с определениями
3. Оцени сложность материала
4. Сгенерируй 3 вопроса для проверки понимания
5. Определи связанные темы для дальнейшего изучения

Верни JSON:
{{
  "summary": "краткое содержание",
  "key_concepts": [
    {{"concept": "...", "definition": "..."}}
  ],
  "difficulty": {{
    "level": "beginner/intermediate/advanced",
    "score": 0.0-1.0,
    "reasoning": "...",
    "prerequisites": [...]
  }},
  "questions": [
    {{"question": "...", "answer": "...", "difficulty": "..."}}
  ],
  "related_topics": ["тема 1", "тема 2", ...]
}}"""

    @classmethod
    def full_enrichment(cls, content: str, context: Optional[str] = None) -> str:
        context_text = context or "Общий образовательный контент"
        return cls.FULL_ENRICHMENT.format(content=content, context=context_text)


class SystemInstructions:
    """Системные инструкции для Gemini"""
    
    DEFAULT = """Ты — AI-ассистент для обработки образовательного контента.

Твои задачи:
- Анализировать учебные материалы
- Извлекать ключевую информацию
- Генерировать структурированные ответы в JSON формате
- Помогать в создании базы знаний

Принципы работы:
- Точность: давай проверяемые и корректные ответы
- Структурированность: всегда возвращай валидный JSON
- Релевантность: фокусируйся на технических деталях
- Краткость: избегай воды, будь конкретен

ВАЖНО: Всегда возвращай ТОЛЬКО валидный JSON без markdown форматирования."""

    TECHNICAL_CONTENT = """Ты — эксперт в анализе технического образовательного контента в области Computer Science, Data Science и Machine Learning.

Специализация:
- Распознавание концепций программирования
- Анализ алгоритмов и структур данных
- Оценка сложности технических материалов
- Извлечение ключевых паттернов и практик

Всегда возвращай ТОЛЬКО валидный JSON."""

    CODE_ANALYSIS = """Ты — опытный код-ревьюер и преподаватель программирования.

Фокус на:
- Качество и читаемость кода
- Выявление потенциальных проблем
- Объяснение концепций простым языком
- Практические рекомендации по улучшению

Возвращай только валидный JSON."""


def get_prompt(prompt_class: type, method_name: str, **kwargs) -> str:
    """
    Универсальный геттер промптов
    
    Args:
        prompt_class: Класс с промптами (например, VideoAnalysisPrompts)
        method_name: Название метода промпта
        **kwargs: Параметры для промпта
        
    Returns:
        Отформатированный промпт
    """
    method = getattr(prompt_class, method_name, None)
    if method and callable(method):
        return method(**kwargs)
    raise ValueError(f"Промпт {prompt_class.__name__}.{method_name} не найден")


def list_available_prompts() -> Dict[str, list]:
    """Возвращает список всех доступных промптов"""
    return {
        "VideoAnalysis": [m for m in dir(VideoAnalysisPrompts) if not m.startswith('_')],
        "ConceptExtraction": [m for m in dir(ConceptExtractionPrompts) if not m.startswith('_')],
        "CodeAnalysis": [m for m in dir(CodeAnalysisPrompts) if not m.startswith('_')],
        "QuestionGeneration": [m for m in dir(QuestionGenerationPrompts) if not m.startswith('_')],
        "Difficulty": [m for m in dir(DifficultyPrompts) if not m.startswith('_')],
        "Summary": [m for m in dir(SummaryPrompts) if not m.startswith('_')],
        "LearningPath": [m for m in dir(LearningPathPrompts) if not m.startswith('_')],
        "ContentEnrichment": [m for m in dir(ContentEnrichmentPrompts) if not m.startswith('_')],
    }