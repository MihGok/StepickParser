import os
import re
import json
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from enum import Enum
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import ValidationError

from .GeminiSchemas import (
    BaseGeminiResponse,
    ScreenshotTimestampsResponse,
    KeyConceptsResponse,
    TranslatedTerm,
    CodeAnalysis,
    QuestionsResponse,
    DifficultyAssessment,
    SummaryResponse,
    LearningPathResponse,
    VideoAnalysis,
    CodeReview,
    ContentEnrichment,
    safe_validate
)

from .GeminiPrompts import (
    VideoAnalysisPrompts,
    ConceptExtractionPrompts,
    CodeAnalysisPrompts,
    QuestionGenerationPrompts,
    DifficultyPrompts,
    SummaryPrompts,
    LearningPathPrompts,
    ContentEnrichmentPrompts,
    SystemInstructions
)

load_dotenv()

T = TypeVar('T', bound=BaseGeminiResponse)


class GeminiModelType(str, Enum):
    """Доступные модели Gemini"""
    FLASH = "gemini-2.0-flash-exp"
    FLASH_THINKING = "gemini-2.0-flash-thinking-exp"
    PRO = "gemini-1.5-pro-latest"
    PRO_VISION = "gemini-1.5-pro-vision-latest"


class GeminiClient:
    """
    Универсальный клиент для работы с Gemini API с автоматической валидацией через Pydantic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GeminiModelType.FLASH,
        temperature: float = 0.7,
        max_retries: int = 3,
        system_instruction: Optional[str] = None
    ):
        """
        Args:
            api_key: API ключ Gemini
            model: Название модели
            temperature: Температура генерации (0.0-1.0)
            max_retries: Количество повторных попыток при ошибках
            system_instruction: Системная инструкция по умолчанию
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY не найден")
        
        self.model_name = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.default_system_instruction = system_instruction or SystemInstructions.DEFAULT
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.default_system_instruction,
            generation_config={
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        print(f"[GeminiClient] Инициализирован: {self.model_name}")

    def _parse_json_response(self, text: str) -> Union[Dict, List, None]:
        """Парсинг JSON из ответа с очисткой markdown"""
        text = re.sub(r'```json\s*|\s*```', '', text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[GeminiClient ERROR] JSON parse failed: {e}")
            print(f"[GeminiClient ERROR] Response: {text[:500]}")
            return None

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        retry_count: int = 0
    ) -> Optional[str]:
        """Базовая генерация текста"""
        try:
            if system_instruction:
                temp_model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    generation_config=self.model._generation_config
                )
                response = temp_model.generate_content(prompt)
            else:
                response = self.model.generate_content(prompt)
            
            return response.text.strip()
            
        except Exception as e:
            print(f"[GeminiClient ERROR] {e}")
            if retry_count < self.max_retries:
                print(f"[GeminiClient] Повтор {retry_count + 1}/{self.max_retries}")
                return self.generate_text(prompt, system_instruction, retry_count + 1)
            return None

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None
    ) -> Optional[T]:
        """
        Генерация с автоматической валидацией через Pydantic
        
        Args:
            prompt: Промпт
            schema: Pydantic схема для валидации
            system_instruction: Системная инструкция
            
        Returns:
            Провалидированный объект или None
        """
        enhanced_prompt = f"{prompt}\n\nВерни ТОЛЬКО валидный JSON без markdown."
        text_response = self.generate_text(enhanced_prompt, system_instruction)
        
        if not text_response:
            return None
        
        data = self._parse_json_response(text_response)
        if not data:
            return None
        
        return safe_validate(data, schema)

    # ========================================
    # СПЕЦИАЛИЗИРОВАННЫЕ МЕТОДЫ
    # ========================================

    def analyze_video_transcript_for_screenshots(
        self,
        transcript: str,
        context: Optional[str] = None
    ) -> List[str]:
        """Анализ транскрипта для скриншотов (с валидацией)"""
        prompt = VideoAnalysisPrompts.screenshot_timestamps(transcript, context)
        result = self.generate_structured(
            prompt=prompt,
            schema=ScreenshotTimestampsResponse,
            system_instruction=SystemInstructions.TECHNICAL_CONTENT
        )
        
        if result:
            print(f"[GeminiClient] ✓ Найдено {len(result.timestamps)} меток")
            return result.timestamps
        return []

    def extract_key_concepts(
        self,
        text: str,
        max_concepts: int = 10
    ) -> List[Dict[str, str]]:
        """Извлечение концепций (с валидацией)"""
        prompt = ConceptExtractionPrompts.key_concepts(text, max_concepts)
        result = self.generate_structured(
            prompt=prompt,
            schema=KeyConceptsResponse,
            system_instruction=SystemInstructions.TECHNICAL_CONTENT
        )
        
        if result:
            print(f"[GeminiClient] ✓ Извлечено {len(result.concepts)} концепций")
            return [{"concept": c.concept, "definition": c.definition} for c in result.concepts]
        return []

    def enrich_code_snippet(
        self,
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Анализ кода (с валидацией)"""
        prompt = CodeAnalysisPrompts.enrich_code(code, language)
        result = self.generate_structured(
            prompt=prompt,
            schema=CodeAnalysis,
            system_instruction=SystemInstructions.CODE_ANALYSIS
        )
        
        if result:
            return {
                "explanation": result.explanation,
                "key_points": result.key_points,
                "complexity": result.complexity,
                "potential_issues": result.potential_issues or []
            }
        return {"explanation": "", "key_points": [], "complexity": "unknown"}

    def extract_questions_and_answers(
        self,
        text: str,
        num_questions: int = 5
    ) -> List[Dict[str, str]]:
        """Генерация вопросов (с валидацией)"""
        prompt = QuestionGenerationPrompts.extract_qa(text, num_questions)
        result = self.generate_structured(
            prompt=prompt,
            schema=QuestionsResponse,
            system_instruction=SystemInstructions.TECHNICAL_CONTENT
        )
        
        if result:
            print(f"[GeminiClient] ✓ Сгенерировано {len(result.questions)} вопросов")
            return [
                {
                    "question": q.question,
                    "answer": q.answer,
                    "difficulty": q.difficulty or "medium"
                }
                for q in result.questions
            ]
        return []

    def classify_content_difficulty(
        self,
        text: str,
        domain: str = "Data Science/ML"
    ) -> Dict[str, Any]:
        """Оценка сложности (с валидацией)"""
        prompt = DifficultyPrompts.classify_difficulty(text, domain)
        result = self.generate_structured(
            prompt=prompt,
            schema=DifficultyAssessment,
            system_instruction=SystemInstructions.TECHNICAL_CONTENT
        )
        
        if result:
            return {
                "level": result.level,
                "score": result.score,
                "reasoning": result.reasoning,
                "prerequisites": result.prerequisites or []
            }
        return {"level": "unknown", "score": 0.5, "reasoning": ""}

    def generate_summary(
        self,
        text: str,
        summary_type: str = "brief",
        max_length: int = 200
    ) -> Optional[str]:
        """Генерация резюме (с валидацией)"""
        prompt = SummaryPrompts.generate_summary(text, summary_type, max_length)
        result = self.generate_structured(
            prompt=prompt,
            schema=SummaryResponse
        )
        
        if result:
            return result.summary
        return None

    def generate_learning_path(
        self,
        topic: str,
        current_level: str = "beginner",
        target_level: str = "intermediate",
        goal: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Генерация учебного плана (с валидацией)"""
        prompt = LearningPathPrompts.generate_path(topic, current_level, target_level, goal)
        result = self.generate_structured(
            prompt=prompt,
            schema=LearningPathResponse,
            system_instruction=SystemInstructions.TECHNICAL_CONTENT
        )
        
        if result:
            print(f"[GeminiClient] ✓ План из {len(result.steps)} шагов")
            return [
                {
                    "step": s.step,
                    "title": s.title,
                    "description": s.description,
                    "estimated_hours": s.estimated_hours,
                    "resources": s.resources,
                    "skills_gained": s.skills_gained or []
                }
                for s in result.steps
            ]
        return []

    def translate_technical_term(
        self,
        term: str,
        source_lang: str = "en",
        target_lang: str = "ru",
        with_context: bool = True
    ) -> Dict[str, str]:
        """Перевод термина (с валидацией)"""
        prompt = ConceptExtractionPrompts.translate_term(
            term, source_lang, target_lang, with_context
        )
        result = self.generate_structured(
            prompt=prompt,
            schema=TranslatedTerm
        )
        
        if result:
            return {
                "term": result.translated,
                "context": result.context or ""
            }
        return {"term": term, "context": ""}

    def analyze_video_full(
        self,
        transcript: str
    ) -> Optional[Dict[str, Any]]:
        """Полный анализ видео (с валидацией)"""
        prompt = VideoAnalysisPrompts.full_video_analysis(transcript)
        result = self.generate_structured(
            prompt=prompt,
            schema=VideoAnalysis,
            system_instruction=SystemInstructions.TECHNICAL_CONTENT
        )
        
        if result:
            return {
                "transcript_summary": result.transcript_summary,
                "screenshot_timestamps": result.screenshot_timestamps,
                "key_moments": result.key_moments or [],
                "topics_covered": result.topics_covered or []
            }
        return None

    def review_code(
        self,
        code: str,
        language: str = "python"
    ) -> Optional[Dict[str, Any]]:
        """Код-ревью (с валидацией)"""
        prompt = CodeAnalysisPrompts.code_review(code, language)
        result = self.generate_structured(
            prompt=prompt,
            schema=CodeReview,
            system_instruction=SystemInstructions.CODE_ANALYSIS
        )
        
        if result:
            return {
                "analysis": {
                    "explanation": result.analysis.explanation,
                    "key_points": result.analysis.key_points,
                    "complexity": result.analysis.complexity
                },
                "suggestions": result.suggestions,
                "best_practices": result.best_practices,
                "security_concerns": result.security_concerns or []
            }
        return None

    def enrich_content_full(
        self,
        content: str,
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Комплексное обогащение (с валидацией)"""
        prompt = ContentEnrichmentPrompts.full_enrichment(content, context)
        result = self.generate_structured(
            prompt=prompt,
            schema=ContentEnrichment,
            system_instruction=SystemInstructions.TECHNICAL_CONTENT
        )
        
        if result:
            enriched = {}
            if result.summary:
                enriched["summary"] = result.summary
            if result.key_concepts:
                enriched["key_concepts"] = [
                    {"concept": c.concept, "definition": c.definition}
                    for c in result.key_concepts
                ]
            if result.difficulty:
                enriched["difficulty"] = {
                    "level": result.difficulty.level,
                    "score": result.difficulty.score,
                    "reasoning": result.difficulty.reasoning
                }
            if result.questions:
                enriched["questions"] = [
                    {"question": q.question, "answer": q.answer}
                    for q in result.questions
                ]
            if result.related_topics:
                enriched["related_topics"] = result.related_topics
            
            return enriched
        return None