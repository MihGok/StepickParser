import os
import google.generativeai as genai
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel

# Дженерик для Pydantic моделей
T = TypeVar("T", bound=BaseModel)

class GeminiService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация сервиса.
        :param api_key: Если None, берется из os.environ["GEMINI_API_KEY"]
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or arguments.")
        
        genai.configure(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        response_schema: Type[T],
        model_name: str = "gemini-2.5-flash",
        image_path: Optional[str] = None,
        temperature: float = 0.5,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None
    ) -> T:
        """
        Универсальный метод генерации с валидацией через Pydantic.
        
        :param prompt: Текст запроса.
        :param response_schema: Класс Pydantic, описывающий желаемую структуру ответа.
        :param model_name: Имя модели (gemini-1.5-pro, gemini-1.5-flash и т.д.).
        :param image_path: Путь к изображению (если нужен мультимодальный запрос).
        :param temperature: Креативность (0.0 - строго, 1.0 - креативно).
        :param top_p: Nucleus sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param max_output_tokens: Лимит токенов на выход.
        :param system_instruction: Системный промпт (инструкция "кто ты").
        :return: Экземпляр класса response_schema с заполненными данными.
        """
        
        # 1. Настройка модели
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=response_schema
        )

        # 3. Подготовка контента (текст + опционально картинка)
        content: list[Any] = [prompt]
        
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            uploaded_file = genai.upload_file(image_path)
            content.append(uploaded_file)

        try:
            # 4. Выполнение запроса
            response = model.generate_content(
                content,
                generation_config=generation_config
            )

            # 5. Валидация и парсинг
            # Gemini вернет JSON строку, которая гарантированно соответствует схеме.
            # Мы используем встроенный метод Pydantic для парсинга.
            return response_schema.model_validate_json(response.text)

        except Exception as e:
            print(f"[GeminiService] Error: {e}")
            # Здесь можно добавить логику повторных попыток (retries)
            raise e