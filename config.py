import os
from typing import Dict, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ComplexityLevel = Literal["easy", "hard"]

class GeminiConfig(BaseSettings):
    api_key: str = Field(..., alias="GEMINI_API_KEY")

    model_routing: Dict[str, str] = Field(
        default={
            "easy": "gemini-2.5-flash",   # Или "gemini-1.5-flash"
            "hard": "gemini-2.5-pro-latest"   # Мощная модель
        },
        alias="GEMINI_MODEL_ROUTING"
    )
    
    # Настройки генерации
    temperature: float = 0.3
    max_output_tokens: int = 8192
    
    # Лимиты ввода (защита от переполнения контекста)
    max_input_chars_basic: int = Field(100000, alias="GEMINI_MAX_INPUT_CHARS_BASIC")
    max_input_chars_advanced: int = Field(10000, alias="GEMINI_MAX_INPUT_CHARS_ADVANCED")
    max_retries: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

class AppConfig(BaseSettings):
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    default_search_query: str = "Python"

settings = AppConfig()