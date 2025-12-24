import os
import json
import re
import cv2
import shutil
import numpy as np
import requests
import google.generativeai as genai
from typing import Any, Dict, List, Iterator, Optional, Tuple

from .StepParser import StepAnalyzer
from ..client_api import Client
from services.storage_service import StorageService

# Импорт сервисов LLM
from services.LLM_Service.llm_service import GeminiService
from services.LLM_Service.schemas import VideoAnalysisResult
from services.LLM_Service.prompts import ANALYZE_TRANSCRIPT_PROMPT_RU

class LessonAnalyzer:
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str, knowledge_base_dir: str):
        self.lesson_dir = lesson_dir
        self.knowledge_base_dir = knowledge_base_dir
        self.storage = StorageService()
        self.llm_service = GeminiService() # Используем наш унифицированный сервис

    # ... (методы iter_step_files, _clean_lesson_title без изменений) ...
    def iter_step_files(self) -> Iterator[str]:
        if not os.path.isdir(self.lesson_dir): return
        for fname in sorted(os.listdir(self.lesson_dir)):
            if fname.startswith(self.STEP_FILENAME_PREFIX) and fname.endswith(self.STEP_FILENAME_SUFFIX):
                yield os.path.join(self.lesson_dir, fname)

    def _clean_lesson_title(self, dir_name: str) -> str:
        match = re.search(r'^Lesson_\d+_(.+)$', dir_name, re.IGNORECASE)
        clean_name = match.group(1).strip() if match else dir_name.replace('_', ' ').strip()
        return re.sub(r'[<>:"/\\|?*]', '', clean_name).strip()

    def _get_analysis_from_gemini(self, transcript: str, duration: float) -> Optional[VideoAnalysisResult]:
        """Получает таймкоды и 'reason' (ожидаемое описание)"""
        if not transcript: return None
        
        prompt = ANALYZE_TRANSCRIPT_PROMPT_RU.format(
            duration=duration,
            transcript=transcript[:50000]
        )
        
        try:
            return self.llm_service.generate(
                prompt=prompt,
                response_schema=VideoAnalysisResult,
                model_name="gemini-1.5-flash", # Или 2.0-flash, если доступна
                temperature=0.2
            )
        except Exception as e:
            print(f"   [Gemini Error] {e}")
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Считает косинусную схожесть двух текстов через TextEncoder (MPNet)"""
        if not text1 or not text2: return 0.0
        
        # Получаем вектора через ML Backend (Client API)
        vec1 = Client.get_text_embedding(text1)
        vec2 = Client.get_text_embedding(text2)
        
        if not vec1 or not vec2: return 0.0
        
        # Cosine similarity
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _process_video(self, video_url: str, transcript: str, step_id: int, lesson_name: str) -> List[Dict]:
        """Новая логика: 5 скриншотов -> LLaVA -> Сравнение с Reason -> Выбор лучшего"""
        temp_dir = os.path.join("temp_frames", str(step_id))
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, "video.mp4")
        final_frames_data = []

        try:
            # 1. Скачивание
            print(f"   [Video] Скачивание...")
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                with open(video_path, 'wb') as f:
                    for chunk in r.iter_content(8192): f.write(chunk)

            # 2. Метаданные видео
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames_count / fps if fps > 0 else 0
            
            # 3. Gemini Анализ (получаем timestamp и reason)
            analysis = self._get_analysis_from_gemini(transcript, duration)
            if not analysis:
                print("   [Skip] Gemini не вернул таймкоды.")
                return []

            timestamps_tasks = analysis.timestamps
            print(f"   [Video] Проверка {len(timestamps_tasks)} смысловых блоков...")

            # 4. Цикл по смысловым блокам
            for task in timestamps_tasks:
                start_time = task.timestamp
                expected_reason = task.reason
                
                candidates = []

                for offset in [0, 3, 6, 9, 12]:
                    current_ts = start_time + offset
                    if current_ts > duration: break
                    
                    cap.set(cv2.CAP_PROP_POS_MSEC, current_ts * 1000)
                    ret, frame = cap.read()
                    if not ret: continue
                    
                    # Сохраняем кандидат
                    cand_path = os.path.join(temp_dir, f"cand_{int(current_ts)}.jpg")
                    cv2.imwrite(cand_path, frame)
                    
                    # Получаем описание LLaVA (Реальность)
                    llava_desc = Client.get_image_description(cand_path)
                    if not llava_desc: continue
                    
                    # Считаем схожесть (Ожидание vs Реальность)
                    score = self._calculate_similarity(expected_reason, llava_desc)
                    print(f"      t={current_ts:.1f}s | Score: {score:.3f} | LLaVA: {llava_desc[:30]}...")
                    
                    if score > 0.60:
                        candidates.append({
                            "score": score,
                            "path": cand_path,
                            "desc": llava_desc,
                            "ts": current_ts
                        })
                
                # Выбираем победителя
                if candidates:
                    # Сортируем по убыванию score
                    best_cand = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
                    print(f"   [Match] Выбран кадр {best_cand['ts']:.1f}s (Score: {best_cand['score']:.3f})")
                    
                    # Загрузка в MinIO
                    s3_key = f"{lesson_name}/step_{step_id}/{int(best_cand['ts'])}.jpg"
                    if self.storage.upload_frame(best_cand["path"], s3_key):
                        final_frames_data.append({
                            "timestamp": best_cand["ts"],
                            "frame_path": s3_key,
                            "description": best_cand["desc"],
                            "validation_score": best_cand["score"],
                            "expected_reason": expected_reason
                        })
                else:
                    print(f"   [Fail] Ни один кадр не прошел порог 0.60 для: '{expected_reason}'")

            cap.release()
            
        except Exception as e:
            print(f"   [Video Error] {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return final_frames_data

    def _save_lesson_content(self, all_parsed_steps: List[Dict], lesson_name: str):
        """
        Сохраняет ВЕСЬ урок в один файл content.txt
        """
        lesson_dir = os.path.join(self.knowledge_base_dir, lesson_name)
        os.makedirs(lesson_dir, exist_ok=True)
        filepath = os.path.join(lesson_dir, "content.txt")

        parts = [f"LESSON: {lesson_name}", "="*50]

        for step in all_parsed_steps:
            parts.append(f"\nSTEP ID: {step['step_id']}")
            if step.get('update_date'): parts.append(f"UPDATED: {step['update_date']}")
            parts.append("-" * 20)
            
            if step.get("text"):
                parts.append(step["text"])
            
            if step.get("transcript"):
                parts.append("\n[TRANSCRIPT]:")
                parts.append(step["transcript"])

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            print(f"   [KB] Сохранен текст урока: {filepath}")
        except Exception as e:
            print(f"[KB Error] {e}")

    def _save_frames_metadata(self, all_frames: List[Dict], lesson_name: str):
        """Сохраняет метаданные картинок для индексации"""
        if not all_frames: return
        
        path = os.path.join(self.knowledge_base_dir, lesson_name, "frames_metadata.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(all_frames, f, ensure_ascii=False, indent=2)
            print(f"   [KB] Сохранено {len(all_frames)} кадров в метаданные.")
        except Exception as e:
            print(f"[KB Error] {e}")

    def parse(self) -> List[Dict[str, Any]]:
        parsed_steps = []
        all_frames_metadata = []
        
        raw_lesson_dir_name = os.path.basename(self.lesson_dir)
        clean_name = self._clean_lesson_title(raw_lesson_dir_name)
        
        print(f"\n[Lesson] Обработка: {clean_name}")

        for step_file in self.iter_step_files():
            try:
                with open(step_file, "r", encoding="utf-8") as f:
                    raw_step = json.load(f)
            except: continue

            # 1. Базовый парсинг
            parsed = StepAnalyzer.parse_step_dict(raw_step, os.path.basename(step_file))
            if not parsed: continue

            # 2. Транскрибация (Whisper)
            if parsed.get("video_url") and not parsed.get("transcript"):
                transcript = Client.transcribe(parsed["video_url"], parsed["step_id"])
                if transcript:
                    parsed["transcript"] = transcript
                    # Кешируем обратно в JSON шагa
                    raw_step["_generated_transcript"] = transcript
                    try:
                        with open(step_file, "w", encoding="utf-8") as f:
                            json.dump(raw_step, f, ensure_ascii=False, indent=2)
                    except: pass

            # 3. Обработка Видео (Gemini -> OpenCV -> MinIO -> LLaVA)
            if parsed.get("video_url") and parsed.get("transcript"):
                frames = self._process_video(
                    parsed["video_url"], 
                    parsed["transcript"], 
                    parsed["step_id"], 
                    clean_name
                )
                if frames:
                    # Добавляем в общий список кадров урока
                    for fr in frames:
                        fr['step_id'] = parsed['step_id']
                        fr['lesson_name'] = clean_name
                        # Удаляем вектор перед сохранением в JSON (он не нужен там)
                        if 'clip_vector' in fr: del fr['clip_vector']
                    
                    all_frames_metadata.extend(frames)

            parsed_steps.append(parsed)

        # 4. Финальное сохранение (Агрегация)
        self._save_lesson_content(parsed_steps, clean_name)
        self._save_frames_metadata(all_frames_metadata, clean_name)
        
        return parsed_steps
