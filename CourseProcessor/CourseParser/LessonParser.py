import os
import json
import re
from typing import Any, Dict, List, Iterator
from .StepParser import StepAnalyzer
from CourseProcessor.TranskriberClient import TranscriberClient


class LessonAnalyzer:
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str, knowledge_base_dir: str):
        self.lesson_dir = lesson_dir
        self.knowledge_base_dir = knowledge_base_dir

    def iter_step_files(self) -> Iterator[str]:
        if not os.path.isdir(self.lesson_dir):
            return
        for fname in sorted(os.listdir(self.lesson_dir)):
            if fname.startswith(self.STEP_FILENAME_PREFIX) and fname.endswith(self.STEP_FILENAME_SUFFIX):
                yield os.path.join(self.lesson_dir, fname)

    def _clean_lesson_title(self, dir_name: str) -> str:
        """
        Превращает 'Lesson_01_Снова возвращаемся к деревьям' 
        в 'Снова возвращаемся к деревьям'.
        Дополнительно чистит от недопустимых символов для пути.
        """
        match = re.search(r'^Lesson_\d+_(.+)$', dir_name, re.IGNORECASE)
        clean_name = match.group(1).strip() if match else dir_name.replace('_', ' ').strip()
        clean_name = re.sub(r'[<>:"/\\|?*]', '', clean_name).strip()
        return clean_name

    def _save_to_knowledge_base(self, parsed_step: Dict[str, Any], raw_lesson_dir_name: str):
        """
        Сохраняет контент шага в структуру: knowledge_base/Query/LessonName/step_ID.txt
        """
        step_id = parsed_step['step_id']
        
        # 1. Получаем чистое имя урока
        clean_lesson_name = self._clean_lesson_title(raw_lesson_dir_name)

        lesson_specific_dir = os.path.join(self.knowledge_base_dir, clean_lesson_name)
        os.makedirs(lesson_specific_dir, exist_ok=True)

        filename = f"step_{step_id}.txt"
        filepath = os.path.join(lesson_specific_dir, filename)

        content_parts = []
        content_parts.append(f"SOURCE_CONTEXT: Курс / {clean_lesson_name} / Step ID: {step_id}")

        if parsed_step.get("update_date"):
            content_parts.append(f"UPDATE_DATE: {parsed_step['update_date']}")
            
        content_parts.append("-" * 20)

        if parsed_step.get("text"):
            content_parts.append("TEXT_CONTENT:")
            content_parts.append(parsed_step["text"])

        if parsed_step.get("transcript"):
            content_parts.append("\nVIDEO_TRANSCRIPT:")
            content_parts.append(parsed_step["transcript"])
            
        full_content = "\n".join(content_parts)

        if len(full_content) < 100: 
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)
        except Exception as e:
            print(f"[KB Error] {e}")

    def parse(self) -> List[Dict[str, Any]]:
        parsed_steps = []
        jsonl_path = os.path.join(self.lesson_dir, "steps_texts.jsonl")
        md_path = os.path.join(self.lesson_dir, "steps_texts.md")

        raw_lesson_dir_name = os.path.basename(self.lesson_dir)

        for p in [jsonl_path, md_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

        for step_file_path in self.iter_step_files():

            try:
                with open(step_file_path, "r", encoding="utf-8") as f:
                    raw_step_obj = json.load(f)
            except Exception as e:
                print(f"[LessonParser] Error reading {step_file_path}: {e}")
                continue

            parsed = StepAnalyzer.parse_step_dict(raw_step_obj, source_file=os.path.basename(step_file_path))
            if not parsed:
                continue

            if parsed.get("video_url") and not parsed.get("transcript"):
                transcript = TranscriberClient.transcribe(parsed["video_url"], parsed["step_id"])
                if transcript:
                    parsed["transcript"] = transcript
                    raw_step_obj["_generated_transcript"] = transcript
                    try:
                        with open(step_file_path, "w", encoding="utf-8") as f:
                            json.dump(raw_step_obj, f, ensure_ascii=False, indent=2)
                    except Exception: pass

            self._save_to_knowledge_base(parsed, raw_lesson_dir_name)

            parsed_steps.append(parsed)

            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(parsed, ensure_ascii=False) + "\n")

            with open(md_path, "a", encoding="utf-8") as mf:
                clean_title = self._clean_lesson_title(raw_lesson_dir_name)
                hdr = f"### {clean_title} (Step {parsed.get('step_id')})\n"
                mf.write(hdr)
                if parsed.get("update_date"):
                     mf.write(f"*Updated: {parsed['update_date']}*\n\n")
                
                if parsed.get("text"):
                    mf.write(parsed["text"] + "\n\n")
                
                if parsed.get("video_url"):
                    mf.write(f"**Video**: {parsed['video_url']}\n")
                    if parsed.get("transcript"):
                        mf.write("\n**Transcript**:\n> " + parsed["transcript"].replace("\n", "\n> ") + "\n\n")
                mf.write("---\n\n")

        return parsed_steps