import os
import json
from typing import Any, Dict, List, Iterator
from .StepParser import StepAnalyzer


class LessonAnalyzer:
    """
    Парсинг урока: читает файлы step_*.json в папке урока и собирает полезные поля через StepParser.
    Сохраняет агрегированный jsonl (steps_texts.jsonl) и per-lesson markdown (steps_texts.md).
    """
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str):
        self.lesson_dir = lesson_dir

    def iter_step_files(self) -> Iterator[str]:
        if not os.path.isdir(self.lesson_dir):
            return
        for fname in sorted(os.listdir(self.lesson_dir)):
            if fname.startswith(self.STEP_FILENAME_PREFIX) and fname.endswith(self.STEP_FILENAME_SUFFIX):
                yield os.path.join(self.lesson_dir, fname)

    def parse(self) -> List[Dict[str, Any]]:
        parsed_steps = []
        jsonl_path = os.path.join(self.lesson_dir, "steps_texts.jsonl")
        md_path = os.path.join(self.lesson_dir, "steps_texts.md")

        # ensure we start fresh (append-safe could be implemented, but here overwrite)
        try:
            if os.path.exists(jsonl_path):
                os.remove(jsonl_path)
        except Exception:
            pass
        try:
            if os.path.exists(md_path):
                os.remove(md_path)
        except Exception:
            pass

        for step_file in self.iter_step_files():
            try:
                with open(step_file, "r", encoding="utf-8") as f:
                    step_obj = json.load(f)
            except Exception as e:
                print(f"[LessonParser] Failed to load {step_file}: {e}")
                continue

            parsed = StepAnalyzer.parse_step_dict(step_obj, source_file=os.path.basename(step_file))
            if not parsed:
                continue

            parsed_steps.append(parsed)
            # write to jsonl
            try:
                with open(jsonl_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[LessonParser] Could not write to {jsonl_path}: {e}")

            # write to markdown
            try:
                with open(md_path, "a", encoding="utf-8") as mf:
                    hdr = f"### {parsed['source_file']} (id={parsed.get('step_id')}, pos={parsed.get('position')})\n\n"
                    mf.write(hdr)
                    if parsed.get("text"):
                        mf.write(parsed["text"] + "\n\n")
                    if parsed.get("video_url"):
                        mf.write(f"Video (min quality): {parsed['video_url']}\n\n")
                    mf.write("---\n\n")
            except Exception as e:
                print(f"[LessonParser] Could not write to {md_path}: {e}")

        return parsed_steps