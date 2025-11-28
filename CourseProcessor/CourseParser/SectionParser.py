import os
import json
from typing import Any, Dict, List, Iterator
from .LessonParser import LessonAnalyzer


class SectionAnalyzer:
    """
    Парсинг секции: обходит папки уроков внутри секции и вызывает LessonParser.
    Сохраняет aggregated per-section jsonl (section_steps_texts.jsonl).
    """

    def __init__(self, section_dir: str):
        self.section_dir = section_dir

    def iter_lesson_dirs(self) -> Iterator[str]:
        if not os.path.isdir(self.section_dir):
            return
        for name in sorted(os.listdir(self.section_dir)):
            full = os.path.join(self.section_dir, name)
            if os.path.isdir(full) and name.lower().startswith("lesson_"):
                yield full

    def parse(self) -> List[Dict[str, Any]]:
        all_steps = []
        section_jsonl = os.path.join(self.section_dir, "section_steps_texts.jsonl")
        try:
            if os.path.exists(section_jsonl):
                os.remove(section_jsonl)
        except Exception:
            pass

        for lesson_dir in self.iter_lesson_dirs():
            lp = LessonAnalyzer(lesson_dir)
            parsed = lp.parse()
            all_steps.extend(parsed)
            # append parsed to section jsonl
            try:
                with open(section_jsonl, "a", encoding="utf-8") as sf:
                    for rec in parsed:
                        # attach lesson path for context
                        rec_copy = dict(rec)
                        rec_copy["lesson_dir"] = os.path.basename(lesson_dir)
                        sf.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[SectionParser] Could not write to {section_jsonl}: {e}")

        return all_steps