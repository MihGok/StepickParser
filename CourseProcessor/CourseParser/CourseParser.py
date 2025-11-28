import os
import json
from typing import Any, Dict, List, Iterator
from .SectionParser import SectionAnalyzer


class CourseAnalyzer:
    """
    Парсинг курса: обходит секции внутри папки курса и вызывает SectionParser.
    Сохраняет aggregated per-course jsonl (course_steps_texts.jsonl).
    """

    def __init__(self, course_dir: str):
        self.course_dir = course_dir

    def iter_section_dirs(self) -> Iterator[str]:
        if not os.path.isdir(self.course_dir):
            return
        for name in sorted(os.listdir(self.course_dir)):
            full = os.path.join(self.course_dir, name)
            # секции у тебя называются Section_<pos>_<title>
            if os.path.isdir(full) and name.lower().startswith("section_"):
                yield full

    def parse(self) -> List[Dict[str, Any]]:
        all_steps = []
        course_jsonl = os.path.join(self.course_dir, "course_steps_texts.jsonl")
        try:
            if os.path.exists(course_jsonl):
                os.remove(course_jsonl)
        except Exception:
            pass

        for section_dir in self.iter_section_dirs():
            sp = SectionAnalyzer(section_dir)
            parsed = sp.parse()
            # annotate section dir
            for rec in parsed:
                rec_copy = dict(rec)
                rec_copy["section_dir"] = os.path.basename(section_dir)
                all_steps.append(rec_copy)

            # append to course-level jsonl
            try:
                with open(course_jsonl, "a", encoding="utf-8") as cf:
                    for rec in parsed:
                        rec_copy = dict(rec)
                        rec_copy["section_dir"] = os.path.basename(section_dir)
                        cf.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[CourseParser] Could not write to {course_jsonl}: {e}")

        return all_steps
