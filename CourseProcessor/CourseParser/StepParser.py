import re
import html
from typing import Any, Dict, Optional, List, Iterator


class StepAnalyzer:
    """
    Парсинг отдельного шага (step JSON).
    Правила:
      - Игнорировать блоки с именами choice/matching (и их вариации).
      - Для text/code/html/markdown сохраняем очищенный от HTML текст (block['text']).
      - Для video сохраняем минимальную ссылку (360p если есть, иначе наименьшее числовое качество)
        в поле 'video_url' и добавляем пустое поле 'transcript'.
      - Для других блоков: если есть поле text — возвращаем его очищенным, иначе игнорируем.
      - Если block — список, берём первый элемент.
    Возвращает словарь с полями:
        { "step_id", "position", "block_name", "text", "video_url", "transcript", "source_file" }
    или None, если блок игнорируется / неполезен.
    """

    IGNORE_BLOCK_NAMES = {"choice", "matching", "match", "multi_choice", "multiple_choice"}

    @staticmethod
    def _normalize_block(block_like: Any) -> Optional[Dict[str, Any]]:
        if block_like is None:
            return None
        if isinstance(block_like, list):
            return block_like[0] if block_like else None
        if isinstance(block_like, dict):
            return block_like
        return None

    @staticmethod
    def _clean_html(text: Optional[str]) -> str:
        if not text:
            return ""
        text = html.unescape(text)
        text = re.sub(r'(?i)<br\s*/?>', '\n', text)
        text = re.sub(r'<[^>]+>', '', text)
        lines = [re.sub(r'[ \t\f\v]+', ' ', ln).strip() for ln in text.splitlines()]
        cleaned_lines = []
        prev_empty = False
        for ln in lines:
            if ln == "":
                if not prev_empty:
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(ln)
                prev_empty = False
        while cleaned_lines and cleaned_lines[0] == "":
            cleaned_lines.pop(0)
        while cleaned_lines and cleaned_lines[-1] == "":
            cleaned_lines.pop()
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _pick_min_quality_url(urls: List[Dict[str, Any]]) -> Optional[str]:
        if not urls:
            return None
        numeric_pairs = []
        fallback = []
        for e in urls:
            q = e.get("quality")
            u = e.get("url") or e.get("src") or e.get("link")
            if not u:
                continue
            if isinstance(q, str):
                m = re.search(r'(\d+)', q)
                if m:
                    try:
                        numeric_pairs.append((int(m.group(1)), u))
                        continue
                    except Exception:
                        pass
            fallback.append(u)

        if numeric_pairs:
            numeric_pairs.sort(key=lambda x: x[0])
            for qv, u in numeric_pairs:
                if qv == 360:
                    return u
            return numeric_pairs[0][1]

        if fallback:
            return fallback[-1]
        return None

    @classmethod
    def parse_step_dict(cls, step: Dict[str, Any], source_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not isinstance(step, dict):
            return None

        sid = step.get("id")
        pos = step.get("position")

        raw_block = cls._normalize_block(step.get("block"))
        if not raw_block:
            return None

        block_name = (raw_block.get("name") or "").strip().lower()
        if block_name in cls.IGNORE_BLOCK_NAMES:
            return None

        result = {
            "step_id": sid,
            "position": pos,
            "block_name": block_name,
            "text": "",
            "video_url": "",
            "transcript": "",
            "source_file": source_file or ""
        }

        if block_name in {"text", "code", "html", "markdown"}:
            raw_text = raw_block.get("text") or ""
            cleaned = cls._clean_html(raw_text)
            if not cleaned:
                return None
            result["text"] = cleaned
            return result

        if block_name == "video":
            video_obj = raw_block.get("video") or raw_block
            urls = video_obj.get("urls") if isinstance(video_obj, dict) else None
            if isinstance(urls, list) and urls:
                best = cls._pick_min_quality_url(urls)
                if best:
                    result["video_url"] = best
                    result["transcript"] = ""
                    raw_text = raw_block.get("text") or ""
                    result["text"] = cls._clean_html(raw_text)
                    return result
            return None

        fallback_text = raw_block.get("text")
        if fallback_text:
            cleaned = cls._clean_html(fallback_text)
            if cleaned:
                result["text"] = cleaned
                return result

        return None
