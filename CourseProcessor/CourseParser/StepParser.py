import re
import html
from typing import Any, Dict, Optional, List
from bs4 import BeautifulSoup

class StepAnalyzer:
    """
    Парсинг отдельного шага. 
    Добавлено: извлечение update_date.
    """

    IGNORE_BLOCK_NAMES = {"choice", "matching", "match", "multi_choice", "multiple_choice", "code"}

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
        
        soup = BeautifulSoup(text, 'html.parser')
        
        # Сохраняем код отдельно
        code_blocks = []
        for code in soup.find_all(['code', 'pre']):
            code_blocks.append(f"\n```\n{code.get_text()}\n```\n")
            code.replace_with(f"__CODE_BLOCK_{len(code_blocks)-1}__")
        
        # Очищаем остальное
        text = soup.get_text(separator='\n')
        
        # Возвращаем код
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)
        
        return text.strip()    

    @staticmethod
    def _pick_min_quality_url(urls: List[Dict[str, Any]]) -> Optional[str]:
        if not urls:
            return None
        numeric_pairs = []
        fallback = []
        for e in urls:
            q = e.get("quality")
            u = e.get("url") or e.get("src") or e.get("link")
            if not u: continue
            if isinstance(q, str):
                m = re.search(r'(\d+)', q)
                if m:
                    try:
                        numeric_pairs.append((int(m.group(1)), u))
                        continue
                    except: pass
            fallback.append(u)

        if numeric_pairs:
            numeric_pairs.sort(key=lambda x: x[0])
            for qv, u in numeric_pairs:
                if qv == 360: return u
            return numeric_pairs[0][1]
        return fallback[-1] if fallback else None

    @classmethod
    def parse_step_dict(cls, step: Dict[str, Any], source_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not isinstance(step, dict):
            return None

        sid = step.get("id")
        pos = step.get("position")
        # Извлекаем дату обновления
        update_date = step.get("update_date", "")

        raw_block = cls._normalize_block(step.get("block"))
        if not raw_block:
            return None

        block_name = (raw_block.get("name") or "").strip().lower()
        if block_name in cls.IGNORE_BLOCK_NAMES:
            return None

        result = {
            "step_id": sid,
            "position": pos,
            "update_date": update_date, # Новое поле
            "block_name": block_name,
            "text": "",
            "video_url": "",
            "transcript": "",
            "source_file": source_file or ""
        }

        if block_name in {"text", "code", "html", "markdown"}:
            raw_text = raw_block.get("text") or ""
            cleaned = cls._clean_html(raw_text)
            if not cleaned: return None
            result["text"] = cleaned
            return result

        if block_name == "video":
            video_obj = raw_block.get("video") or raw_block
            urls = video_obj.get("urls") if isinstance(video_obj, dict) else None
            if isinstance(urls, list) and urls:
                best = cls._pick_min_quality_url(urls)
                if best:
                    result["video_url"] = best
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
