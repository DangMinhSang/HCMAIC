"""Automatic Vietnamese/English handling for CLIP and VQA queries."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache


VIETNAMESE_HINTS = {
    "anh", "bao", "cai", "canh", "cau", "co", "cua", "dang", "duoc", "gi",
    "hay", "hinh", "khong", "mau", "mot", "nguoi", "nhieu", "o", "the", "tim",
    "trong", "tren", "vao", "voi", "xe",
}
WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)
TRAKE_EVENT_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:sự\s*kiện|su\s*kien|event)\s*"
    r"(?P<number>\d+)\s*(?:[—–:]|-|\.)\s*(?P<body>.+?)\s*$",
    flags=re.IGNORECASE,
)
NUMBERED_EVENT_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?P<number>\d+)\s*[.)\-:]\s*(?P<body>.+?)\s*$"
)


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    text_for_model: str
    language: str
    translation_used: bool
    warning: str = ""


def _contains_diacritic(text: str) -> bool:
    decomposed = unicodedata.normalize("NFD", text)
    return any(unicodedata.category(char) == "Mn" for char in decomposed) or "đ" in text.lower()


def looks_vietnamese(text: str) -> bool:
    """Detect Vietnamese without imposing a language-detection model download."""
    lowered = text.lower()
    if _contains_diacritic(lowered):
        return True
    tokens = set(WORD_RE.findall(lowered))
    return len(tokens & VIETNAMESE_HINTS) >= 2


def parse_trake_events(value: str, *, minimum: int = 2, maximum: int = 12) -> list[str]:
    """Extract only the numbered temporal stages from a TRAKE prompt.

    The demo prompt contains an introduction followed by lines such as
    ``Sự kiện 1 — Chuẩn bị: ...``. Treating every non-empty line as an event
    makes the introduction become an extra frame. Numbered event lines are
    therefore authoritative; plain newline-separated text remains supported
    for the compact input format used by the dashboard.
    """
    lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
    explicit: list[tuple[int, str]] = []
    for line in lines:
        match = TRAKE_EVENT_RE.match(line)
        if match:
            body = match.group("body").strip(" \t:-—–")
            if body:
                explicit.append((int(match.group("number")), body))

    if not explicit:
        numbered: list[tuple[int, str]] = []
        for line in lines:
            match = NUMBERED_EVENT_RE.match(line)
            if match:
                body = match.group("body").strip(" \t:-—–")
                if body:
                    numbered.append((int(match.group("number")), body))
        explicit = numbered

    if explicit:
        numbers = [number for number, _body in explicit]
        if len(numbers) != len(set(numbers)):
            raise ValueError("TRAKE có số thứ tự sự kiện bị trùng.")
        events = [body for _number, body in sorted(explicit)]
    else:
        events = [line.lstrip("-*• ").strip() for line in lines]

    if len(events) < minimum:
        raise ValueError(f"TRAKE cần ít nhất {minimum} mốc sự kiện.")
    if len(events) > maximum:
        raise ValueError(f"TRAKE tối đa {maximum} mốc sự kiện.")
    return events


@lru_cache(maxsize=512)
def normalize_query(text: str) -> NormalizedQuery:
    """Return English text for models when a Vietnamese translation is available.

    Translation is deliberately best-effort: a network-disabled Kaggle kernel
    still performs retrieval with the original query instead of failing.
    """
    original = (text or "").strip()
    if not original:
        raise ValueError("Nhập truy vấn.")
    if not looks_vietnamese(original):
        return NormalizedQuery(original, original, "en", False)
    if os.environ.get("AIC_TRANSLATE_VI", "1").lower() in {"0", "false", "no"}:
        return NormalizedQuery(original, original, "vi", False, "Đã tắt dịch tự động.")
    try:
        from deep_translator import GoogleTranslator  # type: ignore

        translated = GoogleTranslator(source="vi", target="en").translate(original)
        if translated and translated.strip():
            return NormalizedQuery(original, translated.strip(), "vi", True)
    except Exception:
        pass
    return NormalizedQuery(
        original,
        original,
        "vi",
        False,
        "Không dịch được tiếng Việt (Internet/API không sẵn sàng); dùng trực tiếp câu gốc.",
    )
