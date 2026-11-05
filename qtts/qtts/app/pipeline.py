"""Разбиение текста на чанки для псевдо-стриминга.

Qwen3-TTS генерирует речь авторегрессионно и декодирует кодек-кадры пачкой, поэтому
«настоящего» потокового вывода из коробки нет. Самый быстрый способ получить низкую
задержку первого звука на RX 9060 XT — разрезать текст на предложения/абзацы,
генерировать их по очереди и отдавать аудио сразу, как только чанк готов
(первый звук приходит через ~1 чанк, а не через весь текст).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

# Сокращения и цифры, после которых точка НЕ заканчивает предложение.
_ABBREV = {
    "т.д", "т.п", "т.к", "др", "г", "гг", "см", "ср", "ул", "пр", "руб", "коп",
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "no", "vol", "fig", "inc", "ltd", "co", "approx",
}

_SENT_END = "!?…"
_SOFT_END = ".;:"

_SENT_SPLIT_RE = re.compile(
    r"""(?<=[!?…])            # после ! ? …
        (?=\s)                # и перед пробелом
      | (?<=\.)               # после точки
        (?=\s["'«»(]*[A-ZА-ЯЁ\u0400-\u04FF])  # если дальше заглавная (в т.ч. кириллица)
      | (?<=[。！？；])         # CJK-пунктуация
      | (?<=\n)               # перевод строки
    """,
    re.VERBOSE | re.UNICODE,
)


@dataclass
class Chunk:
    index: int
    text: str
    total: int = 0

    @property
    def chars(self) -> int:
        return len(self.text)


def clean_text(text: str, max_chars: int = 100_000) -> str:
    """Нормализация входного текста: NFKC, убираем управляющие символы, схлопываем пробелы."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(ch for ch in t if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()[:max_chars]


def _is_abbrev(before: str) -> bool:
    tail = before.rstrip().split(" ")[-2:] if before.strip() else []
    cand = " ".join(tail).lower().rstrip(".")
    for n in (1, 2):
        words = cand.split(" ")[-n:]
        if ".".join(words) in _ABBREV:
            return True
    return False


def split_sentences(text: str) -> List[str]:
    """Разбить на предложения с учётом сокращений, десятичных дробей и CJK."""
    text = clean_text(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    out: List[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf = (buf + " " + p).strip() if buf else p
        last = buf[-1]
        ends = last in _SENT_END or (last in _SOFT_END and not _is_abbrev(buf))
        if ends or len(buf) > 120:
            out.append(buf)
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return [o for o in out if o]


def split_paragraphs(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def _pack(units: List[str], target: int, minimum: int, maximum: int) -> List[str]:
    """Жадная упаковка единиц в чанки длиной ~target символов."""
    chunks: List[str] = []
    cur = ""
    for u in units:
        if not u:
            continue
        if not cur:
            cur = u
            continue
        if len(cur) + len(u) + 1 <= target:
            cur = f"{cur} {u}"
            continue
        # текущий чанк уже «приличный» — закрываем
        if len(cur) >= minimum:
            chunks.append(cur)
            cur = u
        elif len(cur) + len(u) + 1 <= maximum:
            cur = f"{cur} {u}"
        else:
            chunks.append(cur)
            cur = u
    if cur:
        # короткий хвост приклеиваем к предыдущему чанку, если он от этого не раздуется
        if chunks and len(cur) < minimum and len(chunks[-1]) + len(cur) + 1 <= maximum * 1.3:
            chunks[-1] = f"{chunks[-1]} {cur}"
        else:
            chunks.append(cur)
    # слишком длинные чанки режем по предложениям
    final: List[str] = []
    for c in chunks:
        if len(c) <= maximum:
            final.append(c)
            continue
        sub = split_sentences(c)
        if len(sub) <= 1:
            # жёсткая нарезка по пробелам
            words = c.split(" ")
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 > maximum and buf:
                    final.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf:
                final.append(buf)
        else:
            final.extend(_pack(sub, target, minimum, maximum))
    return [f for f in final if f.strip()]


def chunk_text(
    text: str,
    mode: str = "sentence",
    target_chars: int = 220,
    min_chars: int = 40,
    max_chars: int = 700,
) -> List[str]:
    """Разбить текст на фрагменты для последовательной генерации.

    mode:
      off         — один чанк на весь текст (минимальная нагрузка на GPU, максимум задержки);
      paragraph   — по абзацам (хорошо для длинных текстов, меньше швов, чем по предложениям);
      sentence    — по предложениям с упаковкой до target_chars (лучшая задержка первого звука).
    """
    text = clean_text(text)
    if not text:
        return []
    mode = (mode or "sentence").lower()
    if mode in ("off", "none", "whole"):
        return [text]
    if len(text) <= target_chars:
        return [text]
    if mode == "paragraph":
        units = split_paragraphs(text)
        if len(units) <= 1:
            units = split_sentences(text)
        return _pack(units, max(target_chars, 320), min_chars, max_chars)
    units = split_sentences(text)
    if len(units) <= 1:
        return _pack([text], target_chars, min_chars, max_chars)
    return _pack(units, target_chars, min_chars, max_chars)


def chunks(text: str, mode: str = "sentence", **kw) -> List[Chunk]:
    texts = chunk_text(text, mode=mode, **kw)
    total = len(texts)
    return [Chunk(index=i, text=t, total=total) for i, t in enumerate(texts)]


def estimate_frames(text: str, tokens_per_char: float = 4.0, min_tokens: int = 64) -> int:
    """Грубая оценка бюджета токенов: 12.5 codec-кадров/с ≈ 11–14 символов русской речи в секунду."""
    n = len(clean_text(text))
    return max(min_tokens, int(n * max(0.1, tokens_per_char)))


def looks_cyrillic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return cyr / len(letters) > 0.5


def guess_language(text: str) -> Optional[str]:
    """Подсказка языка по символам (используется только если в запросе language=Auto)."""
    t = clean_text(text)
    if not t:
        return None
    if looks_cyrillic(t):
        return "Russian"
    # Порядок важен: японский текст почти всегда содержит иероглифы (漢字),
    # но наличие каны однозначно указывает на японский, а не на китайский.
    if re.search(r"[\u3040-\u30ff]", t):          # хирагана/катакана
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", t):          # хангыль
        return "Korean"
    if re.search(r"[\u4e00-\u9fff]", t):          # иероглифы без каны
        return "Chinese"
    letters = [c.lower() for c in t if c.isalpha()]
    if not letters:
        return None
    ascii_ratio = sum(1 for c in letters if "a" <= c <= "z") / len(letters)
    return "English" if ascii_ratio > 0.85 else None
