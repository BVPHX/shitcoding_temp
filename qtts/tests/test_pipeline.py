"""Подготовка текста: нормализация, разбиение на предложения/чанки, определение языка."""

from __future__ import annotations

import pytest

from qtts.app.pipeline import (
    chunk_text,
    chunks,
    clean_text,
    estimate_frames,
    guess_language,
    looks_cyrillic,
    split_paragraphs,
    split_sentences,
)


def test_clean_text_normalizes_whitespace_and_control_chars():
    raw = "Привет\u00a0\u00a0мир!\u0007\n\n\n\nВторая   строка"
    out = clean_text(raw)
    assert "\u00a0" not in out and "\u0007" not in out
    assert "Привет мир!" in out
    assert "\n\n\n" not in out


def test_clean_text_respects_limit():
    assert clean_text("а" * 500, max_chars=100) == "а" * 100


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_clean_text_empty(bad):
    assert clean_text(bad) == ""


def test_split_sentences_basic():
    parts = split_sentences("Первое предложение. Второе! Третье? Четвёртое…")
    assert len(parts) >= 3
    assert "".join(parts).replace(" ", "").startswith("Первоепредложение")


def test_split_sentences_keeps_abbreviations_together():
    parts = split_sentences("Дом на ул. Ленина выглядит старым. Крыша протекает.")
    assert not any(p.strip() in ("Дом на ул.", "ул.") for p in parts), parts
    assert len(parts) == 2


def test_split_sentences_decimal_numbers():
    parts = split_sentences("Температура 36.6 градуса. Это нормально.")
    assert len(parts) == 2
    assert "36.6" in parts[0]


def test_split_paragraphs():
    text = "Абзац один.\n\nАбзац два.\nАбзац три."
    parts = split_paragraphs(text)
    assert parts[0] == "Абзац один."
    assert len(parts) >= 2


def test_chunk_text_off_mode_returns_single_chunk():
    text = "Раз. " * 200
    assert chunk_text(text, mode="off") == [text.strip()]


def test_chunk_text_short_text_not_split():
    assert chunk_text("Короткий текст.", mode="sentence", target_chars=220) == ["Короткий текст."]


def test_chunk_text_sentence_mode_splits_long_text():
    text = " ".join(f"Это предложение номер {i} для проверки разбиения." for i in range(40))
    parts = chunk_text(text, mode="sentence", target_chars=120, min_chars=40, max_chars=400)
    assert len(parts) > 1
    assert all(p.strip() for p in parts)
    assert all(len(p) <= 400 for p in parts)
    # ничего не потерялось
    assert sum(len(p) for p in parts) >= len(text) * 0.9


def test_chunk_text_hard_limit_for_one_giant_sentence():
    text = "слово " * 500            # одно «предложение» без точек
    parts = chunk_text(text.strip(), mode="sentence", target_chars=100, min_chars=40, max_chars=200)
    assert all(len(p) <= 240 for p in parts), [len(p) for p in parts]


def test_chunk_text_paragraph_mode():
    text = "\n\n".join("Абзац %d. " % i + "Предложение. " * 20 for i in range(4))
    parts = chunk_text(text, mode="paragraph", target_chars=200, min_chars=40, max_chars=600)
    assert len(parts) >= 2


def test_chunks_index_and_total():
    text = " ".join(f"Предложение {i}." for i in range(60))
    ch = chunks(text, mode="sentence", target_chars=60, min_chars=20, max_chars=200)
    assert [c.index for c in ch] == list(range(len(ch)))
    assert all(c.total == len(ch) for c in ch)
    assert all(c.chars > 0 for c in ch)


def test_estimate_frames_grows_with_text():
    assert estimate_frames("") == 64                     # минимум
    assert estimate_frames("а" * 100) > estimate_frames("а" * 10)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Привет, мир!", "Russian"),
        ("Hello, world!", "English"),
        ("你好，世界", "Chinese"),
        ("こんにちは世界", "Japanese"),
        ("", None),
    ],
)
def test_guess_language(text, expected):
    assert guess_language(text) == expected


def test_looks_cyrillic():
    assert looks_cyrillic("Русский текст") is True
    assert looks_cyrillic("English text") is False
    assert looks_cyrillic("12345") is False
