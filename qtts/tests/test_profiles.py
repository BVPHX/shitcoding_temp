"""Хранилище профилей голосов: slug, CRUD, дубликаты, эталонное аудио."""

from __future__ import annotations

import json

import pytest

from qtts.app.profiles import ProfileError, ProfileStore, slugify
from tests.conftest import b64_wav, wav_bytes


@pytest.fixture
def store(tmp_path):
    return ProfileStore(tmp_path / "profiles", min_sec=1.0, max_sec=10.0)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Мама", "mama"),
        ("Диктор Иван", "diktor-ivan"),
        ("Тёплый голос", "teplyy-golos"),     # «ы» не должно превращаться в «i»
        ("Йодль", "yodl"),                    # «й» → y
        ("Ёлка", "elka"),                     # «ё» → e
        ("Voice 1", "voice-1"),
        ("  Hello, World!  ", "hello-world"),
        ("Ünïcödé", "unicode"),               # латинская диакритика снимается
    ],
)
def test_slugify_ascii(name, expected):
    out = slugify(name)
    assert out == expected
    assert out.isascii() and out and "/" not in out and ".." not in out


def test_slugify_fallback_for_non_latin_script():
    """Имя целиком из иероглифов → стабильный fallback с хэшем (разные имена не collide)."""
    a, b = slugify("声音"), slugify("声音и")
    assert a.startswith("voice-") and len(a) == len("voice-") + 6
    assert a == slugify("声音")
    assert a != b


def test_slugify_never_empty():
    assert slugify("") and slugify("???") and slugify("   ")


def test_create_clone_from_bytes(store):
    p, info = store.create("Мама", kind="clone", audio_src=wav_bytes(3.0), ref_text="Привет, малыш.")
    assert p.id == "mama" and p.kind == "clone"
    assert p.ref_text == "Привет, малыш."
    path = store.audio_path(p)
    assert path and path.exists() and path.suffix == ".wav"
    assert "stats" in info or "warnings" in info


def test_create_clone_from_base64(store):
    p, _ = store.create("From Base64", kind="clone", audio_src=b64_wav(2.0))
    assert p.id == "from-base64"
    assert store.audio_path(p).exists()


def test_reference_saved_as_24k_mono_wav(store):
    import soundfile as sf

    p, _ = store.create("LowRate", kind="clone", audio_src=wav_bytes(2.0))
    path = store.audio_path(p)
    assert path is not None and path.suffix == ".wav"
    data, sr = sf.read(str(path))
    assert sr == 24000                       # нативная частота токенайзера Qwen3-TTS
    assert data.ndim == 1                    # моно


def test_create_preset_without_audio(store):
    p, _ = store.create("Тёплый голос", kind="preset", speaker="vivian", instruct="Спокойно")
    assert p.id == "teplyy-golos" and p.speaker == "vivian"
    assert store.audio_path(p) is None


def test_preset_requires_speaker(store):
    with pytest.raises(ProfileError):
        store.create("No speaker", kind="preset")


def test_clone_requires_audio(store):
    with pytest.raises(ProfileError):
        store.create("No audio", kind="clone")


def test_unknown_kind_rejected(store):
    with pytest.raises(ProfileError):
        store.create("X", kind="magic")


def test_duplicate_id_rejected(store):
    store.create("Один", kind="preset", speaker="ryan")
    with pytest.raises(ProfileError):
        store.create("Один", kind="preset", speaker="ryan", profile_id="odin")


def test_update_changes_fields(store):
    p, _ = store.create("Исходный", kind="preset", speaker="ryan")
    updated = store.update(p.id, name="Новое имя", instruct="Строгим тоном", language="Russian",
                           tags=["новость"], notes="заметка")
    assert updated.name == "Новое имя"
    assert updated.instruct == "Строгим тоном"
    assert updated.language == "Russian"
    assert updated.tags == ["новость"]


def test_update_missing_profile(store):
    with pytest.raises(ProfileError):
        store.update("nope", name="x")


def test_duplicate_copies_audio(store):
    p, _ = store.create("Оригинал", kind="clone", audio_src=wav_bytes(2.0), ref_text="Текст.")
    copy = store.duplicate(p.id, "Копия")
    assert copy.id != p.id
    assert store.audio_path(copy).exists()
    assert copy.ref_text == "Текст."


def test_delete_removes_directory(store):
    p, _ = store.create("Удаляемый", kind="clone", audio_src=wav_bytes(2.0))
    d = store.profile_dir(p.id)
    assert d.exists()
    assert store.delete(p.id) is True
    assert not d.exists()
    assert store.delete(p.id) is False


def test_persistence_across_instances(tmp_path):
    root = tmp_path / "profiles"
    s1 = ProfileStore(root, min_sec=1.0)
    p, _ = s1.create("Постоянный", kind="clone", audio_src=wav_bytes(2.0), ref_text="Проверка.")
    s2 = ProfileStore(root, min_sec=1.0)
    got = s2.get(p.id)
    assert got.name == "Постоянный" and got.ref_text == "Проверка."
    assert s2.audio_path(got).exists()
    assert json.loads((root / "profiles.json").read_text(encoding="utf-8"))


def test_describe_payload(store):
    p, _ = store.create("Описание", kind="clone", audio_src=wav_bytes(2.0))
    d = store.describe(p)
    assert d["id"] == "opisanie"
    assert d["url"] == f"/api/profiles/{p.id}/audio"
    assert d["audio"]["duration_sec"] > 0
    assert "extra" not in d


def test_find_by_name_is_case_insensitive(store):
    store.create("Голос Мамы", kind="preset", speaker="sohee")
    assert store.find_by_name("голос мамы") is not None
    assert store.find_by_name("нет такого") is None


def test_load_reference(store):
    p, _ = store.create("С эталоном", kind="clone", audio_src=wav_bytes(2.0), ref_text="Транскрипт.")
    path, text = store.load_reference(p.id)
    assert path and path.exists() and text == "Транскрипт."
