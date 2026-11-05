"""Тесты сервисного слоя TTSService (на dummy-движке, без torch/GPU)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import soundfile as sf
from conftest import b64_wav, wav_bytes

from qtts.app.config import load_config
from qtts.app.engine import EngineError
from qtts.app.engine.base import SynthesisRequest, VoiceSpec
from qtts.app.profiles import ProfileStore
from qtts.app.service import TTSService


def req(text: str, **kw) -> SynthesisRequest:
    kw.setdefault("format", "wav")
    return SynthesisRequest(text=text, **kw)


def voice(svc: TTSService, name: str = "vivian") -> VoiceSpec:
    return svc.resolve_voice(name)


# ------------------------------------------------------------------ голоса

def test_resolve_preset_voice(service):
    vs = voice(service, "Vivian")
    assert vs.kind == "preset" and vs.speaker == "Vivian"
    assert vs.describe() == "preset:Vivian"


def test_resolve_preset_is_case_insensitive(service):
    assert voice(service, "viviAn").speaker.lower() == "vivian"


def test_resolve_none_gives_default_preset(service):
    vs = service.resolve_voice(None)
    assert vs.kind == "preset" and vs.speaker is None


def test_resolve_empty_string_gives_default_preset(service):
    vs = service.resolve_voice("")
    assert vs.kind == "preset" and vs.speaker is None


def test_resolve_unknown_name_passed_through_as_preset(service):
    """Неизвестное имя не ошибка: его проверит сам движок."""
    vs = voice(service, "какой-то-голос")
    assert vs.kind == "preset" and vs.speaker == "какой-то-голос"


def test_resolve_profile_by_id(service):
    prof, _ = service.profiles.create("Тёплый голос", kind="clone",
                                      audio_src=wav_bytes(3.0), ref_text="Привет.")
    vs = service.resolve_voice(prof.id)
    assert vs.kind == "clone" and vs.profile_id == prof.id
    assert vs.ref_path is not None and vs.ref_path.exists()
    assert vs.ref_text == "Привет."
    assert vs.describe() == f"clone:{prof.id}"


def test_resolve_profile_by_display_name(service):
    prof, _ = service.profiles.create("Диктор Иван", kind="clone", audio_src=wav_bytes(3.0))
    vs = service.resolve_voice("Диктор Иван")
    assert vs.kind == "clone" and vs.profile_id == prof.id


def test_resolve_preset_kind_profile(service):
    prof, _ = service.profiles.create("Мой пресет", kind="preset", speaker="ryan", instruct="Бодро")
    vs = service.resolve_voice(prof.id)
    assert vs.kind == "preset" and vs.speaker == "ryan" and vs.instruct == "Бодро"


def test_resolve_dict_with_speaker(service):
    vs = service.resolve_voice({"speaker": "ryan", "instructions": "шёпотом"})
    assert vs.kind == "preset" and vs.speaker == "ryan" and vs.instruct == "шёпотом"


def test_resolve_dict_with_profile_id(service):
    prof, _ = service.profiles.create("Мама", kind="clone", audio_src=wav_bytes(3.0))
    assert service.resolve_voice({"profile_id": prof.id}).kind == "clone"
    assert service.resolve_voice({"profile": prof.id}).profile_id == prof.id


def test_resolve_inline_ref_audio_bytes(service, cfg):
    vs = service.resolve_voice(None, ref_audio=wav_bytes(2.5))
    assert vs.kind == "clone" and vs.ref_path is not None and vs.ref_path.exists()
    assert vs.ref_path.parent == cfg.data_path / "tmp"
    data, sr = sf.read(io.BytesIO(vs.ref_path.read_bytes()))
    assert sr == 24000 and len(data) / sr == pytest.approx(2.5, abs=0.05)


def test_resolve_inline_ref_audio_base64(service):
    vs = service.resolve_voice(None, ref_audio=b64_wav(2.0), ref_text="Текст эталона.")
    assert vs.kind == "clone" and vs.ref_path.exists() and vs.ref_text == "Текст эталона."


def test_resolve_missing_ref_audio_raises(service):
    with pytest.raises(EngineError) as ei:
        service.resolve_voice(None, ref_audio="/нет/такого/файла.wav")
    assert ei.value.code == "ref_not_found"


def test_resolve_task_overrides_kind(service):
    assert service.resolve_voice("vivian", task="design").kind == "design"
    vs = service.resolve_voice("vivian", task="voice_design")
    assert vs.kind == "design" and vs.instruct            # подставлен instruct по умолчанию
    assert service.resolve_voice("vivian", task="base").kind == "clone"
    assert service.resolve_voice("vivian", task="CustomVoice").kind == "preset"


def test_resolve_instruct_overrides_profile_instruct(service):
    prof, _ = service.profiles.create("С инstructом", kind="clone",
                                      audio_src=wav_bytes(3.0), instruct="Спокойно")
    assert service.resolve_voice(prof.id).instruct == "Спокойно"
    assert service.resolve_voice(prof.id, instruct="Грустно").instruct == "Грустно"


def test_resolve_x_vector_only(service):
    prof, _ = service.profiles.create("Без текста", kind="clone", audio_src=wav_bytes(3.0))
    assert service.resolve_voice(prof.id, x_vector_only=True).x_vector_only is True


@pytest.mark.parametrize("raw,expected", [
    ("ru", "Russian"), ("RU-RU", "Russian"), ("en", "English"), ("zh", "Chinese"),
    ("ja", "Japanese"), ("", "Auto"), ("auto", "Auto"), ("detect", "Auto"), (None, ""),
    ("Klingon", "Klingon"),
])
def test_normalize_language(service, raw, expected):
    assert service.normalize_language(raw) == expected


# ------------------------------------------------------------------ синтез

def test_streaming_wav_yields_audio_then_final(service):
    r = req("Первое предложение. Второе предложение.", voice=voice(service), format="wav")
    job = service.jobs.create(r, client="test", endpoint="/api/tts")
    items = list(service.synthesize_chunks_sync(job, r))

    metas = [m for m, _d, _l in items]
    assert [m["type"] for m in metas] == ["audio"] * (len(metas) - 1) + ["final"]
    assert items[-1][2] is True and all(not it[2] for it in items[:-1])
    for m, data, _ in items[:-1]:
        assert data[:4] == b"RIFF" and m["format"] == "wav"
        assert m["sample_rate"] == 24000 and m["audio_sec"] > 0

    final = metas[-1]
    assert final["chunks"] == len(metas) - 1
    assert final["audio_sec"] > 0 and final["rtf"] > 0 and final["ttfa_ms"] > 0
    assert final["bytes"] == len(items[-1][1])

    assert job.state == "done" and job.progress == 1.0
    assert job.chunks_done == final["chunks"] and job.chunks_total == final["chunks"]
    assert job.audio_sec == pytest.approx(final["audio_sec"])
    assert job.bytes_out > 0 and job.sample_rate == 24000
    assert job.output_file and job.output_file.endswith(".wav")


def test_output_file_is_saved_and_readable(service, cfg):
    r = req("Проверка сохранения.", voice=voice(service), streaming=False)
    job = service.jobs.create(r)
    metas = list(service.synthesize_chunks_sync(job, r))
    assert len(metas) == 1 and metas[0][0]["type"] == "final"

    path = cfg.outputs_path / (job.output_file.split("/")[-1])
    data, sr = sf.read(io.BytesIO(path.read_bytes()))
    assert sr == 24000 and len(data) > 0
    assert job.output_file == str(path)


def test_non_streaming_returns_single_final(service):
    r = req("Первое. Второе. Третье.", voice=voice(service), streaming=False)
    job = service.jobs.create(r)
    items = list(service.synthesize_chunks_sync(job, r))
    assert len(items) == 1 and items[0][0]["type"] == "final" and items[0][2] is True
    assert items[0][1][:4] == b"RIFF"
    assert job.state == "done"


def test_pcm_stream_has_no_container(service):
    r = req("Раз. Два. Три.", voice=voice(service), format="pcm", streaming=True)
    job = service.jobs.create(r)
    items = list(service.synthesize_chunks_sync(job, r))
    payloads = [d for _m, d, last in items if not last]
    assert payloads and all(p[:4] != b"RIFF" for p in payloads)
    assert all(len(p) % 2 == 0 for p in payloads)      # 16-bit PCM
    assert items[-1][0]["format"] == "pcm"


def test_long_text_is_split_into_chunks(service):
    text = " ".join([f"Предложение номер {i} достаточно длинное." for i in range(6)])
    r = req(text, voice=voice(service), format="pcm", streaming=True)
    job = service.jobs.create(r)
    items = list(service.synthesize_chunks_sync(job, r))
    n_audio = sum(1 for m, _d, _l in items if m["type"] == "audio")
    assert n_audio >= 3
    assert [m["index"] for m, _d, _l in items if m["type"] == "audio"] == list(range(n_audio))
    assert all(m["total"] == n_audio for m, _d, _l in items if m["type"] == "audio")


def test_sample_rate_override_resamples(service):
    r = req("Частота.", voice=voice(service), sample_rate=16000, streaming=False)
    job = service.jobs.create(r)
    _meta, data, _last = list(service.synthesize_chunks_sync(job, r))[0]
    assert job.sample_rate == 16000
    _wav, sr = sf.read(io.BytesIO(data))
    assert sr == 16000


def test_engine_error_marks_job_and_raises(service):
    class Boom:
        cfg = service.cfg
        def synthesize_chunks(self, r, progress=None):
            raise RuntimeError("взрыв движка")
        def encode(self, *a, **k):
            return b""
        def sample_rate(self):
            return 24000

    service._engine = Boom()
    service._engine_name = "dummy"
    r = req("Что-нибудь.", voice=voice(service))
    job = service.jobs.create(r)
    with pytest.raises(EngineError) as ei:
        list(service.synthesize_chunks_sync(job, r))
    assert "взрыв движка" in str(ei.value)
    assert ei.value.code == "internal"
    assert job.state == "error" and job.error_code == "internal"
    assert "взрыв" in (job.error or "")


def test_cancelled_job_stops_generation(service):
    r = req("Отмена. Отмена. Отмена. Отмена.", voice=voice(service), format="pcm")
    job = service.jobs.create(r)
    job.cancel_event.set()
    with pytest.raises(EngineError) as ei:
        list(service.synthesize_chunks_sync(job, r))
    assert ei.value.code == "cancelled"
    assert job.state == "cancelled"


def test_queue_limit_rejects_request(cfg):
    cfg.generation.queue_limit = 0
    svc = TTSService(cfg)
    r = req("Не пройдёт.", voice=voice(svc))
    job = svc.jobs.create(r)
    with pytest.raises(EngineError) as ei:
        list(svc.synthesize_chunks_sync(job, r))
    assert ei.value.code == "queue_full"
    assert job.state == "queued"


# ------------------------------------------------- асинхронные обёртки

def test_synthesize_full_collects_single_payload(service):
    asyncio = pytest.importorskip("asyncio")
    r = req("Полный синтез.", voice=voice(service), streaming=False)
    out = asyncio.run(service.synthesize_full(r, client="t", endpoint="/api/tts"))
    assert out["audio"][:4] == b"RIFF"
    assert out["mime"] == "audio/wav"
    assert out["meta"]["type"] == "final"
    assert out["job"].state == "done"


def test_stream_synthesis_bridges_thread_to_loop(service):
    asyncio = pytest.importorskip("asyncio")
    r = req("Раз. Два. Три.", voice=voice(service), format="pcm", streaming=True)
    job = service.jobs.create(r)

    async def collect():
        return [item async for item in service.stream_synthesis(job, r)]

    items = asyncio.run(collect())
    assert len(items) >= 2
    assert items[-1]["last"] is True
    assert items[-1]["meta"]["type"] == "final"
    assert job.state == "done"


def test_self_test_ok_on_dummy(service):
    out = service.self_test()
    assert out["ok"] is True
    assert out["audio_bytes"] > 0 and out["elapsed_ms"] > 0
    assert out["audio_sec"] > 0 and out["rtf"] > 0
    assert out["job"]["state"] == "done"


def test_self_test_custom_text_and_voice(service):
    prof, _ = service.profiles.create("Тестовый", kind="clone", audio_src=wav_bytes(3.0))
    out = service.self_test(text="Своя фраза.", voice=prof.id)
    assert out["ok"] is True and out["job"]["voice"].endswith(prof.id)


def test_self_test_reports_engine_error(service):
    service._engine = type("X", (), {
        "cfg": service.cfg,
        "synthesize_chunks": lambda self, r, progress=None: (_ for _ in ()).throw(RuntimeError("нет связи")),
        "sample_rate": lambda self: 24000,
    })()
    service._engine_name = "dummy"
    with pytest.raises(EngineError):
        service.self_test()


# ------------------------------------------------------- состояние/модель

def test_status_shape(service):
    st = service.status()
    assert st["backend"] == "dummy"
    assert st["profiles"] == 0
    assert st["uptime_sec"] >= 0
    assert st["jobs"]["total"] == 0
    assert st["ready"] is False                    # модель ещё не загружена


def test_load_and_unload_model(service):
    meta = service.load_model()
    assert meta["key"] == service.cfg.models[0].key
    st = service.status()
    assert st["ready"] is True and st["loaded_models"] == [meta["key"]]
    assert st["active_model"] == meta["key"]

    service.unload_model()
    assert service.status()["ready"] is False


def test_output_sample_rate(service):
    assert service.output_sample_rate() == 24000
    assert service.output_sample_rate(req("текст", sample_rate=22050)) == 22050


def test_engine_speakers_and_languages(service):
    service.load_model()
    assert "vivian" in service.engine.speakers()
    assert "Russian" in service.engine.languages()


def test_profiles_store_is_wired(cfg, service):
    assert isinstance(service.profiles, ProfileStore)
    assert service.profiles.root == cfg.profiles_path
    assert service.profiles.list() == []


# ------------------------------------------------- горячее обновление конфига

def test_update_config_applies_patch_and_saves_same_file(service, cfg):
    new_cfg, restart = service.update_config({"generation": {"chunk_target_chars": 999}})
    assert restart == []
    assert new_cfg.generation.chunk_target_chars == 999
    assert service.cfg.generation.chunk_target_chars == 999

    saved = load_config(cfg.config_path)
    assert saved.generation.chunk_target_chars == 999
    # другие поля не потерялись при слиянии
    assert saved.device.backend == "dummy"
    assert saved.server.port == cfg.server.port


def test_update_config_does_not_touch_repo_default(service, cfg):
    """Сохранение идёт в тот файл, откуда конфиг загрузили (--config / QTTSS_CONFIG)."""
    from qtts.app.config import DEFAULT_CONFIG_PATH, active_config_path

    before = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") if DEFAULT_CONFIG_PATH.exists() else None
    service.update_config({"generation": {"warmup_text": "Своя фраза прогрева."}})

    assert Path(cfg.config_path) != DEFAULT_CONFIG_PATH
    assert active_config_path() == Path(cfg.config_path)
    assert load_config(cfg.config_path).generation.warmup_text == "Своя фраза прогрева."
    after = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") if DEFAULT_CONFIG_PATH.exists() else None
    assert after == before


def test_update_config_requests_backend_restart(service):
    _cfg, restart = service.update_config({"device": {"backend": "qwen_tts"}})
    assert "backend" in restart


def test_update_config_requests_model_reload(service):
    _cfg, restart = service.update_config({"device": {"dtype": "float16"}})
    assert "model" in restart
    _cfg, restart = service.update_config({"device": {"attn": "flash_attention_2"}})
    assert "model" in restart


def test_update_config_model_source_change(service):
    patch = {"models": [{"key": service.cfg.models[0].key, "source": "/models/qwen3-tts"}]}
    _cfg, restart = service.update_config(patch)
    assert "model" in restart
    assert service.cfg.model().source == "/models/qwen3-tts"


def test_update_config_rebuilds_semaphore(service):
    old = service._sem
    service.update_config({"generation": {"max_concurrent": 4}})
    assert service._sem is not old
    assert service.cfg.generation.max_concurrent == 4


def test_update_config_rebinds_engine_cfg(service):
    _ = service.engine                                # создаём dummy-движок
    new_cfg, _r = service.update_config({"generation": {"temperature": 0.42}})
    assert service._engine.cfg is new_cfg
    assert service._engine.cfg.generation.temperature == 0.42


def test_update_config_keeps_working_engine(service):
    _ = service.engine
    engine_before = service._engine
    service.update_config({"generation": {"chunk_target_chars": 120}})
    assert service._engine is engine_before           # движок не пересоздавали зря
    r = req("После обновления.", voice=voice(service), streaming=False)
    job = service.jobs.create(r)
    list(service.synthesize_chunks_sync(job, r))
    assert job.state == "done"


def test_new_profile_visible_without_config_reload(service):
    prof, _ = service.profiles.create("Свежий", kind="clone", audio_src=wav_bytes(3.0))
    assert service.resolve_voice(prof.id).profile_id == prof.id
    assert service.status()["profiles"] == 1
