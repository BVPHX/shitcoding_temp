"""Локальный движок на базе пакета `qwen-tts` (Qwen3-TTS 12Hz 0.6B/1.7B).

Оптимизации под AMD ROCm (Ryzen + RX 9060 XT, gfx1200):
  * bf16 — нативный и самый быстрый формат для RDNA4 (fp32 в ~2 раза медленнее);
  * attn_implementation="sdpa" + TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 →
    SDPA уходит в AOTriton-ядра (flash/mem-efficient). flash-attn из PyPI под ROCm не собирается;
  * expandable_segments:True — меньше фрагментации VRAM на длинных сессиях;
  * torch.compile по умолчанию ВЫКЛЮЧЕН: на ROCm inductor/Triton уходит в бесконечные
    рекомпиляции из-за динамических форм TTS (это главная причина «тормозов» в типовых серверах);
  * prompt-кэш клонирования: x-vector и codec-коды эталона считаются один раз на профиль;
  * чанкование текста + немедленная отдача чанка → низкая задержка первого звука (TTFA);
  * бюджет max_new_tokens по длине текста (12.5 кадра/с) — не даём модели «зависать» до 4096 токенов.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .. import audio as audio_mod
from .. import pipeline
from ..config import ModelCfg
from ..logging_setup import GEN, GPU, get_logger
from .base import ChunkEvent, EngineBase, EngineError, EngineStatus, ProgressFn, SynthesisRequest, VoiceSpec

log = get_logger("engine.qwen")

_DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16",
           "half": "float16", "float32": "float32", "fp32": "float32", "float": "float32"}


class QwenEngine(EngineBase):
    name = "qwen_tts"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._lock = threading.RLock()
        self._models: "OrderedDict[str, Any]" = OrderedDict()
        self._model_meta: Dict[str, Dict[str, Any]] = {}
        self._prompt_cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
        self._torch = None
        self._qwen = None
        self._sdpa_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Импорт тяжёлых зависимостей
    # ------------------------------------------------------------------ #

    def _ensure_imports(self):
        if self._torch is not None:
            return self._torch, self._qwen
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                "PyTorch не установлен или не импортируется. Для ROCm:\n"
                "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm10.0",
                code="no_torch",
            ) from exc
        try:
            from qwen_tts import Qwen3TTSModel  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Пакет qwen-tts не установлен ({exc}). Выполните: pip install qwen-tts==0.1.1",
                code="no_qwen_tts",
            ) from exc
        self._torch = torch
        self._qwen = Qwen3TTSModel
        log.info(
            "torch %s | hip=%s | cuda_visible=%s | gpu_count=%d",
            torch.__version__,
            getattr(torch.version, "hip", None),
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        return torch, Qwen3TTSModel

    # ------------------------------------------------------------------ #
    # Устройство / dtype / attention
    # ------------------------------------------------------------------ #

    def resolve_device(self) -> str:
        torch, _ = self._ensure_imports()
        want = (self.cfg.device.device or "auto").lower()
        if want in ("auto", ""):
            want = "cuda:0" if torch.cuda.is_available() else "cpu"
        if want.startswith("cuda") and not torch.cuda.is_available():
            raise EngineError(
                " запрошено GPU-устройство, но torch.cuda.is_available() == False. "
                "Откройте GUI → Система → Диагностика ROCm, чтобы узнать причину.",
                code="gpu_unavailable",
            )
        return want

    def resolve_dtype(self, device: str):
        torch, _ = self._ensure_imports()
        name = _DTYPES.get((self.cfg.device.dtype or "bfloat16").lower(), "bfloat16")
        dt = getattr(torch, name)
        if device.startswith("cuda") and name == "bfloat16":
            try:
                if not torch.cuda.is_bf16_supported():
                    log.warning("GPU не поддерживает bf16 — переключаюсь на float16")
                    return torch.float16, "float16"
            except Exception:  # noqa: BLE001
                pass
        if name in ("bfloat16", "float16") and device == "cpu":
            log.warning("На CPU %s медленнее float32 — оставляю, как задано в конфиге", name)
        return dt, name

    def resolve_attn(self, device: str) -> str:
        """Выбор реализации attention: auto → flash_attention_2 (если реально есть) → sdpa → eager."""
        want = (self.cfg.device.attn or "auto").lower()
        if want != "auto":
            return want
        try:
            import importlib.util  # noqa: PLC0415

            if importlib.util.find_spec("flash_attn") is not None:
                log.info("найден flash_attn — пробую attn_implementation=flash_attention_2")
                return "flash_attention_2"
        except Exception:  # noqa: BLE001
            pass
        if device.startswith("cuda"):
            import os  # noqa: PLC0415

            if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") != "1":
                log.warning(
                    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL != 1 → SDPA может уйти в медленный math-backend. "
                    "Переменная задаётся до запуска (см. config.json: device.env)."
                )
        return "sdpa"

    def _probe_sdpa(self, device: str) -> Dict[str, Any]:
        torch, _ = self._ensure_imports()
        if not device.startswith("cuda"):
            return {"note": "CPU"}
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

            q = torch.randn(1, 8, 64, 64, device=device, dtype=torch.bfloat16)
            out: Dict[str, Any] = {}
            for nm, be in (("flash", SDPBackend.FLASH_ATTENTION),
                           ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                           ("math", SDPBackend.MATH)):
                try:
                    with sdpa_kernel(be):
                        torch.nn.functional.scaled_dot_product_attention(q, q, q)
                    out[nm] = True
                except Exception as exc:  # noqa: BLE001
                    out[nm] = f"{type(exc).__name__}"
            del q
            return out
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Загрузка модели
    # ------------------------------------------------------------------ #

    def load_model(self, mcfg: Optional[ModelCfg] = None, force: bool = False) -> Dict[str, Any]:
        mcfg = mcfg or self.cfg.model()
        with self._lock:
            if mcfg.key in self._models and not force:
                self._models.move_to_end(mcfg.key)
                return dict(self._model_meta[mcfg.key], cached=True)

            torch, Qwen3TTSModel = self._ensure_imports()
            if self.cfg.device.allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision(self.cfg.device.matmul_precision or "high")
            except Exception:  # noqa: BLE001
                pass
            if self.cfg.device.deterministic:
                torch.use_deterministic_algorithms(False)
                torch.manual_seed(self.cfg.generation.seed or 0)

            device = self.resolve_device()
            dtype, dtype_name = self.resolve_dtype(device)
            attn = self.resolve_attn(device)

            src = mcfg.source
            is_local = Path(src).exists()
            log.log(
                GPU,
                "Загрузка модели %s (%s) → device=%s dtype=%s attn=%s",
                mcfg.key, src if is_local else f"HF:{src}", device, dtype_name, attn,
            )
            t0 = time.perf_counter()
            kwargs: Dict[str, Any] = {
                "device_map": device,
                "dtype": dtype,
                "attn_implementation": attn,
                "local_files_only": bool(mcfg.local_files_only or (is_local and not _looks_like_repo(src))),
            }
            if mcfg.cache_dir:
                kwargs["cache_dir"] = str(Path(mcfg.cache_dir).expanduser())
            if mcfg.revision and not is_local:
                kwargs["revision"] = mcfg.revision
            if mcfg.trust_remote_code:
                kwargs["trust_remote_code"] = True

            model = None
            errors: List[str] = []
            for attempt_attn in _attn_fallback_chain(attn):
                try:
                    model = Qwen3TTSModel.from_pretrained(src, **{**kwargs, "attn_implementation": attempt_attn})
                    attn = attempt_attn
                    break
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:400]
                    errors.append(f"attn={attempt_attn}: {msg}")
                    log.warning("загрузка с attn=%s не удалась: %s", attempt_attn, msg[:200])
            if model is None:
                raise EngineError(
                    "Не удалось загрузить модель. Последняя ошибка: " + (errors[-1] if errors else "unknown"),
                    code="model_load_failed",
                    details={"attempts": errors},
                )

            inner = getattr(model, "model", None)
            kind = _detect_kind(mcfg, inner)
            self._models[mcfg.key] = model
            self._models.move_to_end(mcfg.key)
            meta = {
                "key": mcfg.key,
                "label": mcfg.label or mcfg.key,
                "source": src,
                "kind": kind,
                "device": device,
                "dtype": dtype_name,
                "attn": attn,
                "load_ms": round((time.perf_counter() - t0) * 1000, 1),
                "loaded_at": time.time(),
                "sample_rate": self._model_sample_rate(model),
                "speakers": _safe_list(model, "get_supported_speakers"),
                "languages": _safe_list(model, "get_supported_languages"),
                "params": _count_params(inner),
                "gfx": _device_gfx(torch, device),
            }
            self._model_meta[mcfg.key] = meta
            self._sdpa_info = self._probe_sdpa(device)
            meta["sdpa"] = self._sdpa_info
            log.log(
                GPU,
                "Модель %s загружена за %.1f с: kind=%s device=%s dtype=%s attn=%s params=%s VRAM=%s",
                mcfg.key, meta["load_ms"] / 1000, kind, device, dtype_name, attn,
                meta["params"], _vram_str(torch, device),
            )
            if self.cfg.generation.warmup:
                self.warmup(model, mcfg.key, kind)
            return meta

    def _model_sample_rate(self, model) -> int:
        try:
            return int(model.model.speech_tokenizer.get_output_sample_rate())
        except Exception:  # noqa: BLE001
            return audio_mod.NATIVE_SR

    def warmup(self, model, key: str, kind: str) -> None:
        """Прогрев: первая генерация компилирует Triton/HIP-ядра и занимает кэш — дальше быстрее."""
        text = self.cfg.generation.warmup_text or "Проверка."
        t0 = time.perf_counter()
        try:
            if kind == "preset":
                spk = (self._model_meta[key].get("speakers") or ["vivian"])[0]
                model.generate_custom_voice(text=text, speaker=spk, language="Auto", max_new_tokens=96)
            elif kind == "design":
                model.generate_voice_design(text=text, instruct="Нейтральный голос.", language="Auto", max_new_tokens=96)
            else:
                # для base-модели без эталона греем только токенайзер-декодер
                sr = self._model_sample_rate(model)
                codes = np.zeros((4, 1), dtype=np.int64)
                model.model.speech_tokenizer.decode([{"audio_codes": self._torch.as_tensor(codes)}])
                log.debug("warmup: декодер токенайзера прогрет (sr=%d)", sr)
            log.info("warmup модели %s выполнен за %.2f с", key, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            log.warning("warmup не удался (%s) — продолжаю без прогрева", exc)

    def unload_model(self, model_key: Optional[str] = None) -> None:
        with self._lock:
            keys = [model_key] if model_key else list(self._models.keys())
            for k in keys:
                if k in self._models:
                    del self._models[k]
                    self._model_meta.pop(k, None)
                    log.log(GPU, "Модель %s выгружена", k)
            self.invalidate_cache(model_key)
            if self._torch is not None and self._torch.cuda.is_available():
                try:
                    import gc  # noqa: PLC0415

                    gc.collect()
                    self._torch.cuda.empty_cache()
                    self._torch.cuda.synchronize()
                except Exception:  # noqa: BLE001
                    pass

    def invalidate_cache(self, model_key: Optional[str] = None) -> None:
        with self._lock:
            if model_key is None:
                self._prompt_cache.clear()
                log.debug("prompt-кэш очищен полностью")
            else:
                for k in [k for k in self._prompt_cache if k[0] == model_key]:
                    del self._prompt_cache[k]

    # ------------------------------------------------------------------ #
    # Статус
    # ------------------------------------------------------------------ #

    def status(self) -> EngineStatus:
        torch = self._torch
        st = EngineStatus(name=self.name)
        st.loaded_models = list(self._models.keys())
        st.active_model = next(reversed(self._models), None) if self._models else None
        st.uptime_sec = round(time.time() - self.started_at, 1)
        st.counters = dict(self.counters)
        st.info = {
            "models": [self._model_meta[k] for k in self._models if k in self._model_meta],
            "prompt_cache": [f"{k[0]}/{k[1]}" for k in self._prompt_cache],
            "sdpa": self._sdpa_info,
            "configured_models": self.models(),
        }
        if self._models and torch is not None:
            meta = self._model_meta.get(st.active_model or "", {})
            st.device = meta.get("device")
            st.dtype = meta.get("dtype")
            st.attn = meta.get("attn")
            st.gfx = meta.get("gfx")
            st.ready = True
            try:
                if str(st.device).startswith("cuda"):
                    st.vram_allocated = int(torch.cuda.memory_allocated(0))
                    st.vram_reserved = int(torch.cuda.memory_reserved(0))
                    free, total = torch.cuda.mem_get_info(0)
                    st.vram_total = int(total)
            except Exception:  # noqa: BLE001
                pass
        return st

    # ------------------------------------------------------------------ #
    # Выбор голоса и prompt-кэш
    # ------------------------------------------------------------------ #

    def speakers(self, model_key: Optional[str] = None) -> List[str]:
        m = self._get_model(model_key)
        return list(self._model_meta[m[0]].get("speakers") or [])

    def languages(self, model_key: Optional[str] = None) -> List[str]:
        m = self._get_model(model_key)
        langs = list(self._model_meta[m[0]].get("languages") or [])
        out = ["Auto"] + [x for x in langs if str(x).lower() != "auto"]
        return out

    def _get_model(self, model_key: Optional[str] = None) -> Tuple[str, Any]:
        with self._lock:
            if model_key and model_key in self._models:
                self._models.move_to_end(model_key)
                return model_key, self._models[model_key]
            if self._models:
                key = next(reversed(self._models))
                return key, self._models[key]
        mcfg = self.cfg.model(model_key)
        self.load_model(mcfg)
        with self._lock:
            return mcfg.key, self._models[mcfg.key]

    def _model_kind(self, model_key: str) -> str:
        return self._model_meta.get(model_key, {}).get("kind", "clone")

    def _resolve_clone_prompt(self, model_key: str, model, voice: VoiceSpec):
        """Готовим voice_clone_prompt (с кэшем по профилю) для base-модели."""
        cache_on = bool(self.cfg.generation.clone.get("prompt_cache", True))
        cache_key = (model_key, voice.profile_id or f"inline:{hash((str(voice.ref_path), voice.ref_text))}")
        if cache_on:
            hit = self._prompt_cache.get(cache_key)
            if hit is not None:
                self._prompt_cache.move_to_end(cache_key)
                log.debug("prompt-кэш: HIT %s", cache_key)
                return hit, True

        ref_audio: Any
        if voice.ref_path is not None:
            ref_audio = str(voice.ref_path)
        elif voice.profile_id:
            from ..profiles import ProfileStore  # noqa: PLC0415

            store: Optional[ProfileStore] = getattr(self, "_profiles", None)
            if store is None:
                raise EngineError("хранилище профилей не подключено к движку", code="no_profiles")
            path, ref_text = store.load_reference(voice.profile_id)
            if path is None:
                raise EngineError(
                    f"у профиля '{voice.profile_id}' нет эталонного аудио", code="profile_no_audio"
                )
            ref_audio = str(path)
            voice.ref_text = voice.ref_text or ref_text
        else:
            raise EngineError("для клонирования нужен profile_id или ref_audio", code="no_reference")

        x_only = bool(voice.x_vector_only) or not (voice.ref_text or "").strip()
        t0 = time.perf_counter()
        items = model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=None if x_only else voice.ref_text, x_vector_only_mode=x_only
        )
        ms = (time.perf_counter() - t0) * 1000
        log.log(GEN, "Этап: prompt клонирования (%s, x_vector_only=%s) — %.0f мс",
                voice.describe(), x_only, ms)
        if cache_on and voice.profile_id:
            self._prompt_cache[cache_key] = items
            while len(self._prompt_cache) > max(1, self.cfg.profiles.prompt_cache_size):
                self._prompt_cache.popitem(last=False)
        return items, False

    def attach_profiles(self, store) -> None:
        self._profiles = store

    # ------------------------------------------------------------------ #
    # Генерация
    # ------------------------------------------------------------------ #

    def synthesize_chunks(self, req: SynthesisRequest, progress: ProgressFn = None) -> Iterator[ChunkEvent]:
        import torch  # noqa: PLC0415

        def emit(stage: str, **kw) -> None:
            if progress:
                progress(stage, kw)

        model_key, model = self._get_model(req.model_key)
        kind = self._model_kind(model_key)
        lang = self._resolve_language(req)
        gen = self._gen_kwargs(req)
        sr_model = self._model_sample_rate(model)
        target_sr = req.sample_rate or self.cfg.audio.sample_rate or sr_model

        use_chunking = bool(req.streaming if req.streaming is not None else self.cfg.generation.streaming)
        mode = self.cfg.generation.chunking if use_chunking else "off"
        chunks = pipeline.chunks(
            req.text,
            mode=mode,
            target_chars=self.cfg.generation.chunk_target_chars,
            min_chars=self.cfg.generation.chunk_min_chars,
            max_chars=self.cfg.generation.chunk_max_chars,
        )
        if not chunks:
            raise EngineError("пустой текст после нормализации", code="empty_text")

        log.log(
            GEN,
            "Этап: подготовка — модель=%s kind=%s язык=%s голос=%s чанков=%d/%d режим=%s",
            model_key, kind, lang, req.voice.describe(), len(chunks), len(chunks), mode,
        )
        emit("prepared", model=model_key, kind=kind, language=lang, chunks=len(chunks), voice=req.voice.describe())

        cancel = req.gen_overrides.get("_cancel_event")

        with self._lock, torch.inference_mode():
            if kind == "clone":
                prompt_items, cached = self._resolve_clone_prompt(model_key, model, req.voice)
                emit("prompt_ready", cached=cached)
            else:
                prompt_items = None

            for ch in chunks:
                if cancel is not None and getattr(cancel, "is_set", lambda: False)():
                    raise EngineError("генерация отменена пользователем", code="cancelled")
                budget = self._token_budget(ch.text)
                emit("chunk_start", index=ch.index, total=ch.total, chars=ch.chars, budget=budget)
                log.log(GEN, "Этап: чанк %d/%d — генерация (%d симв., бюджет %d токенов)…",
                        ch.index + 1, ch.total, ch.chars, budget)
                t0 = time.perf_counter()
                wav, sr, tokens = self._run_chunk(model, kind, ch.text, lang, req.voice, prompt_items, gen, budget)
                gen_ms = (time.perf_counter() - t0) * 1000

                if sr != sr_model:
                    sr_model = sr
                if target_sr and target_sr != sr:
                    t1 = time.perf_counter()
                    wav = audio_mod.resample(wav, sr, target_sr)
                    decode_ms = (time.perf_counter() - t1) * 1000
                    sr = target_sr
                else:
                    decode_ms = 0.0

                audio_sec = len(wav) / float(sr) if sr else 0.0
                tps = tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
                ev = ChunkEvent(
                    index=ch.index,
                    total=ch.total,
                    text=ch.text,
                    audio=wav,
                    sample_rate=int(sr),
                    stage="decoded",
                    gen_ms=round(gen_ms, 1),
                    decode_ms=round(decode_ms, 1),
                    tokens=tokens,
                    tok_per_sec=round(tps, 1),
                    audio_sec=round(audio_sec, 3),
                )
                self.counters["chunks"] = self.counters.get("chunks", 0) + 1
                self.counters["audio_sec"] = round(self.counters.get("audio_sec", 0.0) + audio_sec, 3)
                log.log(
                    GEN,
                    "Этап: чанк %d/%d готов — %.2f с аудио за %.0f мс (RTF %.2f, %.1f ток/с), всего аудио %.2f с",
                    ch.index + 1, ch.total, audio_sec, gen_ms,
                    (gen_ms / 1000.0) / audio_sec if audio_sec else 0.0, tps,
                    self.counters["audio_sec"],
                )
                emit(
                    "chunk_done",
                    index=ch.index, total=ch.total, audio_sec=round(audio_sec, 3),
                    gen_ms=round(gen_ms, 1), rtf=round((gen_ms / 1000.0) / audio_sec, 3) if audio_sec else None,
                    tok_per_sec=round(tps, 1),
                )
                yield ev

    # ------------------------------------------------------------------ #

    def _run_chunk(self, model, kind: str, text: str, lang: str, voice: VoiceSpec,
                   prompt_items, gen: Dict[str, Any], budget: int) -> Tuple[np.ndarray, int, int]:
        gen = dict(gen)
        non_streaming = bool(gen.pop("non_streaming_mode", True))
        common = dict(language=lang, max_new_tokens=budget, **gen)
        if kind == "clone":
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt_items,
                non_streaming_mode=non_streaming,
                **common,
            )
        elif kind == "preset":
            spk = (voice.speaker or "").strip()
            if not spk:
                raise EngineError("для preset-голоса нужен speaker (см. /api/voices/presets)", code="no_speaker")
            wavs, sr = model.generate_custom_voice(
                text=text, speaker=spk, instruct=voice.instruct or None,
                non_streaming_mode=non_streaming, **common,
            )
        elif kind == "design":
            if not (voice.instruct or "").strip():
                raise EngineError("для VoiceDesign нужен instruct (описание голоса)", code="no_instruct")
            wavs, sr = model.generate_voice_design(
                text=text, instruct=voice.instruct, non_streaming_mode=non_streaming, **common
            )
        else:
            raise EngineError(f"неизвестный тип модели: {kind}", code="bad_model_kind")

        if not wavs:
            raise EngineError("модель не вернула аудио", code="empty_audio")
        wav = audio_mod.to_mono(np.asarray(wavs[0], dtype=np.float32))
        if wav.size == 0:
            raise EngineError("модель вернула пустой массив аудио", code="empty_audio")
        tokens = int(wav.size / max(1, int(sr)) * 12.5)
        return wav, int(sr), tokens

    def _resolve_language(self, req: SynthesisRequest) -> str:
        lang = (req.language or req.voice.language_hint or self.cfg.generation.language or "Auto")
        if str(lang).lower() in ("", "auto", "none", "detect"):
            guessed = pipeline.guess_language(req.text)
            if guessed:
                log.debug("язык определён по тексту: %s", guessed)
                return guessed
            return "Auto"
        return str(lang)

    def _gen_kwargs(self, req: SynthesisRequest) -> Dict[str, Any]:
        g = self.cfg.generation
        ov = {k: v for k, v in (req.gen_overrides or {}).items() if not k.startswith("_") and v is not None}
        kw: Dict[str, Any] = dict(
            do_sample=g.do_sample,
            top_k=g.top_k,
            top_p=g.top_p,
            temperature=g.temperature,
            repetition_penalty=g.repetition_penalty,
            subtalker_dosample=g.subtalker_do_sample,
            subtalker_top_k=g.subtalker_top_k,
            subtalker_top_p=g.subtalker_top_p,
            subtalker_temperature=g.subtalker_temperature,
            non_streaming_mode=bool(g.non_streaming_mode),
        )
        # маппинг «человеческих» имён из API
        alias = {
            "top_k": "top_k", "top_p": "top_p", "temperature": "temperature",
            "repetition_penalty": "repetition_penalty", "do_sample": "do_sample",
            "max_new_tokens": "max_new_tokens", "seed": "seed",
            "subtalker_temperature": "subtalker_temperature", "subtalker_top_k": "subtalker_top_k",
            "subtalker_top_p": "subtalker_top_p", "subtalker_do_sample": "subtalker_dosample",
        }
        for k, v in ov.items():
            if k in alias:
                kw[alias[k]] = v
        seed = req.seed if req.seed is not None else kw.pop("seed", None)
        if seed is not None:
            try:
                self._torch.manual_seed(int(seed))
                kw.setdefault("do_sample", True)
            except Exception:  # noqa: BLE001
                pass
        kw["min_new_tokens"] = max(2, int(g.min_new_tokens))
        return kw

    def _token_budget(self, text: str) -> int:
        g = self.cfg.generation
        est = pipeline.estimate_frames(text, g.tokens_per_char, min_tokens=64)
        return int(max(g.min_new_tokens, min(est, g.max_new_tokens)))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def _looks_like_repo(src: str) -> bool:
    """'Qwen/Qwen3-TTS-12Hz-1.7B-Base' — repo id; '/models/base' — локальный путь."""
    if Path(src).exists():
        return False
    return bool(_REPO_ID_RE.match(src))


def _attn_fallback_chain(attn: str) -> List[str]:
    order = ["flash_attention_2", "sdpa", "eager"]
    if attn in order:
        i = order.index(attn)
        return order[i:]
    return [attn] + order


def _detect_kind(mcfg: ModelCfg, inner) -> str:
    if mcfg.kind and mcfg.kind != "auto":
        return mcfg.kind
    t = ""
    try:
        t = str(getattr(inner, "tts_model_type", "") or "").lower()
    except Exception:  # noqa: BLE001
        t = ""
    if t == "base":
        return "clone"
    if t in ("custom_voice", "customvoice"):
        return "preset"
    if t in ("voice_design", "voicedesign"):
        return "design"
    src = (mcfg.source or "").lower()
    if "base" in src:
        return "clone"
    if "customvoice" in src or "custom_voice" in src:
        return "preset"
    if "voicedesign" in src or "voice_design" in src:
        return "design"
    return "clone"


def _safe_list(model, method: str) -> List[str]:
    try:
        fn = getattr(model, method, None)
        if callable(fn):
            v = fn()
            return [str(x) for x in v] if v else []
    except Exception:  # noqa: BLE001
        pass
    return []


def _count_params(inner) -> Optional[str]:
    try:
        n = sum(p.numel() for p in inner.parameters())
        return f"{n / 1e9:.2f}B" if n else None
    except Exception:  # noqa: BLE001
        return None


def _device_gfx(torch, device: str) -> Optional[str]:
    try:
        if str(device).startswith("cuda"):
            return str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":")[0] or None
    except Exception:  # noqa: BLE001
        pass
    return None


def _vram_str(torch, device: str) -> str:
    try:
        if not str(device).startswith("cuda"):
            return "n/a"
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        res = torch.cuda.memory_reserved(0) / 1024**3
        free, total = torch.cuda.mem_get_info(0)
        return f"alloc {alloc:.2f} GiB / reserved {res:.2f} GiB / free {free / 1024**3:.2f} GiB из {total / 1024**3:.2f} GiB"
    except Exception:  # noqa: BLE001
        return "n/a"
