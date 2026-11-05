"""Сервисный слой: очередь, движок, профили, кодирование, метрики.

Здесь живёт вся логика «запрос → аудио». API-роуты и GUI-эндпоинты — тонкие обёртки над ним.
Все блокирующие вызовы движка выполняются в рабочем потоке (asyncio.to_thread / run_in_executor).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from . import audio as audio_mod
from . import diagnostics
from .config import Config, save_config
from .engine import EngineBase, EngineError, create_engine
from .engine.base import SynthesisRequest, VoiceSpec
from .jobs import Job, JobStore
from .logging_setup import GEN, get_logger
from .profiles import ProfileStore

log = get_logger("service")

PRESET_SPEAKERS = [
    "vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee",
]
LANG_ALIASES = {
    "": "Auto", "auto": "Auto", "detect": "Auto", "ru": "Russian", "ru-ru": "Russian",
    "en": "English", "en-us": "English", "en-gb": "English", "zh": "Chinese", "zh-cn": "Chinese",
    "ja": "Japanese", "jp": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
    "pt": "Portuguese", "es": "Spanish", "it": "Italian",
}


class TTSService:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # куда сохраняем конфиг при изменениях из GUI/API (тот же файл, что загрузили)
        self.config_path = getattr(cfg, "config_path", None)
        self.profiles = ProfileStore(
            cfg.profiles_path, min_sec=cfg.profiles.ref_min_sec, max_sec=cfg.profiles.ref_max_sec
        )
        self.jobs = JobStore()
        self._engine: Optional[EngineBase] = None
        self._engine_name: Optional[str] = None
        self._sem = threading.Semaphore(max(1, cfg.generation.max_concurrent))
        self._engine_lock = threading.RLock()
        self.started_at = time.time()
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Движок
    # ------------------------------------------------------------------ #

    @property
    def engine(self) -> EngineBase:
        with self._engine_lock:
            name = (self.cfg.device.backend or "qwen_tts").lower()
            if self._engine is None or self._engine_name != name:
                if self._engine is not None:
                    log.info("останавливаю предыдущий движок '%s'", self._engine_name)
                    self._engine.shutdown()
                self._engine = create_engine(self.cfg, name)
                if hasattr(self._engine, "attach_profiles"):
                    self._engine.attach_profiles(self.profiles)
                self._engine_name = name
                log.info("движок синтеза: %s", name)
            return self._engine

    def load_model(self, model_key: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        mcfg = self.cfg.model(model_key)
        log.info("загружаю модель '%s' ← %s", mcfg.key, mcfg.source)
        meta = self.engine.load_model(mcfg, force=force)
        return meta

    def unload_model(self, model_key: Optional[str] = None) -> None:
        self.engine.unload_model(model_key)

    def status(self) -> Dict[str, Any]:
        st = self.engine.status().to_dict()
        st.update(
            backend=self._engine_name or self.cfg.device.backend,
            profiles=len(self.profiles.list()),
            jobs=self.jobs.stats(),
            uptime_sec=round(time.time() - self.started_at, 1),
            config_models=[m.__dict__ for m in self.cfg.models],
        )
        return st

    def gpu_stats(self) -> Dict[str, Any]:
        return diagnostics.gpu_live_stats()

    def diagnostics(self, force: bool = False, bench: bool = True) -> Dict[str, Any]:
        return diagnostics.collect(force=force, bench=bench)

    # ------------------------------------------------------------------ #
    # Голоса
    # ------------------------------------------------------------------ #

    def normalize_language(self, lang: Optional[str]) -> str:
        if lang is None:
            return ""
        key = str(lang).strip().lower()
        return LANG_ALIASES.get(key, str(lang).strip().capitalize() if key else "Auto")

    def resolve_voice(
        self,
        voice: Any = None,
        *,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Any = None,
        ref_text: Optional[str] = None,
        x_vector_only: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> VoiceSpec:
        """Собирает VoiceSpec из разнородных входов API/GUI.

        voice может быть: строкой (id/имя профиля ИЛИ имя пресета), словарём или None.
        """
        vs = VoiceSpec(kind="preset")
        raw: Any = voice

        if isinstance(raw, dict):
            if raw.get("profile"):
                raw = raw["profile"]
            elif raw.get("profile_id"):
                raw = raw["profile_id"]
            elif raw.get("speaker"):
                speaker = speaker or raw.get("speaker")
                instruct = instruct or raw.get("instruct") or raw.get("instructions")
                raw = None
            elif raw.get("ref_audio"):
                ref_audio = ref_audio or raw.get("ref_audio")
                ref_text = ref_text or raw.get("ref_text")
                x_vector_only = raw.get("x_vector_only_mode", x_vector_only)
                instruct = instruct or raw.get("instruct")
                raw = None
        if isinstance(raw, list) and raw:
            raw = raw[0]

        if isinstance(raw, str) and raw.strip():
            name = raw.strip()
            prof = self.profiles.find_by_name(name)
            if prof is not None:
                vs = self._voice_from_profile(prof, instruct=instruct, x_vector_only=x_vector_only)
            elif name.lower() in PRESET_SPEAKERS or name.lower() in [s.lower() for s in self._engine_speakers()]:
                vs = VoiceSpec(kind="preset", speaker=name, instruct=instruct)
            elif self.profiles.exists(name):
                vs = self._voice_from_profile(self.profiles.get(name), instruct=instruct,
                                              x_vector_only=x_vector_only)
            else:
                # неизвестное имя — считаем пресетом, движок проверит его сам
                vs = VoiceSpec(kind="preset", speaker=name, instruct=instruct)
        else:
            vs = VoiceSpec(kind="preset", speaker=speaker, instruct=instruct)

        if ref_audio:
            path: Optional[Path] = None
            if isinstance(ref_audio, (bytes, bytearray)):
                tmp = self.cfg.data_path / "tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                path = tmp / f"ref_{int(time.time() * 1000)}.wav"
                wav, sr = audio_mod.load_audio_bytes(bytes(ref_audio))
                wav, sr = audio_mod.prepare_reference(wav, sr)
                audio_mod.write_wav_file(path, wav, sr)
            elif isinstance(ref_audio, (str, Path)):
                s = str(ref_audio)
                if _path_exists(s):
                    path = Path(s)
                elif audio_mod.maybe_base64(s) is not None or s.lower().startswith(("http://", "https://", "data:")):
                    wav, sr, _ = audio_mod.resolve_audio_source(s)
                    tmp = self.cfg.data_path / "tmp"
                    tmp.mkdir(parents=True, exist_ok=True)
                    path = tmp / f"ref_{int(time.time() * 1000)}.wav"
                    wav, sr = audio_mod.prepare_reference(wav, sr)
                    audio_mod.write_wav_file(path, wav, sr)
                else:
                    raise EngineError(f"ref_audio не найден: {s}", code="ref_not_found")
            vs = VoiceSpec(
                kind="clone",
                profile_id=vs.profile_id,
                ref_path=path,
                ref_text=ref_text,
                instruct=instruct or vs.instruct,
                x_vector_only=bool(x_vector_only) if x_vector_only is not None else vs.x_vector_only,
            )

        task_l = (task or "").lower()
        if task_l in ("design", "voicedesign", "voice_design"):
            vs.kind = "design"
        elif task_l in ("base", "clone", "voice_clone"):
            vs.kind = "clone"
        elif task_l in ("customvoice", "custom_voice", "preset"):
            vs.kind = "preset"
        if vs.kind == "design" and not vs.instruct:
            vs.instruct = instruct or "Естественный нейтральный голос."
        return vs

    def _voice_from_profile(self, prof, *, instruct: Optional[str] = None,
                            x_vector_only: Optional[bool] = None) -> VoiceSpec:
        if prof.kind == "preset":
            return VoiceSpec(kind="preset", speaker=prof.speaker, instruct=instruct or prof.instruct,
                             language_hint=prof.language)
        ref_path = self.profiles.audio_path(prof)
        return VoiceSpec(
            kind="clone" if prof.kind == "clone" else "clone",
            profile_id=prof.id,
            ref_path=ref_path,
            ref_text=prof.ref_text,
            instruct=instruct or prof.instruct,
            x_vector_only=bool(x_vector_only if x_vector_only is not None else prof.x_vector_only),
            language_hint=prof.language,
        )

    def _engine_speakers(self) -> List[str]:
        try:
            return self.engine.speakers()
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ #
    # Синтез
    # ------------------------------------------------------------------ #

    def _queue_guard(self, job: Job) -> None:
        queued = sum(1 for j in self.jobs.list(limit=1000, state="queued"))
        if queued >= self.cfg.generation.queue_limit:
            raise EngineError(
                f"очередь переполнена ({queued}/{self.cfg.generation.queue_limit}) — повторите позже",
                code="queue_full",
            )
        if not self._sem.acquire(timeout=self.cfg.generation.request_timeout_sec):
            raise EngineError("таймаут ожидания свободного слота генерации", code="queue_timeout")
        self.jobs.update(job, state="running", started_at=time.time())
        job.touch("running")

    def _finish(self, job: Job, state: str, **kw: Any) -> None:
        job.finished_at = time.time()
        self.jobs.update(job, state=state, **kw)
        job.touch(state)

    def synthesize_chunks_sync(self, job: Job, req: SynthesisRequest):
        """Генератор (выполняется в рабочем потоке): (meta, encoded_bytes, last).

        Потоковый режим отдаёт ЕДИНЫЙ байтовый поток:
          * fmt=wav — сначала один RIFF-заголовок (размеры 0xFFFFFFFF, «бесконечный»
            стрим), затем сырые PCM16-чанки; склейка ответа = один валидный WAV;
          * fmt=pcm — только сырые PCM16-чанки.
        Чанки приводятся к выходной частоте/speed/паузам, поэтому поток совпадает
        с итоговым файлом (который уходит в last-элементе и сохраняется на диске;
        HTTP-роуты его байты в поток НЕ дублируют).
        """
        fmt = (req.format or self.cfg.audio.format or "wav").lower()
        streaming = req.streaming if req.streaming is not None else self.cfg.generation.streaming
        if streaming and fmt not in ("pcm", "raw", "wav"):
            log.warning("стриминг недоступен для формата '%s' — отдаю единым файлом", fmt)
            streaming = False
        stream_fmt = fmt if fmt in ("pcm", "raw") else "wav" if streaming else fmt

        sr_out = self._out_sr(req)
        speed = req.speed if req.speed is not None else self.cfg.generation.speed
        speed = float(speed) if speed else 1.0
        use_speed = abs(speed - 1.0) > 0.01
        gap_ms = self._gap_ms()

        self._queue_guard(job)
        t_start = time.perf_counter()
        first_at: Optional[float] = None
        total_audio = 0.0
        parts: List[np.ndarray] = []
        idx = 0
        try:
            if streaming and stream_fmt == "wav":
                header = audio_mod.wav_stream_header(sr_out)
                job.bytes_out += len(header)
                yield (
                    {
                        "type": "header",
                        "job_id": job.id,
                        "sample_rate": sr_out,
                        "format": "wav",
                        "codec": "pcm_s16le",
                    },
                    header,
                    False,
                )
            progress = self._progress_cb(job)
            # отмена на уровне движка: ключи с «_» исключаются из kwargs модели (_gen_kwargs).
            # Перезаписываем всегда: событие должно указывать на ТЕКУЩУЮ задачу.
            req.gen_overrides = {**(req.gen_overrides or {}), "_cancel_event": job.cancel_event}
            for ev in self.engine.synthesize_chunks(req, progress=progress):
                if job.cancel_event.is_set():
                    raise EngineError("отменено клиентом", code="cancelled")
                idx += 1
                if first_at is None:
                    first_at = time.perf_counter()
                    job.ttfa_ms = round((first_at - t_start) * 1000, 1)
                    log.log(GEN, "Этап: первый аудио-чанк через %.0f мс (job %s)", job.ttfa_ms, job.id)
                total_audio += ev.audio_sec or (len(ev.audio) / float(ev.sample_rate or 24000))
                parts.append(ev.audio)
                job.chunks_done = idx
                job.chunks_total = ev.total or job.chunks_total
                job.audio_sec = round(total_audio, 3)
                job.progress = (idx / max(1, job.chunks_total or idx))
                if streaming:
                    pcm = ev.audio
                    ev_sr = ev.sample_rate or sr_out
                    if ev_sr != sr_out:
                        pcm = audio_mod.resample(pcm, ev_sr, sr_out)
                    if use_speed:
                        pcm = self._apply_speed(pcm, speed)
                    if idx > 1 and gap_ms > 0:
                        pcm = np.concatenate([audio_mod.add_silence(gap_ms, sr_out), pcm])
                    data = audio_mod.float_to_pcm16(pcm)
                    job.bytes_out += len(data)
                    meta = {
                        "type": "audio",
                        "job_id": job.id,
                        "index": idx - 1,
                        "total": ev.total or 0,
                        "sample_rate": sr_out,
                        "format": stream_fmt,
                        "codec": "pcm_s16le",
                        "audio_sec": round(len(pcm) / float(sr_out), 3),
                        "gen_ms": ev.gen_ms,
                        "decode_ms": ev.decode_ms,
                        "tok_per_sec": ev.tok_per_sec,
                        "text": ev.text,
                        "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    }
                    self.jobs.publish(job)
                    yield meta, data, False
            # ---- финал ----
            wav = audio_mod.concat(parts, gap_ms=gap_ms, sr=sr_out)
            if use_speed:
                wav = self._apply_speed(wav, speed)
                total_audio = len(wav) / float(sr_out)
            data = self.engine.encode(wav, sr_out, fmt)
            elapsed = (time.perf_counter() - t_start) * 1000
            out_path = self._maybe_save(data, fmt, job)
            self._finish(
                job,
                "done",
                audio_sec=round(total_audio, 3),
                bytes_out=len(data) if not streaming else job.bytes_out,
                sample_rate=sr_out,
                format=fmt,
                gen_ms=round(elapsed, 1),
                rtf=round(elapsed / 1000.0 / total_audio, 3) if total_audio > 0 else None,
                output_file=str(out_path) if out_path else None,
                stage="done",
                progress=1.0,
            )
            meta = {
                "type": "final",
                "job_id": job.id,
                "sample_rate": sr_out,
                "format": fmt,
                "audio_sec": round(total_audio, 3),
                "bytes": len(data),
                "elapsed_ms": round(elapsed, 1),
                "ttfa_ms": job.ttfa_ms,
                "rtf": job.rtf,
                "chunks": idx,
                "output_file": job.output_file,
            }
            log.log(
                GEN,
                "Готово (job %s): %.2f с аудио, %d чанк(ов), %.0f мс, RTF %s, TTFA %s мс, %s",
                job.id, total_audio, idx, elapsed,
                f"{job.rtf}" if job.rtf else "—", f"{job.ttfa_ms}" if job.ttfa_ms else "—",
                human_size(len(data)),
            )
            yield meta, data, True
        except EngineError as exc:
            self._finish(job, "cancelled" if exc.code == "cancelled" else "error",
                         error=str(exc), error_code=exc.code, stage="error")
            log.error("ошибка синтеза (job %s): [%s] %s", job.id, exc.code, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            self._finish(job, "error", error=f"{type(exc).__name__}: {exc}", error_code="internal", stage="error")
            log.exception("непредвиденная ошибка синтеза (job %s)", job.id)
            raise EngineError(f"{type(exc).__name__}: {exc}", code="internal") from exc
        finally:
            try:
                self._sem.release()
            except ValueError:
                pass

    async def stream_synthesis(self, job: Job, req: SynthesisRequest) -> AsyncIterator[Dict[str, Any]]:
        """Асинхронная обёртка: мост между рабочим потоком и event loop."""
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Tuple[Dict[str, Any], bytes, bool]]]" = asyncio.Queue(maxsize=64)
        gen = self.synthesize_chunks_sync(job, req)
        done = False

        def pump() -> None:
            try:
                for item in gen:
                    asyncio.run_coroutine_threadsafe(q.put(item), loop).result()
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result()
            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(q.put(("__error__", exc, True)), loop).result()

        fut = loop.run_in_executor(None, pump)
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, tuple) and len(item) == 3 and item[0] == "__error__":
                    raise item[1]
                meta, data, last = item
                yield {"meta": meta, "data": data, "last": last}
                if last:
                    done = True
                    break
        finally:
            if not done:
                job.cancel_event.set()
            try:
                await fut
            except Exception:  # noqa: BLE001
                pass

    async def synthesize_full(
        self,
        req: SynthesisRequest,
        client: str = "",
        endpoint: str = "",
        job: Optional[Job] = None,
    ) -> Dict[str, Any]:
        """Синтез одним файлом. `job` — уже созданная задача (иначе создаём новую).

        Важно передавать job из роута: иначе на запрос заводятся две задачи —
        одна «висит» в очереди, а метрики/файл уезжают в другую (X-Job-Id врёт).
        """
        job = job or self.jobs.create(req, client=client, endpoint=endpoint)
        chunks: List[bytes] = []
        final: Dict[str, Any] = {}
        try:
            async for item in self.stream_synthesis(job, req):
                if item["meta"]["type"] == "final":
                    final = item["meta"]
                    chunks = [item["data"]]
                # промежуточные чанки в нестриминговом режиме игнорируем
        except EngineError:
            raise
        return {
            "job": job,
            "audio": chunks[-1] if chunks else b"",
            "meta": final,
            "mime": audio_mod.MIME.get(final.get("format", req.format), "application/octet-stream"),
        }

    # ------------------------------------------------------------------ #
    # Вспомогательное
    # ------------------------------------------------------------------ #

    def _progress_cb(self, job: Job):
        def cb(stage: str, data: Dict[str, Any]) -> None:
            if stage == "prepared":
                job.chunks_total = int(data.get("chunks") or 0)
                job.model_key = data.get("model") or job.model_key
            elif stage == "chunk_start":
                job.stage = f"chunk {int(data.get('index', 0)) + 1}/{data.get('total', '?')}"
            self.jobs.publish(job)

        return cb

    def _gap_ms(self) -> int:
        return max(0, int(self.cfg.generation.inter_chunk_silence_ms))

    def _out_sr(self, req: SynthesisRequest) -> int:
        return int(req.sample_rate or self.cfg.audio.sample_rate or self.engine.sample_rate() or 24000)

    def output_sample_rate(self, req: Optional[SynthesisRequest] = None) -> int:
        """Частота дискретизации ответа (публичная обёртка для роутов)."""
        if req is not None:
            return self._out_sr(req)
        return int(self.cfg.audio.sample_rate or self.engine.sample_rate() or 24000)

    def _apply_speed(self, wav: np.ndarray, speed: float) -> np.ndarray:
        speed = max(0.5, min(2.0, float(speed)))
        try:
            import librosa  # noqa: PLC0415

            return np.asarray(librosa.effects.time_stretch(wav, rate=speed), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            log.warning("изменение скорости недоступно (%s) — возвращаю как есть", exc)
            return wav

    def _maybe_save(self, data: bytes, fmt: str, job: Job) -> Optional[Path]:
        if not self.cfg.audio.save_outputs:
            return None
        p = audio_mod.save_output(data, self.cfg.outputs_path, fmt, prefix=f"job_{job.id}")
        audio_mod.cleanup_outputs(self.cfg.outputs_path, self.cfg.audio.keep_outputs)
        return p

    # ------------------------------------------------------------------ #
    # Настройки из GUI
    # ------------------------------------------------------------------ #

    def update_config(self, patch: Dict[str, Any]) -> Tuple[Config, List[str]]:
        """Применяет изменения конфига «на лету». Возвращает (новый конфиг, список нужных перезапусков)."""
        from .config import _from_dict  # noqa: PLC0415

        merged = self.cfg.to_dict()
        restart: List[str] = []

        def merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], v)
                else:
                    dst[k] = v

        merge(merged, patch or {})
        new_cfg = _from_dict(merged)
        old = self.cfg
        # что требует пересоздания движка / перезагрузки модели
        if new_cfg.device.backend != old.device.backend:
            restart.append("backend")
        if new_cfg.generation.max_concurrent != old.generation.max_concurrent:
            self._sem = threading.Semaphore(max(1, new_cfg.generation.max_concurrent))
        old_models = {m.key: m.source for m in old.models}
        new_models = {m.key: m.source for m in new_cfg.models}
        if old_models != new_models:
            restart.append("model")
        if (new_cfg.device.dtype, new_cfg.device.attn, new_cfg.device.device) != (
            old.device.dtype, old.device.attn, old.device.device
        ):
            restart.append("model")

        self.cfg = new_cfg
        new_cfg.config_path = self.config_path
        save_config(new_cfg, self.config_path)
        if self._engine is not None:
            self._engine.cfg = new_cfg
        log.info("конфиг обновлён (требуется перезапуск: %s)", restart or "нет")
        return new_cfg, restart

    def reload_engine(self) -> Dict[str, Any]:
        with self._engine_lock:
            if self._engine is not None:
                self._engine.shutdown()
            self._engine = None
            self._engine_name = None
        meta = self.load_model(force=True)
        return meta

    # ------------------------------------------------------------------ #
    # Быстрый тест из GUI («Проверить режим работы»)
    # ------------------------------------------------------------------ #

    def self_test(self, text: str = "Проверка связи. Синтез работает.", voice: Any = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        req = SynthesisRequest(
            text=text,
            voice=self.resolve_voice(voice or (PRESET_SPEAKERS[0] if self._model_kind() == "preset" else None)),
            language=self.cfg.generation.language,
            format="wav",
            streaming=False,
        )
        if req.voice.kind == "clone" and req.voice.ref_path is None:
            raise EngineError("для теста клон-голоса нужен профиль с эталоном", code="no_reference")
        coro = self.synthesize_full(req, client="self-test", endpoint="/api/system/test")
        res = asyncio.run(coro) if _no_loop() else _run_sync(coro)
        return {
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "audio_bytes": len(res["audio"]),
            "audio_sec": res["meta"].get("audio_sec"),
            "rtf": res["meta"].get("rtf"),
            "job": res["job"].to_dict(with_events=True),
        }

    def _model_kind(self) -> str:
        try:
            st = self.engine.status()
            for m in st.info.get("models", []):
                if m.get("key") == st.active_model:
                    return m.get("kind", "clone")
        except Exception:  # noqa: BLE001
            pass
        return "clone"


def _path_exists(s: str) -> bool:
    """Безопасная проверка «это существующий путь?».

    Строка может оказаться base64-эталона в несколько десятков КБ — Path().exists()
    на ней бросает OSError(ENAMETOOLONG), что раньше давало 500 вместо 422/рабочего клона.
    """
    if not s or len(s) > 4096 or "\x00" in s:
        return False
    try:
        return Path(s).exists()
    except OSError:
        return False


def _no_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return False
    except RuntimeError:
        return True


def _run_sync(coro):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"
