"""Бэкенд-прокси на vLLM-Omni — самый быстрый вариант вывода Qwen3-TTS.

vLLM-Omni умеет Qwen3-TTS «из коробки» (day-0 support) и предоставляет OpenAI-совместимый
`POST /v1/audio/speech` со стримингом (`stream=true, stream_format="audio", response_format="pcm"`),
двухстадийным пайплайном (talker + code2wav) и async-чанкингом для низкой задержки первого звука.
На ROCm vLLM-Omni использует model runner V1 (см. deploy-config `qwen3_tts.yaml`).

Запуск отдельного сервера (пример):

    vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
        --omni --port 8091 --gpu-memory-utilization 0.6

После этого в нашем конфиге: device.backend = "vllm_omni", device.vllm_base_url = "http://127.0.0.1:8091".
Наш сервер продолжает отдавать тот же API и GUI, а тяжёлую работу делает vLLM.
"""

from __future__ import annotations

import base64
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ..config import ModelCfg
from ..logging_setup import GEN, get_logger
from .base import ChunkEvent, EngineBase, EngineError, EngineStatus, ProgressFn, SynthesisRequest

log = get_logger("engine.vllm")

TASK_MAP = {"preset": "CustomVoice", "design": "VoiceDesign", "clone": "Base"}


class VLLMOmniEngine(EngineBase):
    name = "vllm_omni"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._lock = threading.RLock()
        self._client = None
        self._info: Dict[str, Any] = {}
        self._ready = False
        self._loaded: List[str] = []
        self._metas: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #

    @property
    def base_url(self) -> str:
        return (self.cfg.device.vllm_base_url or "http://127.0.0.1:8091").rstrip("/")

    def _http(self):
        if self._client is None:
            try:
                import httpx  # noqa: PLC0415
            except Exception as exc:  # noqa: BLE001
                raise EngineError("нужен пакет httpx: pip install httpx", code="no_httpx") from exc
            self._client = httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0))
        return self._client

    def probe(self) -> Dict[str, Any]:
        """Проверяем доступность vLLM-Omni и узнаём список моделей."""
        info: Dict[str, Any] = {"base_url": self.base_url, "ok": False}
        try:
            r = self._http().get(f"{self.base_url}/v1/models", timeout=10.0)
            info["status_code"] = r.status_code
            if r.status_code == 200:
                data = r.json()
                info["models"] = [m.get("id") for m in data.get("data", [])]
                info["ok"] = bool(info["models"])
        except Exception as exc:  # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {exc}"
        self._info = info
        self._ready = bool(info.get("ok"))
        return info

    def load_model(self, mcfg: Optional[ModelCfg] = None, force: bool = False) -> Dict[str, Any]:
        mcfg = mcfg or self.cfg.model()
        info = self.probe()
        if not info.get("ok"):
            raise EngineError(
                "vLLM-Omni сервер недоступен: "
                f"{info.get('error') or info.get('status_code')} (url={self.base_url}). "
                "Запустите: vllm serve " + mcfg.source + " --deploy-config vllm_omni/deploy/qwen3_tts.yaml --omni --port 8091",
                code="vllm_unreachable",
                details=info,
            )
        with self._lock:
            if mcfg.key not in self._loaded:
                self._loaded.append(mcfg.key)
        meta = {
            "key": mcfg.key,
            "label": mcfg.label or mcfg.key,
            "source": mcfg.source,
            "kind": mcfg.kind if mcfg.kind != "auto" else ("clone" if "base" in mcfg.source.lower() else "preset"),
            "device": f"vllm-omni @ {self.base_url}",
            "dtype": "remote",
            "attn": "remote",
            "load_ms": 0.0,
            "loaded_at": time.time(),
            "sample_rate": 24000,
            "speakers": [],
            "languages": [],
            "remote_models": info.get("models"),
        }
        self._meta = meta
        with self._lock:
            self._metas[mcfg.key] = meta
        try:
            r = self._http().get(f"{self.base_url}/v1/audio/voices", timeout=10.0)
            if r.status_code == 200:
                payload = r.json()
                voices = payload.get("voices") or payload.get("data") or []
                meta["speakers"] = [v.get("voice_id") or v.get("id") or v if isinstance(v, dict) else v for v in voices]
        except Exception as exc:  # noqa: BLE001
            log.debug("не удалось получить список голосов vLLM: %s", exc)
        log.info("vLLM-Omni подключён: %s, модели=%s", self.base_url, info.get("models"))
        return meta

    def unload_model(self, model_key: Optional[str] = None) -> None:
        with self._lock:
            self._loaded = []
            self._metas = {}
        self._ready = False

    def status(self) -> EngineStatus:
        st = EngineStatus(name=self.name)
        st.ready = bool(self._ready or self._loaded)
        st.loaded_models = list(self._loaded)
        st.active_model = self._loaded[-1] if self._loaded else None
        st.device = f"vllm-omni @ {self.base_url}"
        st.uptime_sec = round(time.time() - self.started_at, 1)
        st.counters = dict(self.counters)
        st.info = {
            "probe": self._info,
            "models": [self._metas[k] for k in self._loaded if k in self._metas],
        }
        return st

    def speakers(self, model_key: Optional[str] = None) -> List[str]:
        return list(getattr(self, "_meta", {}).get("speakers") or [])

    # ------------------------------------------------------------------ #

    def synthesize_chunks(self, req: SynthesisRequest, progress: ProgressFn = None) -> Iterator[ChunkEvent]:
        mcfg = self.cfg.model(req.model_key)
        payload: Dict[str, Any] = {
            "model": mcfg.source,
            "input": req.text,
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "task_type": TASK_MAP.get(req.voice.kind, "CustomVoice"),
        }
        if req.language:
            payload["language"] = req.language
        if req.voice.kind == "preset" and req.voice.speaker:
            payload["voice"] = req.voice.speaker
        if req.voice.instruct:
            payload["instructions"] = req.voice.instruct
        gen = req.gen_overrides or {}
        if gen.get("max_new_tokens"):
            payload["max_new_tokens"] = int(gen["max_new_tokens"])
        if gen.get("seed") is not None:
            payload["seed"] = int(gen["seed"])
        if req.voice.kind == "clone":
            ref_path, ref_text = req.voice.ref_path, req.voice.ref_text
            if not ref_path and req.voice.profile_id and getattr(self, "_profiles", None):
                ref_path, t = self._profiles.load_reference(req.voice.profile_id)
                ref_text = ref_text or t
            if not ref_path:
                raise EngineError("для клонирования нужен эталонный аудиофайл профиля", code="no_reference")
            payload["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(ref_path.read_bytes()).decode()
            if ref_text:
                payload["ref_text"] = ref_text
            if req.voice.x_vector_only:
                payload["x_vector_only_mode"] = True

        sr = 24000
        if progress:
            progress("prepared", {"backend": self.name, "task_type": payload["task_type"]})
        log.log(GEN, "Этап: запрос к vLLM-Omni (%s, %d симв.)", payload["task_type"], len(req.text))

        t0 = time.perf_counter()
        buf = bytearray()
        first = True
        idx = 0
        try:
            with self._http().stream("POST", f"{self.base_url}/v1/audio/speech", json=payload) as r:
                if r.status_code != 200:
                    body = r.read().decode(errors="ignore")[:500]
                    raise EngineError(f"vLLM-Omni вернул {r.status_code}: {body}", code="vllm_error")
                for piece in r.iter_bytes(chunk_size=48_000):
                    if not piece:
                        continue
                    if first:
                        log.log(GEN, "Этап: первый аудио-пакет от vLLM за %.0f мс", (time.perf_counter() - t0) * 1000)
                        first = False
                    buf.extend(piece)
                    # режем на ~2-секундные блоки, чтобы отдавать их сразу
                    while len(buf) >= sr * 2 * 2:
                        cut = (len(buf) // (sr * 2)) * (sr * 2)
                        block = bytes(buf[:cut])
                        del buf[:cut]
                        yield self._make_event(block, sr, idx, progress)
                        idx += 1
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"ошибка стрима vLLM-Omni: {exc}", code="vllm_stream_error") from exc
        if buf:
            yield self._make_event(bytes(buf), sr, idx, progress)
            idx += 1
        total_ms = (time.perf_counter() - t0) * 1000
        log.log(GEN, "Этап: vLLM-Omni завершил за %.0f мс", total_ms)

    def _make_event(self, pcm: bytes, sr: int, index: int, progress: ProgressFn) -> ChunkEvent:
        wav = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
        dur = len(wav) / float(sr)
        self.counters["chunks"] = self.counters.get("chunks", 0) + 1
        self.counters["audio_sec"] = round(self.counters.get("audio_sec", 0.0) + dur, 3)
        if progress:
            progress("chunk_done", {"index": index, "audio_sec": round(dur, 3)})
        return ChunkEvent(index=index, total=0, text="", audio=wav, sample_rate=sr, audio_sec=round(dur, 3))

    def attach_profiles(self, store) -> None:
        self._profiles = store

    def shutdown(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception:  # noqa: BLE001
            pass
