"""Тестовый движок: генерирует тоны вместо речи.

Нужен, чтобы проверить весь стек (сервер, API, очередь, стриминг, GUI, профили)
без GPU, без ROCm и без 4 ГБ весов: `--backend dummy` или `device.backend=dummy`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ..config import ModelCfg
from ..logging_setup import get_logger
from .base import ChunkEvent, EngineBase, EngineStatus, ProgressFn, SynthesisRequest

log = get_logger("engine.dummy")

_NOTES = [220.0, 261.63, 293.66, 329.63, 392.0, 440.0]


class DummyEngine(EngineBase):
    name = "dummy"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._loaded: List[str] = []
        self._metas: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._sr = 24000

    def load_model(self, mcfg: Optional[ModelCfg] = None, force: bool = False) -> Dict[str, Any]:
        mcfg = mcfg or self.cfg.model()
        with self._lock:
            if mcfg.key not in self._loaded:
                self._loaded.append(mcfg.key)
        meta = {
            "key": mcfg.key,
            "label": mcfg.label or mcfg.key,
            "source": "dummy",
            "kind": "preset",
            "device": "none",
            "dtype": "float32",
            "attn": "none",
            "load_ms": 1.0,
            "loaded_at": time.time(),
            "sample_rate": self._sr,
            "speakers": ["vivian", "serena", "ryan", "aiden", "dylan", "eric", "ono_anna", "sohee", "uncle_fu"],
            "languages": ["Auto", "Russian", "English", "Chinese"],
            "params": "0.00B",
            "gfx": None,
        }
        self._meta = meta
        with self._lock:
            self._metas[mcfg.key] = meta
        log.warning("DUMMY-движок: вместо речи генерируются тоны (режим отладки API/GUI без GPU)")
        return meta

    def unload_model(self, model_key: Optional[str] = None) -> None:
        with self._lock:
            self._loaded = []
            self._metas = {}

    def status(self) -> EngineStatus:
        st = EngineStatus(name=self.name)
        st.ready = bool(self._loaded)
        st.loaded_models = list(self._loaded)
        st.active_model = self._loaded[-1] if self._loaded else None
        st.device = "none (dummy)"
        st.dtype = "float32"
        st.uptime_sec = round(time.time() - self.started_at, 1)
        st.counters = dict(self.counters)
        st.info = {
            "warning": "dummy-режим: аудио синтетическое",
            # формат как у QwenEngine: /api/models и service._model_kind() читают именно его
            "models": [self._metas[k] for k in self._loaded if k in self._metas],
        }
        return st

    def speakers(self, model_key: Optional[str] = None) -> List[str]:
        return list(getattr(self, "_meta", {}).get("speakers") or [])

    def languages(self, model_key: Optional[str] = None) -> List[str]:
        return ["Auto", "Russian", "English", "Chinese", "Japanese"]

    def synthesize_chunks(self, req: SynthesisRequest, progress: ProgressFn = None) -> Iterator[ChunkEvent]:
        from .. import pipeline

        if not self._loaded:
            self.load_model()
        chunks = pipeline.chunks(
            req.text,
            mode=self.cfg.generation.chunking if req.streaming is not False else "off",
            target_chars=self.cfg.generation.chunk_target_chars,
            min_chars=self.cfg.generation.chunk_min_chars,
            max_chars=self.cfg.generation.chunk_max_chars,
        )
        if progress:
            progress("prepared", {"model": "dummy", "chunks": len(chunks)})
        for ch in chunks:
            dur = max(0.35, len(ch.text) / 14.0)  # ~14 символов в секунду
            if progress:
                progress("chunk_start", {"index": ch.index, "total": ch.total, "chars": ch.chars})
            t0 = time.perf_counter()
            # «генерация» занимает ~0.25 от длительности аудио — имитируем RTF 0.25
            time.sleep(min(2.0, dur * 0.25))
            gen_ms = (time.perf_counter() - t0) * 1000
            freq = _NOTES[ch.index % len(_NOTES)]
            t = np.arange(int(dur * self._sr), dtype=np.float32) / self._sr
            env = np.minimum(1.0, np.minimum(t / 0.02, (dur - t) / 0.05)).clip(0, 1)
            wav = (0.28 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            wav += 0.06 * env * np.sin(2 * np.pi * freq * 2 * t).astype(np.float32)
            self.counters["chunks"] = self.counters.get("chunks", 0) + 1
            self.counters["audio_sec"] = round(self.counters.get("audio_sec", 0.0) + dur, 3)
            if progress:
                progress("chunk_done", {"index": ch.index, "total": ch.total, "audio_sec": round(dur, 2)})
            yield ChunkEvent(
                index=ch.index,
                total=ch.total,
                text=ch.text,
                audio=wav,
                sample_rate=self._sr,
                gen_ms=round(gen_ms, 1),
                tokens=int(dur * 12.5),
                tok_per_sec=round(dur * 12.5 / max(gen_ms, 1) * 1000, 1),
                audio_sec=round(dur, 3),
            )

    def sample_rate(self) -> int:
        return self._sr
