"""Общие интерфейсы движков синтеза.

Движок (engine) — это то, чем реально считается аудио:
  * qwen_tts   — локальный вывод через пакет `qwen-tts` (PyTorch/HIP + ROCm). Основной режим.
  * vllm_omni  — прокси на отдельный сервер vLLM-Omni (самый быстрый вариант, если он у вас собрался).
  * dummy      — генератор тестовых тонов: позволяет гонять весь стек (API/GUI/стриминг) без GPU и весов.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from ..config import Config, ModelCfg


class EngineError(RuntimeError):
    """Любая ошибка движка синтеза (с человекочитаемым сообщением для API/GUI)."""

    def __init__(self, message: str, *, code: str = "engine_error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass
class VoiceSpec:
    """Куда именно «целимся» голосом."""

    kind: str = "preset"                       # preset | clone | design
    speaker: Optional[str] = None              # для preset (vivian, ryan, ...)
    profile_id: Optional[str] = None           # для clone
    instruct: Optional[str] = None             # стиль/эмоция (preset и design)
    ref_path: Optional[Path] = None            # эталонное аудио (clone, если задано напрямую)
    ref_text: Optional[str] = None             # транскрипт эталона (clone)
    x_vector_only: bool = False
    language_hint: Optional[str] = None

    def describe(self) -> str:
        if self.kind == "preset":
            return f"preset:{self.speaker}"
        if self.kind == "clone":
            return f"clone:{self.profile_id or (self.ref_path.name if self.ref_path else 'inline')}"
        return f"design:{(self.instruct or '')[:40]}"


@dataclass
class SynthesisRequest:
    text: str
    model_key: Optional[str] = None
    language: Optional[str] = None             # None/Auto = определить автоматически
    voice: VoiceSpec = field(default_factory=VoiceSpec)
    gen_overrides: Dict[str, Any] = field(default_factory=dict)
    format: str = "wav"
    sample_rate: Optional[int] = None
    speed: Optional[float] = None
    streaming: Optional[bool] = None
    job_id: Optional[str] = None
    seed: Optional[int] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "chars": len(self.text),
            "model": self.model_key,
            "voice": self.voice.describe(),
            "language": self.language,
            "format": self.format,
            "streaming": self.streaming,
        }


@dataclass
class ChunkEvent:
    index: int
    total: int
    text: str
    audio: np.ndarray                          # float32, моно, нативная частота модели
    sample_rate: int
    stage: str = "generated"                   # generated | decoded
    gen_ms: float = 0.0
    decode_ms: float = 0.0
    tokens: int = 0
    tok_per_sec: float = 0.0
    audio_sec: float = 0.0
    t_start: float = field(default_factory=time.time)


@dataclass
class EngineStatus:
    name: str = "unknown"
    ready: bool = False
    loaded_models: List[str] = field(default_factory=list)
    active_model: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn: Optional[str] = None
    gfx: Optional[str] = None
    vram_allocated: int = 0
    vram_reserved: int = 0
    vram_total: int = 0
    uptime_sec: float = 0.0
    counters: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


ProgressFn = Optional[Callable[[str, Dict[str, Any]], None]]


class EngineBase(abc.ABC):
    """Интерфейс движка. Все методы вызываются из рабочего потока (не из event loop)."""

    name = "base"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.started_at = time.time()
        self.counters: Dict[str, Any] = {"requests": 0, "chunks": 0, "audio_sec": 0.0, "errors": 0}

    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def load_model(self, mcfg: ModelCfg, force: bool = False) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def unload_model(self, model_key: Optional[str] = None) -> None: ...

    @abc.abstractmethod
    def status(self) -> EngineStatus: ...

    @abc.abstractmethod
    def synthesize_chunks(self, req: SynthesisRequest, progress: ProgressFn = None) -> Iterator[ChunkEvent]: ...

    # ------------------------------------------------------------------ #

    def models(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": m.key,
                "label": m.label or m.key,
                "source": m.source,
                "kind": m.kind,
                "local": Path(m.source).exists(),
            }
            for m in self.cfg.models
        ]

    def speakers(self, model_key: Optional[str] = None) -> List[str]:
        return []

    def languages(self, model_key: Optional[str] = None) -> List[str]:
        return [
            "Auto", "Chinese", "English", "Japanese", "Korean", "German",
            "French", "Russian", "Portuguese", "Spanish", "Italian",
        ]

    def encode(self, wav: np.ndarray, sr: int, fmt: str) -> bytes:
        from .. import audio as audio_mod

        return audio_mod.encode(wav, sr, fmt, normalize=self.cfg.audio.normalize, peak=self.cfg.audio.peak)

    def sample_rate(self) -> int:
        return 24000

    def invalidate_cache(self, model_key: Optional[str] = None) -> None:  # noqa: B027
        """Сброс кэша prompt'ов клонирования (вызывается при выгрузке модели).

        Не абстрактный: движкам без кэша (dummy, vllm_omni) переопределять его не нужно.
        """

    def shutdown(self) -> None:
        try:
            self.unload_model()
        except Exception:  # noqa: BLE001
            pass
