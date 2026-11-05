"""Фабрика движков синтеза."""

from __future__ import annotations

from typing import Optional

from ..config import Config
from ..logging_setup import get_logger
from .base import ChunkEvent, EngineBase, EngineError, EngineStatus, SynthesisRequest, VoiceSpec

log = get_logger("engine")

BACKENDS = ("qwen_tts", "vllm_omni", "dummy")


def create_engine(cfg: Config, backend: Optional[str] = None) -> EngineBase:
    name = (backend or cfg.device.backend or "qwen_tts").lower()
    if name not in BACKENDS:
        log.warning("неизвестный backend '%s' — использую qwen_tts", name)
        name = "qwen_tts"
    if name == "dummy":
        from .dummy_engine import DummyEngine

        return DummyEngine(cfg)
    if name == "vllm_omni":
        from .vllm_engine import VLLMOmniEngine

        return VLLMOmniEngine(cfg)
    from .qwen_engine import QwenEngine

    return QwenEngine(cfg)


__all__ = [
    "BACKENDS",
    "ChunkEvent",
    "EngineBase",
    "EngineError",
    "EngineStatus",
    "SynthesisRequest",
    "VoiceSpec",
    "create_engine",
]
