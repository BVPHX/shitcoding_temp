"""Pydantic-схемы запросов/ответов API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

VoiceLike = Union[str, Dict[str, Any], List[str], None]


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, str_strip_whitespace=True)


class TTSRequest(_Base):
    """Основной запрос синтеза.

    Пример (клонирование по профилю):
        {"text": "Привет!", "voice": "mama", "language": "Russian", "stream": true}
    Пример (встроенный тембр + эмоция):
        {"text": "Hello!", "speaker": "ryan", "instruct": "Say it excitedly"}
    """

    text: Optional[str] = Field(None, max_length=100_000, description="Текст для озвучивания")
    input: Optional[str] = Field(None, max_length=100_000, description="Синоним text (OpenAI-стиль)")
    voice: VoiceLike = Field(None, description="id/имя профиля ИЛИ имя встроенного тембра")
    profile: Optional[str] = Field(None, description="id профиля-клона (альтернатива voice)")
    speaker: Optional[str] = Field(None, description="встроенный тембр CustomVoice (vivian, ryan, ...)")
    instruct: Optional[str] = Field(None, description="инструкция стиля/эмоции (для CustomVoice и VoiceDesign)")
    instructions: Optional[str] = Field(None, description="синоним instruct")
    ref_audio: Optional[Union[str, List[str]]] = Field(
        None, description="эталон для клонирования: путь, URL, base64 или data-URI"
    )
    ref_text: Optional[str] = Field(None, description="транскрипт эталона (нужен для качественного клона)")
    x_vector_only_mode: Optional[bool] = Field(None, description="клонировать только тембр (без транскрипта)")
    language: Optional[str] = Field(None, description="Auto | Russian | English | Chinese | ...")
    model: Optional[str] = Field(None, description="ключ модели из config.json (models[].key)")
    task_type: Optional[str] = Field(None, description="clone | preset | design (обычно определяется автоматически)")
    format: Optional[str] = Field(None, description="wav | pcm | mp3 | ogg | flac | opus")
    response_format: Optional[str] = Field(None, description="синоним format (OpenAI-стиль)")
    sample_rate: Optional[int] = Field(None, ge=8000, le=96000)
    speed: Optional[float] = Field(None, ge=0.5, le=2.0)
    stream: Optional[bool] = Field(None, description="true = отдавать аудио чанками (chunked transfer / WS)")
    seed: Optional[int] = None
    temperature: Optional[float] = Field(None, ge=0.01, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0, le=200)
    repetition_penalty: Optional[float] = Field(None, ge=0.5, le=3.0)
    max_new_tokens: Optional[int] = Field(None, ge=16, le=16384)
    do_sample: Optional[bool] = None

    @model_validator(mode="after")
    def _require_text(self) -> "TTSRequest":
        t = (self.text or self.input or "").strip()
        if not t:
            raise ValueError("нужно передать 'text' (или 'input' в OpenAI-стиле) — непустую строку")
        self.text = t
        return self

    def effective_text(self) -> str:
        return (self.text or self.input or "").strip()

    def effective_format(self) -> str:
        return (self.format or self.response_format or "wav").lower()

    def effective_instruct(self) -> Optional[str]:
        return self.instruct or self.instructions

    def gen_overrides(self) -> Dict[str, Any]:
        keys = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "do_sample", "seed")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


class ProfileCreate(_Base):
    name: str = Field(..., min_length=1, max_length=80)
    kind: Literal["clone", "preset", "designed"] = "clone"
    ref_text: Optional[str] = Field(None, max_length=4000, description="транскрипт эталонной записи")
    instruct: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    x_vector_only: bool = False
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    model_key: Optional[str] = None
    audio: Optional[str] = Field(None, description="base64/data-URI/URL/путь на сервере (для JSON-запроса)")
    trim: bool = True
    normalize: bool = True
    profile_id: Optional[str] = None


class ProfileUpdate(_Base):
    name: Optional[str] = None
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    x_vector_only: Optional[bool] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    kind: Optional[str] = None
    audio: Optional[str] = None


class TestRequest(_Base):
    text: str = "Проверка режима работы: видеокарта определена, синтез запущен."
    voice: VoiceLike = None
    language: Optional[str] = None


class ConfigPatch(_Base):
    server: Optional[Dict[str, Any]] = None
    device: Optional[Dict[str, Any]] = None
    generation: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    profiles: Optional[Dict[str, Any]] = None
    log: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    models: Optional[List[Dict[str, Any]]] = None

    def as_patch(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump(exclude_none=True).items()}


class ModelLoadRequest(_Base):
    model_key: Optional[str] = None
    force: bool = False


class ModelAddRequest(_Base):
    key: str = Field(..., min_length=1, max_length=40)
    source: str = Field(..., min_length=1)
    kind: str = "auto"
    label: str = ""
    local_files_only: Optional[bool] = None


class BackendRequest(_Base):
    backend: Literal["qwen_tts", "vllm_omni", "dummy"]


class EnvApplyRequest(_Base):
    env: Dict[str, str] = Field(default_factory=dict)
    preset: Optional[str] = None
    overwrite: bool = True
