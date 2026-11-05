"""Конфигурация приложения.

Приоритет (сверху вниз):
    1. аргументы командной строки (--host, --port, ...)
    2. переменные окружения  QTTSS_SERVER__PORT=9000  (двойное подчёркивание = вложенность)
    3. файл config.json (или QTTSS_CONFIG=/path/to/config.json)
    4. значения по умолчанию из этого модуля
"""

from __future__ import annotations

import copy
import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_DIR = Path(__file__).resolve().parent.parent  # каталог ПАКЕТА (в site-packages может быть read-only)


def home_dir() -> Path:
    """Домашний каталог приложения: конфиг, профили голосов, результаты, журналы.

    Порядок поиска:
      1. `$QTTSS_HOME` — явное указание (используется в systemd-юните и в тестах);
      2. `$XDG_DATA_HOME/qtts` — стандарт Linux для пользовательских данных;
      3. `~/.local/share/qtts` — если XDG_DATA_HOME не задан.

    В каталог пакета намеренно не пишем: при установке через pip он может быть
    доступен только для чтения, а данные должны переживать обновление версии.
    """
    env = os.environ.get("QTTSS_HOME")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "qtts").resolve()


HOME_DIR = home_dir()
DEFAULT_CONFIG_PATH = Path(os.environ.get("QTTSS_CONFIG") or (HOME_DIR / "config.json"))
# Фактически использованный путь конфига: save_config() без явного пути пишет туда же,
# откуда конфиг загружали (учитывает --config и QTTSS_CONFIG), а не в дефолт.
_ACTIVE_CONFIG_PATH: Path = DEFAULT_CONFIG_PATH
DEFAULT_DATA_DIR = HOME_DIR

ENV_PREFIX = "QTTSS_"

# --------------------------------------------------------------------------- #
# Структура конфига
# --------------------------------------------------------------------------- #


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"          # 0.0.0.0 — для доступа из сети (не забудьте api_key!)
    port: int = 8770
    api_key: str = ""                # пустая строка = авторизация выключена
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    workers: int = 1                 # модель одна, GPU один — больше не нужно
    preload_model: bool = False      # грузить модель сразу при старте сервера
    open_browser: bool = False
    ssl_certfile: str = ""
    ssl_keyfile: str = ""
    proxy_headers: bool = True       # за reverse-proxy (nginx) — читать X-Forwarded-*
    max_upload_mb: int = 64          # потолок на загружаемый эталон голоса
    shutdown_timeout_sec: int = 10   # сколько ждём текущие генерации перед выходом


@dataclass
class ModelCfg:
    key: str = "model"
    source: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # repo id на HuggingFace ИЛИ локальный каталог
    kind: str = "auto"               # base | custom_voice | voice_design | auto
    label: str = ""                  # человекочитаемое имя для GUI
    tokenizer_source: str = ""       # обычно не нужно: токенайзер лежит внутри чекпойнта
    revision: str = "main"
    cache_dir: str = ""              # кэш HuggingFace (пусто = стандартный ~/.cache/huggingface)
    local_files_only: bool = False   # True = не ходить в сеть, только локальные файлы
    trust_remote_code: bool = False


@dataclass
class DeviceCfg:
    backend: str = "qwen_tts"        # qwen_tts | vllm_omni | dummy
    device: str = "auto"             # auto | cuda:0 (= ROCm/HIP) | cpu
    dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    attn: str = "auto"               # auto | sdpa | flash_attention_2 | eager
    env: Dict[str, str] = field(default_factory=dict)   # применить ПЕРЕД импортом torch
    gpu_memory_utilization: float = 0.9
    allow_tf32: bool = True
    matmul_precision: str = "high"   # highest | high | medium
    deterministic: bool = False
    vllm_base_url: str = "http://127.0.0.1:8091"


@dataclass
class GenerationCfg:
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    subtalker_do_sample: bool = True
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    subtalker_temperature: float = 0.9
    seed: Optional[int] = None
    max_new_tokens: int = 4096        # жёсткий потолок
    min_new_tokens: int = 8
    tokens_per_char: float = 4.0      # бюджет токенов = min(max_new_tokens, min_new + chars*tokens_per_char)
    language: str = "Auto"            # язык по умолчанию (Auto = модель определит сама)
    speed: float = 1.0                # 1.0 = как есть (меняется длиной пауз/склейкой чанков)
    streaming: bool = True
    # non_streaming_mode=False включает «симуляцию» потокового входа Dual-Track (медленнее,
    # чуть естественнее просодия на длинных фразах). Для чанкового стриминга лучше True.
    non_streaming_mode: bool = True
    chunking: str = "sentence"        # off | sentence | paragraph
    chunk_target_chars: int = 220
    chunk_min_chars: int = 40
    chunk_max_chars: int = 700
    inter_chunk_silence_ms: int = 90
    warmup: bool = True
    warmup_text: str = "Проверка синтеза."
    compile: bool = False             # torch.compile на ROCm обычно даёт долгие рекомпиляции
    max_concurrent: int = 1
    queue_limit: int = 32
    request_timeout_sec: float = 900.0
    clone: Dict[str, Any] = field(default_factory=lambda: {"x_vector_only_mode": False, "prompt_cache": True})


@dataclass
class AudioCfg:
    format: str = "wav"              # wav | pcm | mp3 | ogg | flac
    sample_rate: int = 0             # 0 = нативная частота модели (24000 Гц)
    channels: int = 1
    sample_width: int = 2            # 16-bit PCM
    normalize: bool = True
    peak: float = 0.97
    save_outputs: bool = True
    outputs_dir: str = "outputs"
    keep_outputs: int = 300
    output_sr_for_clone_ref: int = 24000


@dataclass
class ProfilesCfg:
    dir: str = "profiles"
    ref_min_sec: float = 2.0
    ref_max_sec: float = 30.0
    ref_target_sec: float = 10.0
    prompt_cache_size: int = 4


@dataclass
class LogCfg:
    level: str = "INFO"
    file: str = "logs/qtts.log"
    max_bytes: int = 20_000_000
    backups: int = 5
    console: bool = True
    color: bool = True
    json_console: bool = False
    ring_size: int = 2000
    log_requests: bool = True
    log_health: bool = False
    log_generation: bool = True
    log_devices: bool = True


@dataclass
class DataCfg:
    dir: str = str(DEFAULT_DATA_DIR)


@dataclass
class Config:
    server: ServerCfg = field(default_factory=ServerCfg)
    device: DeviceCfg = field(default_factory=DeviceCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    profiles: ProfilesCfg = field(default_factory=ProfilesCfg)
    log: LogCfg = field(default_factory=LogCfg)
    data: DataCfg = field(default_factory=DataCfg)
    models: List[ModelCfg] = field(default_factory=lambda: [ModelCfg()])

    # ---------------- производные пути ---------------- #
    @property
    def data_path(self) -> Path:
        return _abs(Path(self.data.dir))

    @property
    def profiles_path(self) -> Path:
        return self.data_path / self.profiles.dir

    @property
    def outputs_path(self) -> Path:
        return self.data_path / self.audio.outputs_dir

    @property
    def log_path(self) -> Optional[Path]:
        if not self.log.file:
            return None
        p = Path(self.log.file)
        return p if p.is_absolute() else (HOME_DIR / p)

    def model(self, key: Optional[str] = None) -> ModelCfg:
        if not self.models:
            raise ValueError("В конфиге нет ни одной модели (секция models)")
        if not key:
            return self.models[0]
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"Модель '{key}' не найдена. Доступны: {[m.key for m in self.models]}")

    def model_keys(self) -> List[str]:
        return [m.key for m in self.models]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self) -> "Config":
        return _from_dict(self.to_dict())


# --------------------------------------------------------------------------- #
# Сериализация
# --------------------------------------------------------------------------- #


def _coerce(value: Any, target: Any) -> Any:
    """Привести value к типу поля target (поддержка Optional и Dict[str,str])."""
    if target is Any or target is None:
        return value
    origin = getattr(target, "__origin__", None)
    if origin is Optional or origin is type(Optional[int]):  # Optional[X]
        args = [a for a in target.__args__ if a is not type(None)]  # noqa: E721
        if value is None:
            return None
        return _coerce(value, args[0])
    if origin in (list, List):
        inner = target.__args__[0]
        if not isinstance(value, list):
            raise TypeError(f"ожидался список, получено {type(value).__name__}")
        return [_coerce(v, inner) for v in value]
    if origin in (dict, Dict):
        return dict(value)
    if target is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "да")
        return bool(value)
    if target in (int, float, str):
        return target(value)
    return value


_UNKNOWN: List[str] = []


def _warn_unknown(key: str) -> None:
    if key not in _UNKNOWN:
        _UNKNOWN.append(key)


def _from_dict(data: Dict[str, Any]) -> Config:
    cfg = Config()
    for f in fields(cfg):
        if f.name not in data:
            continue
        raw = data[f.name]
        cur = getattr(cfg, f.name)
        if f.name == "models":
            models: List[ModelCfg] = []
            for i, item in enumerate(raw or []):
                if isinstance(item, str):
                    item = {"source": item, "key": f"model{i}"}
                m = ModelCfg()
                for k, v in dict(item).items():
                    if hasattr(m, k):
                        setattr(m, k, v)
                    else:
                        _warn_unknown(f"models[].{k}")
                if not m.key:
                    m.key = f"model{len(models)}"
                if not m.label:
                    m.label = m.key
                models.append(m)
            cfg.models = models or [ModelCfg()]
            continue
        if is_dataclass(cur):
            for k, v in dict(raw or {}).items():
                if hasattr(cur, k):
                    hint = type(getattr(cur, k))
                    try:
                        setattr(cur, k, _coerce(v, hint) if hint in (bool, int, float, str) else v)
                    except Exception:
                        setattr(cur, k, v)
                else:
                    _warn_unknown(f"{f.name}.{k}")
        else:
            setattr(cfg, f.name, raw)
    return cfg


def _abs(p: Path) -> Path:
    """Относительные пути в конфиге считаем от домашнего каталога приложения."""
    return p if p.is_absolute() else (HOME_DIR / p)


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #

_LOCK = threading.RLock()


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_dict() -> Dict[str, Any]:
    return asdict(Config())


def active_config_path() -> Path:
    """Путь, который load_config() использовал последним (для сохранения «туда же»)."""
    return _ACTIVE_CONFIG_PATH


def load_config(path: Optional[Path | str] = None) -> Config:
    """Загрузить config.json (если есть) и наложить переменные окружения QTTSS_*."""
    global _ACTIVE_CONFIG_PATH
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    _ACTIVE_CONFIG_PATH = p
    data: Dict[str, Any] = {}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception as exc:  # noqa: BLE001
            _warn_unknown(f"config parse error: {exc}")
            data = {}
    data.setdefault("data", {})["dir"] = data.get("data", {}).get("dir") or os.environ.get(
        "QTTSS_DATA_DIR", str(DEFAULT_DATA_DIR)
    )
    data = _deep_merge(default_dict(), data)
    data = _deep_merge(data, _env_overrides())
    cfg = _from_dict(data)
    cfg.config_path = p          # runtime-атрибут (не поле dataclass → в to_dict не попадает)
    _ensure_dirs(cfg)
    return cfg


def _env_overrides() -> Dict[str, Any]:
    """QTTSS_SERVER__PORT=9000 -> {"server": {"port": "9000"}}"""
    out: Dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(ENV_PREFIX) or key in ("QTTSS_CONFIG", "QTTSS_DATA_DIR"):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if isinstance(node, dict):
            node[parts[-1]] = val
    return out


def save_config(cfg: Config, path: Optional[Path | str] = None) -> Path:
    global _ACTIVE_CONFIG_PATH
    p = Path(path or getattr(cfg, "config_path", None) or _ACTIVE_CONFIG_PATH)
    _ACTIVE_CONFIG_PATH = p
    cfg.config_path = p
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return p


def ensure_dirs(cfg: Config) -> None:
    _ensure_dirs(cfg)


def _ensure_dirs(cfg: Config) -> None:
    for p in (cfg.data_path, cfg.profiles_path, cfg.outputs_path):
        p.mkdir(parents=True, exist_ok=True)
    lp = cfg.log_path
    if lp is not None:
        lp.parent.mkdir(parents=True, exist_ok=True)


def unknown_keys_warning() -> List[str]:
    return list(_UNKNOWN)


def apply_cli_overrides(cfg: Config, args) -> Config:
    """Небольшой набор CLI-флагов, которые перекрывают конфиг."""
    if getattr(args, "host", None):
        cfg.server.host = args.host
    if getattr(args, "port", None):
        cfg.server.port = int(args.port)
    if getattr(args, "api_key", None):
        cfg.server.api_key = args.api_key
    if getattr(args, "model", None):
        cfg.models[0].source = args.model
    if getattr(args, "device", None):
        cfg.device.device = args.device
    if getattr(args, "dtype", None):
        cfg.device.dtype = args.dtype
    if getattr(args, "backend", None):
        cfg.device.backend = args.backend
    if getattr(args, "attn", None):
        cfg.device.attn = args.attn
    if getattr(args, "cpu", False):
        cfg.device.device = "cpu"
    if getattr(args, "no_stream", False):
        cfg.generation.streaming = False
    if getattr(args, "log_level", None):
        cfg.log.level = args.log_level
    if getattr(args, "preload", False):
        cfg.server.preload_model = True
    if getattr(args, "data_dir", None):
        cfg.data.dir = args.data_dir
        _ensure_dirs(cfg)
    return cfg


with _LOCK:
    _CURRENT: Optional[Config] = None


def set_current(cfg: Config) -> Config:
    global _CURRENT
    with _LOCK:
        _CURRENT = cfg
    return cfg


def get_current() -> Optional[Config]:
    with _LOCK:
        return _CURRENT
