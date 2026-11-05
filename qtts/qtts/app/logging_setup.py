"""Логирование: цветная консоль + файл с ротацией + ring-buffer для веб-GUI.

В терминал попадают:
  * статус сервера (старт/стоп, загрузка модели, переключение бэкенда);
  * внешние запросы (метод, путь, клиент, код ответа, длительность);
  * этапы генерации (чанк i/N, prefill, декод, TTFA, RTF, ток/с);
  * диагностика ROCm;
  * ошибки и трейсбеки.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .config import LogCfg

APP_LOGGER = "qtts"
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("qtts_rid", default="-")

_COLORS = {
    "TRACE": "\033[38;5;244m",
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "SUCCESS": "\033[1;32m",
    "WARN": "\033[33m",
    "WARNING": "\033[33m",
    "ERROR": "\033[1;31m",
    "CRITICAL": "\033[1;97;41m",
    "REQ": "\033[35m",
    "GEN": "\033[1;34m",
    "GPU": "\033[1;36m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"

TRACE = 5
SUCCESS = 25
REQ = 26
GEN = 27
GPU = 28
logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(REQ, "REQ")
logging.addLevelName(GEN, "GEN")
logging.addLevelName(GPU, "GPU")


# --------------------------------------------------------------------------- #
# Шина логов (для GUI и /api/logs/stream)
# --------------------------------------------------------------------------- #


@dataclass
class LogRecordLite:
    ts: float
    level: str
    name: str
    message: str
    rid: str = "-"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ts": round(self.ts, 3),
            "time": time.strftime("%H:%M:%S", time.localtime(self.ts)),
            "level": self.level,
            "name": self.name,
            "msg": self.message,
            "rid": self.rid,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


class LogBus:
    """Кольцевой буфер записей + подписчики (каждая подписка — своя очередь)."""

    def __init__(self, maxlen: int = 2000) -> None:
        self._buf: Deque[LogRecordLite] = deque(maxlen=maxlen)
        self._subs: List["queue.Queue[Optional[LogRecordLite]]"] = []
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = {}

    def publish(self, rec: LogRecordLite) -> None:
        with self._lock:
            self._buf.append(rec)
            self.counters[rec.level] = self.counters.get(rec.level, 0) + 1
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(rec)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(rec)
                except Exception:  # noqa: BLE001
                    pass

    def tail(self, n: int = 200, level: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        if level:
            items = [r for r in items if r.level == level.upper()]
        return [r.to_dict() for r in items[-n:]]

    def subscribe(self, maxsize: int = 1000) -> "queue.Queue[Optional[LogRecordLite]]":
        q: "queue.Queue[Optional[LogRecordLite]]" = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[Optional[LogRecordLite]]") -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


_BUS: Optional[LogBus] = None


def bus() -> LogBus:
    global _BUS
    if _BUS is None:
        _BUS = LogBus()
    return _BUS


# --------------------------------------------------------------------------- #
# Форматтеры и хендлеры
# --------------------------------------------------------------------------- #


class ColorFormatter(logging.Formatter):
    def __init__(self, color: bool = True) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S")
        self.color = color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def format(self, record: logging.LogRecord) -> str:
        rid = getattr(record, "rid", None) or _request_id.get()
        lvl = record.levelname
        msg = record.getMessage()
        if self.color:
            c = _COLORS.get(lvl, "")
            line = (
                f"{_DIM}{time.strftime('%H:%M:%S', time.localtime(record.created))}{_RESET} "
                f"{c}{lvl:<7}{_RESET} {_DIM}{rid:>6}{_RESET} {c}{msg}{_RESET}"
            )
        else:
            line = f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {lvl:<7} {rid:>6} {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "rid": getattr(record, "rid", None) or _request_id.get(),
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in ("args", "msg", "levelname", "levelno", "pathname", "filename", "module",
                     "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created",
                     "msecs", "relativeCreated", "thread", "threadName", "processName",
                     "process", "name", "taskName", "rid"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except Exception:  # noqa: BLE001
                payload[k] = str(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class BusHandler(logging.Handler):
    """Перекладывает записи в LogBus (для веб-GUI)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rec = LogRecordLite(
                ts=record.created,
                level=record.levelname,
                name=record.name.replace(APP_LOGGER + ".", ""),
                message=record.getMessage(),
                rid=getattr(record, "rid", None) or _request_id.get(),
                extra={
                    k: v
                    for k, v in record.__dict__.items()
                    if k in ("job", "model", "voice", "chunk", "chunks", "ms", "rtf", "ttfa", "status", "path")
                },
            )
            if record.exc_info:
                rec.message += " | " + self.format(record).splitlines()[-1]
            bus().publish(rec)
        except Exception:  # noqa: BLE001
            self.handleError(record)


# --------------------------------------------------------------------------- #
# Инициализация
# --------------------------------------------------------------------------- #

_CONFIGURED = False


def setup_logging(cfg: LogCfg, force: bool = False) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return logging.getLogger(APP_LOGGER)

    global _BUS
    _BUS = LogBus(maxlen=max(50, int(cfg.ring_size)))
    level = getattr(logging, str(cfg.level).upper(), logging.INFO)
    root.setLevel(min(level, TRACE))

    for h in list(root.handlers):
        root.removeHandler(h)

    if cfg.console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(JsonFormatter() if cfg.json_console else ColorFormatter(cfg.color))
        root.addHandler(sh)

    if cfg.file:
        from .config import HOME_DIR

        p = cfg.file if os.path.isabs(cfg.file) else str(HOME_DIR / cfg.file)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            p, maxBytes=cfg.max_bytes, backupCount=cfg.backups, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s [%(process)d] %(message)s")
        )
        root.addHandler(fh)

    bh = BusHandler()
    bh.setLevel(level)
    root.addHandler(bh)

    # uvicorn: перехватываем в нашу схему (иначе он печатает своим форматом)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(level)

    # uvicorn.access дублирует наши REQ-строки (там же client/path/код/тайминг + request-id),
    # поэтому при включённом журнале запросов гасим его — иначе каждая строка печатается дважды.
    # Заодно убираем служебный шум WS-подключений из uvicorn.error: о них мы пишем сами
    # («WS /ws/tts подключён/отключён client=...»), а важные события ([rejected], ошибки) остаются.
    if cfg.log_requests:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        uerr = logging.getLogger("uvicorn.error")
        if not any(isinstance(f, _WsNoiseFilter) for f in uerr.filters):
            uerr.addFilter(_WsNoiseFilter())   # не дублируем фильтр при force-перезапуске

    # приглушаем болтливые библиотеки
    # websockets/wsproto печатают «connection open/closed» на каждый WS-запрос — у нас есть свои REQ-строки
    for noisy in ("httpx", "httpcore", "huggingface_hub", "transformers", "filelock", "urllib3", "asyncio",
                  "websockets", "websockets.server", "wsproto"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger(APP_LOGGER)


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{APP_LOGGER}.{name}" if name else APP_LOGGER)


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


# --------------------------------------------------------------------------- #
# Перехват логов uvicorn'а -> наш формат с client/path
# --------------------------------------------------------------------------- #


class _WsNoiseFilter(logging.Filter):
    """Режет служебные строки uvicorn про WS-соединения (дублируются нашими REQ-записями)."""

    _NOISE = ("[accepted]", "connection open", "connection closed")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        return not any(n in msg for n in self._NOISE)


def log_banner(lines: List[str], log: Optional[logging.Logger] = None) -> None:
    log = log or get_logger("boot")
    width = max((len(x) for x in lines), default=40) + 4
    log.info("┌" + "─" * width + "┐")
    for ln in lines:
        log.info("│ " + ln.ljust(width - 2) + " │")
    log.info("└" + "─" * width + "┘")


def subscribe(maxsize: int = 1000) -> "queue.Queue[Optional[LogRecordLite]]":
    """Подписаться на поток записей (используется WebSocket /api/logs/ws)."""
    return bus().subscribe(maxsize=maxsize)
