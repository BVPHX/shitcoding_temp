"""Очередь задач синтеза, их статусы и метрики.

Задача (job) — единица работы: один HTTP/WS-запрос на синтез. Всё, что происходит с задачей,
видно в терминале (этапы генерации), в GUI (вкладка «Журнал») и через API (/api/jobs, /api/jobs/{id}).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .engine.base import SynthesisRequest
from .logging_setup import get_logger

log = get_logger("jobs")

STATES = ("queued", "running", "streaming", "done", "error", "cancelled")


@dataclass
class Job:
    id: str
    request: SynthesisRequest
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    client: str = ""
    endpoint: str = ""
    stage: str = "queued"
    progress: float = 0.0
    chunks_done: int = 0
    chunks_total: int = 0
    audio_sec: float = 0.0
    bytes_out: int = 0
    sample_rate: int = 0
    format: str = "wav"
    ttfa_ms: Optional[float] = None          # time-to-first-audio
    gen_ms: float = 0.0
    rtf: Optional[float] = None              # real-time factor (время счёта / длительность аудио)
    tok_per_sec: float = 0.0
    model_key: Optional[str] = None
    voice: str = ""
    language: str = ""
    error: Optional[str] = None
    error_code: Optional[str] = None
    output_file: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def touch(self, stage: str, **kw: Any) -> None:
        self.stage = stage
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.events.append({"ts": time.time(), "stage": stage, **kw})
        if len(self.events) > 200:
            self.events = self.events[-200:]

    def elapsed_ms(self) -> float:
        end = self.finished_at or time.time()
        return round((end - (self.started_at or self.created_at)) * 1000, 1)

    def to_dict(self, with_events: bool = False, with_text: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "state": self.state,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": self.elapsed_ms(),
            "client": self.client,
            "endpoint": self.endpoint,
            "model": self.model_key,
            "voice": self.voice,
            "language": self.language,
            "format": self.format,
            "chunks_done": self.chunks_done,
            "chunks_total": self.chunks_total,
            "audio_sec": round(self.audio_sec, 3),
            "bytes_out": self.bytes_out,
            "sample_rate": self.sample_rate,
            "ttfa_ms": self.ttfa_ms,
            "gen_ms": round(self.gen_ms, 1),
            "rtf": self.rtf,
            "tok_per_sec": self.tok_per_sec,
            "error": self.error,
            "error_code": self.error_code,
            "output_file": self.output_file,
        }
        if with_text:
            d["text"] = self.request.text
            d["chars"] = len(self.request.text)
        if with_events:
            d["events"] = self.events
        return d


class JobStore:
    def __init__(self, max_history: int = 300) -> None:
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._subs: List[Any] = []
        self.max_history = max_history

    def create(self, req: SynthesisRequest, client: str = "", endpoint: str = "") -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, request=req, client=client, endpoint=endpoint, format=req.format)
        job.touch("queued", model_key=req.model_key, voice=req.voice.describe(),
                  language=req.language or "", format=req.format)
        with self._lock:
            self._jobs[jid] = job
            while len(self._jobs) > self.max_history:
                self._jobs.popitem(last=False)
        self.publish(job)
        return job

    def get(self, jid: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(jid)

    def list(self, limit: int = 50, state: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._jobs.values())[::-1]
        if state:
            items = [j for j in items if j.state == state]
        return [j.to_dict(with_text=True) for j in items[:limit]]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            items = list(self._jobs.values())
        by_state: Dict[str, int] = {}
        for j in items:
            by_state[j.state] = by_state.get(j.state, 0) + 1
        done = [j for j in items if j.state == "done" and j.audio_sec > 0]
        rtfs = [j.rtf for j in done if j.rtf]
        ttfas = [j.ttfa_ms for j in done if j.ttfa_ms]
        return {
            "total": len(items),
            "by_state": by_state,
            "audio_sec_total": round(sum(j.audio_sec for j in items), 2),
            "avg_rtf": round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
            "avg_ttfa_ms": round(sum(ttfas) / len(ttfas), 1) if ttfas else None,
            "errors": sum(1 for j in items if j.state == "error"),
        }

    def update(self, job: Job, **kw: Any) -> Job:
        for k, v in kw.items():
            if hasattr(job, k):
                setattr(job, k, v)
        self.publish(job)
        return job

    # ------------------------------ pub/sub ------------------------------ #

    def subscribe(self):
        import queue as _q

        q: "_q.Queue[Optional[Job]]" = _q.Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, job: Job) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(job)
            except Exception:  # noqa: BLE001
                pass

    def clear(self) -> int:
        with self._lock:
            n = len(self._jobs)
            self._jobs.clear()
        return n
