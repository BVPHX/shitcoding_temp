"""Middleware: id запроса, лог внешних обращений в терминал, тайминги."""

from __future__ import annotations

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..logging_setup import REQ, get_logger, set_request_id

log = get_logger("http")

# Пути, которые не стоит печатать на каждый чих (опрашиваются GUI раз в секунду).
QUIET_PATHS = ("/api/system/gpu", "/api/health", "/api/jobs/active")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Присваивает каждому запросу короткий id и логирует его в терминал."""

    def __init__(self, app, log_requests: bool = True, log_health: bool = False) -> None:
        super().__init__(app)
        self.log_requests = log_requests
        self.log_health = log_health

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = uuid.uuid4().hex[:6]
        request.state.rid = rid
        set_request_id(rid)
        t0 = time.perf_counter()
        path = request.url.path
        client = request.client.host if request.client else "?"
        ua = request.headers.get("user-agent", "")[:60]

        quiet = path in QUIET_PATHS or (path == "/api/health" and not self.log_health)
        if self.log_requests and not quiet:
            log.log(REQ, "→ %s %s  client=%s%s", request.method, path, client,
                    f"  ua={ua}" if log.isEnabledFor(10) else "")

        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            if self.log_requests and not quiet:
                log.error("✖ %s %s → 500 (%.0f мс) %s: %s", request.method, path, ms,
                          type(exc).__name__, str(exc)[:200])
            raise
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = rid
        if self.log_requests and not quiet:
            streamed = type(response).__name__ in ("StreamingResponse", "EventSourceResponse")
            log.log(
                REQ,
                "← %s %s → %d (%.0f мс)%s",
                request.method, path, response.status_code, ms,
                " [stream]" if streamed else "",
            )
        return response
