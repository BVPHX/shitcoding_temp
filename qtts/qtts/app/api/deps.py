"""Общие зависимости роутов: доступ к сервису, авторизация, клиент."""

from __future__ import annotations

import hmac
import time
from typing import Optional

from fastapi import HTTPException, Request, WebSocket, WebSocketException, status
from starlette.requests import HTTPConnection

from ..service import TTSService


def get_service(conn: HTTPConnection) -> TTSService:
    svc: Optional[TTSService] = getattr(conn.app.state, "service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="сервис ещё не инициализирован")
    return svc


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _extract_token(conn) -> str:
    """Достаёт ключ из заголовков или query-параметра (работает и для HTTP, и для WS)."""
    headers = conn.headers
    given = headers.get("x-api-key") or ""
    if not given:
        auth = headers.get("authorization") or ""
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token", "apikey"):
            given = parts[1]
    if not given:
        try:
            given = conn.query_params.get("api_key") or conn.query_params.get("key") or ""
        except Exception:  # noqa: BLE001
            given = ""
    return str(given)


def require_auth(conn: HTTPConnection) -> None:
    """Проверка ключа. Пропускаем, если server.api_key в конфиге пустой.

    Принимает: `X-API-Key: <key>` | `Authorization: Bearer <key>` | `?api_key=<key>`.
    Веб-GUI получает одноразовый ui_token (встраивается в index.html при отдаче),
    поэтому браузеру не нужно знать основной ключ.
    """
    cfg = getattr(conn.app.state, "cfg", None)
    expected = (cfg.server.api_key if cfg else "") or ""
    if not expected:
        return
    ui_token = getattr(conn.app.state, "ui_token", "") or ""
    given = _extract_token(conn)
    if ui_token and hmac.compare_digest(given, str(ui_token)):
        return
    if given and hmac.compare_digest(given, expected):
        return
    if isinstance(conn, WebSocket):
        # для WS нельзя отдать 401 — закрываем соединение с кодом policy violation
        raise WebSocketException(code=1008, reason="требуется API-ключ (X-API-Key / ?api_key=)")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="требуется API-ключ: заголовок X-API-Key или Authorization: Bearer <key>",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def ws_require_auth(ws: WebSocket) -> None:
    """Тот же require_auth, но с явным типом для WebSocket-роутов."""
    require_auth(ws)


def request_id(request: Request) -> str:
    return getattr(request.state, "rid", "-")


def now() -> float:
    return time.time()
