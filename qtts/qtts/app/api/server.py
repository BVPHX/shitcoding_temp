"""Сборка FastAPI-приложения: lifespan, middleware, роуты, веб-GUI."""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from qtts import __version__ as APP_VERSION

from .. import diagnostics
from ..config import Config, ensure_dirs
from ..engine.base import EngineError
from ..logging_setup import GPU, get_logger, log_banner, setup_logging
from ..profiles import ProfileError
from ..service import TTSService
from .middleware import RequestContextMiddleware
from .routes_profiles import router as profiles_router
from .routes_system import protected as system_protected
from .routes_system import router as system_router
from .routes_tts import router as tts_router

log = get_logger("server")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _local_ips() -> list:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:  # noqa: BLE001
        pass
    return sorted(ips)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Config = app.state.cfg
    svc: TTSService = app.state.service
    app.state.started_at = time.time()

    ensure_dirs(cfg)
    log_banner(
        [
            f"qwen-tts-studio v{APP_VERSION}",
            f"backend: {cfg.device.backend}   device: {cfg.device.device}   dtype: {cfg.device.dtype}",
            f"модель:  {cfg.models[0].key} ← {cfg.models[0].source}",
            f"адрес:   http://{cfg.server.host}:{cfg.server.port}  (GUI: /)",
            f"данные:  {cfg.data_path}",
        ]
    )
    for ln in diagnostics.short_report(
        await asyncio.to_thread(lambda: diagnostics.collect(force=True, bench=False))
    ).splitlines():
        log.log(GPU, ln)

    if cfg.server.preload_model:
        log.info("предзагрузка модели включена — запускаю в фоне")

        async def _preload() -> None:
            try:
                await asyncio.to_thread(svc.load_model)
            except EngineError as exc:
                log.error("предзагрузка модели не удалась: %s", exc)
            except Exception:  # noqa: BLE001
                log.exception("предзагрузка модели упала")

        app.state.preload_task = asyncio.create_task(_preload())

    log.info("сервер готов принимать запросы")
    try:
        yield
    finally:
        log.info("остановка сервера: выгружаю модель и закрываю движок")
        try:
            await asyncio.to_thread(svc.engine.shutdown)
        except Exception:  # noqa: BLE001
            log.exception("ошибка при остановке движка")
        log.info("сервер остановлен")


def create_app(cfg: Config) -> FastAPI:
    setup_logging(cfg.log)
    ensure_dirs(cfg)

    app = FastAPI(
        title="Qwen TTS Studio",
        version=APP_VERSION,
        description=(
            "Локальный TTS-сервер на Qwen3-TTS (1.7B/0.6B) с оптимизацией под AMD ROCm.\n\n"
            "**Основные эндпоинты**\n"
            "* `POST /api/tts` — синтез (JSON → аудио, `stream=true` для потоковой отдачи)\n"
            "* `POST /v1/audio/speech` — OpenAI-совместимый эндпоинт\n"
            "* `WS /ws/tts` — стриминг аудио + событий\n"
            "* `POST /api/profiles` — профили голосов (образцы для клонирования)\n"
            "* `GET /api/system/diagnostics` — проверка ROCm/GPU\n"
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.state.cfg = cfg
    app.state.version = APP_VERSION
    app.state.service = TTSService(cfg)
    app.state.ui_token = secrets.token_urlsafe(24)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Job-Id", "X-Audio-Duration", "X-TTS-RTF", "X-TTS-TTFA-ms",
            "X-TTS-Elapsed-ms", "X-TTS-Chunks", "X-Request-ID",
        ],
    )
    app.add_middleware(RequestContextMiddleware, log_requests=cfg.log.log_requests,
                       log_health=cfg.log.log_health)

    # ------------------------------- ошибки ------------------------------- #

    @app.exception_handler(EngineError)
    async def _engine_error(request: Request, exc: EngineError):
        status = {
            "queue_full": 429, "gpu_unavailable": 503, "no_torch": 503, "no_qwen_tts": 503,
            "model_load_failed": 503, "vllm_unreachable": 502, "cancelled": 499,
            "empty_text": 422, "no_reference": 422, "no_speaker": 422, "no_instruct": 422,
        }.get(exc.code, 500)
        return JSONResponse(status_code=status, content={"error": str(exc), "code": exc.code, **(exc.details or {})})

    @app.exception_handler(ProfileError)
    async def _profile_error(request: Request, exc: ProfileError):
        return JSONResponse(status_code=404, content={"error": str(exc), "code": "profile_error"})

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errs = []
        for e in exc.errors():
            errs.append({"field": ".".join(str(x) for x in e.get("loc", [])), "msg": e.get("msg"),
                         "type": e.get("type")})
        log.warning("ошибка валидации запроса %s: %s", request.url.path, errs)
        return JSONResponse(status_code=422, content={"error": "некорректный запрос", "details": errs})

    # -------------------------------- роуты -------------------------------- #

    app.include_router(system_router)
    app.include_router(tts_router)
    app.include_router(profiles_router)
    app.include_router(system_protected)

    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        return HTMLResponse(_render_index(request))

    @app.get("/docs", include_in_schema=False)
    async def docs_redirect():
        return RedirectResponse(url="/api/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        f = WEB_DIR / "favicon.svg"
        if f.exists():
            return FileResponse(str(f), media_type="image/svg+xml")
        return JSONResponse({}, status_code=204)

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"ok": True}

    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=False), name="ui")
    return app


def _render_index(request: Request) -> str:
    app = request.app
    cfg: Config = app.state.cfg
    index = WEB_DIR / "index.html"
    if not index.exists():
        return "<h1>qwen-tts-studio</h1><p>web/index.html не найден</p>"
    html = index.read_text(encoding="utf-8")
    boot: Dict[str, Any] = {
        "version": APP_VERSION,
        "uiToken": app.state.ui_token,
        "apiKeySet": bool(cfg.server.api_key),
        "backend": cfg.device.backend,
        "model": cfg.models[0].source if cfg.models else "",
        "host": cfg.server.host,
        "port": cfg.server.port,
        "ips": _local_ips(),
        "streaming": cfg.generation.streaming,
    }
    return html.replace("/*__BOOT__*/", f"window.__QTTS_BOOT__ = {json.dumps(boot, ensure_ascii=False)};")
