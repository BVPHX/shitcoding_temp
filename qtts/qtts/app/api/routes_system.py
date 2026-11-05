"""Системные роуты: диагностика ROCm, статус, модели, настройки, журнал."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from .. import diagnostics
from ..config import Config, ModelCfg, active_config_path, save_config
from ..engine import BACKENDS
from ..engine.base import EngineError
from ..env import PRESETS, apply_env, current_relevant_env
from ..logging_setup import GPU, bus, get_logger, setup_logging, subscribe
from ..service import TTSService
from .deps import get_service, require_auth
from .routes_tts import engine_error_to_http
from .schemas import BackendRequest, ConfigPatch, EnvApplyRequest, ModelAddRequest, ModelLoadRequest, TestRequest

log = get_logger("api.system")
router = APIRouter()
protected = APIRouter(dependencies=[Depends(require_auth)])


# --------------------------------------------------------------------------- #
# Здоровье и статус
# --------------------------------------------------------------------------- #


@router.get("/api/health", summary="Жив ли сервер")
async def health(svc: TTSService = Depends(get_service)):
    st = svc.engine.status()
    return {
        "status": "ok",
        "time": time.time(),
        "uptime_sec": round(time.time() - svc.started_at, 1),
        "engine": st.name,
        "model_loaded": bool(st.loaded_models),
        "active_model": st.active_model,
        "device": st.device,
    }


@protected.get("/api/system/info", summary="Сводка: версии, конфиг, движок")
async def system_info(request: Request, svc: TTSService = Depends(get_service)):
    cfg: Config = request.app.state.cfg
    versions: Dict[str, Any] = {"app": request.app.state.version}
    for mod in ("torch", "torchaudio", "transformers", "qwen_tts", "fastapi", "uvicorn", "pydantic", "numpy",
                "librosa", "soundfile", "httpx", "vllm", "flash_attn", "triton"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "installed")
        except Exception:  # noqa: BLE001
            versions[mod] = None
    st = svc.engine.status().to_dict()
    return {
        "versions": versions,
        "config": cfg.to_dict(),
        "config_path": str(getattr(cfg, "config_path", None) or active_config_path()),
        "paths": {
            "data": str(cfg.data_path),
            "profiles": str(cfg.profiles_path),
            "outputs": str(cfg.outputs_path),
            "log": str(cfg.log_path) if cfg.log_path else None,
        },
        "engine": st,
        "env": current_relevant_env(),
        "jobs": svc.jobs.stats(),
        "profiles": len(svc.profiles.list()),
        "backends": list(BACKENDS),
    }


@protected.get("/api/system/status", summary="Статус движка и очереди")
async def system_status(svc: TTSService = Depends(get_service)):
    return svc.status()


@protected.get("/api/system/diagnostics", summary="Полная диагностика ROCm/GPU")
async def system_diagnostics(
    refresh: bool = Query(False, description="пересчитать, даже если есть кэш"),
    bench: bool = Query(True, description="выполнить микро-бенчмарк GPU"),
    svc: TTSService = Depends(get_service),
):
    log.log(GPU, "запрошена диагностика (refresh=%s, bench=%s) от %s", refresh, bench, "gui")
    try:
        rep = await asyncio.to_thread(
            lambda: diagnostics.collect(force=refresh, ttl=0.0 if refresh else 5.0, bench=bench)
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("диагностика упала")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return rep


@protected.get("/api/system/diagnostics/text", summary="Диагностика текстом (для копирования в issue)")
async def diagnostics_text(refresh: bool = Query(False), svc: TTSService = Depends(get_service)):
    rep = await asyncio.to_thread(
        lambda: diagnostics.collect(force=refresh, ttl=0.0 if refresh else 5.0, bench=True)
    )
    return PlainTextResponse(diagnostics.short_report(rep))


@protected.get("/api/system/gpu", summary="Быстрый снимок GPU (VRAM/загрузка)")
async def gpu(svc: TTSService = Depends(get_service)):
    return await asyncio.to_thread(svc.gpu_stats)


@protected.get("/api/system/readiness", summary="Готовность к синтезу + что мешает")
async def readiness(svc: TTSService = Depends(get_service)):
    rep = await asyncio.to_thread(lambda: diagnostics.collect(force=False, ttl=5.0, bench=False))
    st = svc.engine.status()
    problems = [c for c in rep.get("checks", []) if c["status"] in ("warn", "error")]
    ready = bool(rep.get("torch", {}).get("gpu_visible")) or st.name == "dummy" or st.name == "vllm_omni"
    return {
        "ready": ready and (bool(st.loaded_models) or svc.cfg.device.backend != "qwen_tts"),
        "gpu_visible": bool(rep.get("torch", {}).get("gpu_visible")),
        "engine": st.name,
        "model_loaded": bool(st.loaded_models),
        "active_model": st.active_model,
        "device": st.device,
        "rocm_version": rep.get("rocm", {}).get("version"),
        "gpu_name": (rep.get("torch", {}).get("devices") or [{}])[0].get("name"),
        "gfx": (rep.get("torch", {}).get("devices") or [{}])[0].get("gfx"),
        "problems": problems,
        "health": rep.get("health"),
    }


@protected.post("/api/system/test", summary="Проверка режима работы: диагностика + тестовый синтез")
async def system_test(body: Optional[TestRequest] = None, request: Request = None,
                      svc: TTSService = Depends(get_service)):
    """То, что нужно для кнопки «Проверить» в настройках: сначала диагностика, потом живой синтез."""
    body = body or TestRequest()
    t0 = time.perf_counter()
    rep = await asyncio.to_thread(diagnostics.collect, True, 5.0, True)
    diag_ms = round((time.perf_counter() - t0) * 1000, 1)
    result: Dict[str, Any] = {"diagnostics": rep, "diagnostics_ms": diag_ms}
    t1 = time.perf_counter()
    try:
        test = await asyncio.to_thread(svc.self_test, body.text, body.voice)
        result["test"] = test
        result["test_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        result["ok"] = True
    except EngineError as exc:
        result["ok"] = False
        result["test_error"] = {"error": str(exc), "code": exc.code}
        log.error("проверка режима работы провалена: [%s] %s", exc.code, exc)
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["test_error"] = {"error": f"{type(exc).__name__}: {exc}"}
        log.exception("проверка режима работы упала")
    return result


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #


@protected.get("/api/models", summary="Список настроенных моделей")
async def list_models(request: Request, svc: TTSService = Depends(get_service)):
    st = svc.engine.status()
    loaded = {m["key"]: m for m in st.info.get("models", [])}
    out = []
    for m in request.app.state.cfg.models:
        d = m.__dict__.copy()
        d["loaded"] = m.key in loaded
        d["info"] = loaded.get(m.key)
        out.append(d)
    return {"models": out, "active": st.active_model, "backend": st.name}


@protected.post("/api/models/load", summary="Загрузить модель в VRAM")
async def load_model(body: ModelLoadRequest, svc: TTSService = Depends(get_service)):
    try:
        meta = await asyncio.to_thread(svc.load_model, body.model_key, body.force)
    except EngineError as exc:
        raise engine_error_to_http(exc) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("ошибка загрузки модели")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"ok": True, "model": meta}


@protected.post("/api/models/unload", summary="Выгрузить модель из VRAM")
async def unload_model(body: ModelLoadRequest, svc: TTSService = Depends(get_service)):
    await asyncio.to_thread(svc.unload_model, body.model_key)
    return {"ok": True, "unloaded": body.model_key, "status": svc.engine.status().to_dict()}


@protected.post("/api/models/add", summary="Добавить модель в конфиг")
async def add_model(body: ModelAddRequest, request: Request, svc: TTSService = Depends(get_service)):
    cfg: Config = request.app.state.cfg
    if any(m.key == body.key for m in cfg.models):
        raise HTTPException(status_code=409, detail=f"модель с ключом '{body.key}' уже есть")
    m = ModelCfg(
        key=body.key, source=body.source, kind=body.kind or "auto", label=body.label or body.key,
        local_files_only=bool(body.local_files_only if body.local_files_only is not None else False),
    )
    cfg.models.append(m)
    save_config(cfg)
    svc.cfg = cfg
    log.info("в конфиг добавлена модель %s → %s", m.key, m.source)
    return {"ok": True, "models": [x.__dict__ for x in cfg.models]}


@protected.delete("/api/models/{key}", summary="Убрать модель из конфига")
async def remove_model(key: str, request: Request, svc: TTSService = Depends(get_service)):
    cfg: Config = request.app.state.cfg
    before = len(cfg.models)
    cfg.models = [m for m in cfg.models if m.key != key]
    if len(cfg.models) == before:
        raise HTTPException(status_code=404, detail=f"модель '{key}' не найдена")
    if not cfg.models:
        cfg.models = [ModelCfg()]
    save_config(cfg)
    svc.unload_model(key)
    return {"ok": True, "models": [x.__dict__ for x in cfg.models]}


# --------------------------------------------------------------------------- #
# Настройки
# --------------------------------------------------------------------------- #


@protected.get("/api/config", summary="Текущий конфиг")
async def get_config(request: Request):
    cfg: Config = request.app.state.cfg
    d = cfg.to_dict()
    if d.get("server", {}).get("api_key"):
        d["server"]["api_key"] = "***"
    return d


@protected.patch("/api/config", summary="Изменить настройки (применяется сразу + пишется в config.json)")
async def patch_config(body: ConfigPatch, request: Request, svc: TTSService = Depends(get_service)):
    patch = body.as_patch()
    if not patch:
        raise HTTPException(status_code=422, detail="пустой патч")
    cfg, restart = await asyncio.to_thread(svc.update_config, patch)
    request.app.state.cfg = cfg
    if "log" in patch:
        setup_logging(cfg.log, force=True)
        log.info("логирование перенастроено: level=%s", cfg.log.level)
    if restart:
        log.warning("для применения %s перезагрузите движок (POST /api/models/load?force) или перезапустите сервер",
                    ", ".join(restart))
    return {"ok": True, "restart_required": restart, "config": _masked(cfg.to_dict())}


@protected.post("/api/config/save", summary="Сохранить текущий конфиг в config.json")
async def save_current_config(request: Request):
    cfg: Config = request.app.state.cfg
    path = save_config(cfg, getattr(cfg, "config_path", None))
    return {"ok": True, "path": str(path)}


@protected.post("/api/config/reload", summary="Перечитать config.json с диска")
async def reload_config(request: Request, svc: TTSService = Depends(get_service)):
    from ..config import load_config

    old_cfg: Config = request.app.state.cfg
    cfg = load_config(getattr(old_cfg, "config_path", None))   # перечитываем тот же файл
    request.app.state.cfg = cfg
    svc.cfg = cfg
    svc.config_path = getattr(cfg, "config_path", None)
    setup_logging(cfg.log, force=True)
    log.info("конфиг перечитан с диска")
    return {"ok": True, "config": _masked(cfg.to_dict())}


@protected.post("/api/backend", summary="Переключить движок синтеза")
async def set_backend(body: BackendRequest, request: Request, svc: TTSService = Depends(get_service)):
    cfg: Config = request.app.state.cfg
    cfg.device.backend = body.backend
    save_config(cfg, getattr(cfg, "config_path", None))
    svc.cfg = cfg
    meta = await asyncio.to_thread(svc.reload_engine)
    log.log(GPU, "движок переключён на '%s'", body.backend)
    return {"ok": True, "backend": body.backend, "model": meta}


@protected.post("/api/system/env", summary="Применить переменные окружения ROCm")
async def set_env(body: EnvApplyRequest, svc: TTSService = Depends(get_service)):
    env = dict(body.env or {})
    if body.preset and body.preset in PRESETS:
        env = {**PRESETS[body.preset], **env}
    applied = apply_env(env, overwrite=body.overwrite)
    torch_loaded = "torch" in sys.modules
    note = (
        "переменные применены к текущему процессу"
        if not torch_loaded
        else "ВНИМАНИЕ: torch уже импортирован — HSA_*/GPU_* применяются только при старте процесса. "
             "Сохраните их в config.json (device.env) и перезапустите сервер."
    )
    log.log(GPU, "env применён: %s", ", ".join(f"{k}={v}" for k, v, _ in applied) or "—")
    return {"ok": True, "applied": [{"name": k, "value": v, "action": a} for k, v, a in applied],
            "current": current_relevant_env(), "note": note, "restart_recommended": torch_loaded}


@protected.get("/api/system/env", summary="Текущие значимые переменные окружения")
async def get_env():
    return {"env": current_relevant_env(), "presets": {k: v for k, v in PRESETS.items()}}


# --------------------------------------------------------------------------- #
# Журнал и результаты
# --------------------------------------------------------------------------- #


@protected.get("/api/logs", summary="Хвост журнала (ring buffer)")
async def get_logs(limit: int = Query(200, ge=1, le=2000), level: Optional[str] = None):
    return {"logs": bus().tail(limit, level=level), "counters": bus().counters}


@protected.get("/api/logs/file", summary="Скачать файл лога")
async def get_log_file(request: Request):
    cfg: Config = request.app.state.cfg
    p = cfg.log_path
    if p is None or not p.exists():
        raise HTTPException(status_code=404, detail="файл лога не ведётся")
    return FileResponse(str(p), media_type="text/plain", filename=p.name)


@protected.get("/api/logs/stream", summary="SSE-поток журнала")
async def stream_logs(level: Optional[str] = None):
    """Отдаёт записи журнала по мере появления (для вкладки «Журнал» в GUI)."""
    q = subscribe()
    lvl = level.upper() if level else None

    async def gen():
        import queue as _queue

        try:
            yield f"retry: 2000\nevent: hello\ndata: {json.dumps({'ts': time.time()})}\n\n"
            last_beat = time.time()
            while True:
                try:
                    rec = q.get_nowait()
                except _queue.Empty:
                    if time.time() - last_beat > 12:
                        last_beat = time.time()
                        yield ": keep-alive\n\n"
                    await asyncio.sleep(0.15)
                    continue
                if rec is None:
                    break
                if lvl and rec.level != lvl:
                    continue
                last_beat = time.time()
                yield f"data: {json.dumps(rec.to_dict(), ensure_ascii=False)}\n\n"
        finally:
            bus().unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@protected.get("/api/outputs", summary="Список сохранённых результатов синтеза")
async def list_outputs(request: Request, limit: int = Query(100, ge=1, le=1000),
                       svc: TTSService = Depends(get_service)):
    d = request.app.state.cfg.outputs_path
    files = sorted(d.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit] if d.exists() else []
    return {
        "dir": str(d),
        "files": [
            {"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime,
             "url": f"/api/outputs/{f.name}"}
            for f in files
        ],
    }


@protected.get("/api/outputs/{name}", summary="Отдать файл результата")
async def get_output(name: str, request: Request):
    d = request.app.state.cfg.outputs_path
    p = (d / name).resolve()
    if not str(p).startswith(str(d.resolve())) or not p.exists():
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(str(p), filename=name)


def _masked(d: Dict[str, Any]) -> Dict[str, Any]:
    if d.get("server", {}).get("api_key"):
        d = json.loads(json.dumps(d))
        d["server"]["api_key"] = "***"
    return d
