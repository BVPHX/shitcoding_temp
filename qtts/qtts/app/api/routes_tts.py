"""Роуты синтеза: REST (обычный и OpenAI-совместимый), стриминг, задания, WebSocket."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from .. import audio as audio_mod
from ..engine.base import EngineError, SynthesisRequest
from ..logging_setup import REQ, get_logger
from ..service import TTSService
from .deps import client_ip, get_service, require_auth, ws_require_auth
from .schemas import TTSRequest

log = get_logger("api.tts")
router = APIRouter(dependencies=[Depends(require_auth)])

STREAM_CT = {
    "wav": "audio/wav",
    "pcm": "audio/L16;rate=24000;channels=1",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def build_request(svc: TTSService, body: TTSRequest, *, force_stream: Optional[bool] = None) -> SynthesisRequest:
    text = body.effective_text()
    if not text:
        raise HTTPException(status_code=422, detail="пустой текст (text/input)")
    fmt = body.effective_format()
    if fmt not in ("wav", "pcm", "mp3", "ogg", "flac", "opus", "m4a", "raw"):
        raise HTTPException(status_code=422, detail=f"неподдерживаемый формат: {fmt}")
    if fmt in ("mp3", "ogg", "opus", "flac", "m4a") and not audio_mod.ffmpeg_available():
        raise HTTPException(
            status_code=501,
            detail=f"формат '{fmt}' требует ffmpeg (не найден). Используйте wav/pcm или установите ffmpeg",
        )
    ref = body.ref_audio[0] if isinstance(body.ref_audio, list) and body.ref_audio else body.ref_audio
    try:
        voice = svc.resolve_voice(
            body.voice if body.voice is not None else body.profile,
            speaker=body.speaker,
            instruct=body.effective_instruct(),
            ref_audio=ref,
            ref_text=body.ref_text,
            x_vector_only=body.x_vector_only_mode,
            task=body.task_type,
        )
    except (EngineError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if voice.kind == "clone" and voice.profile_id is None and voice.ref_path is None:
        prof = svc.profiles.find_by_name(str(body.voice or ""))
        if prof is not None:
            voice = svc._voice_from_profile(prof)  # noqa: SLF001
    return SynthesisRequest(
        text=text,
        model_key=body.model,
        language=svc.normalize_language(body.language),
        voice=voice,
        gen_overrides=body.gen_overrides(),
        format=fmt,
        sample_rate=body.sample_rate,
        speed=body.speed,
        streaming=force_stream if force_stream is not None else body.stream,
        seed=body.seed,
    )


def meta_headers(meta: Dict[str, Any], job_id: str) -> Dict[str, str]:
    h = {
        "X-Job-Id": job_id,
        "X-Audio-Duration": str(meta.get("audio_sec", "")),
        "X-Sample-Rate": str(meta.get("sample_rate", "")),
        "X-TTS-RTF": str(meta.get("rtf", "")),
        "X-TTS-TTFA-ms": str(meta.get("ttfa_ms", "")),
        "X-TTS-Elapsed-ms": str(meta.get("elapsed_ms", "")),
        "X-TTS-Chunks": str(meta.get("chunks", "")),
        "Cache-Control": "no-store",
    }
    return {k: v for k, v in h.items() if v and v != "None"}


def engine_error_to_http(exc: EngineError) -> HTTPException:
    code_map = {
        "queue_full": 429,
        "queue_timeout": 503,
        "cancelled": 499,
        "gpu_unavailable": 503,
        "no_torch": 503,
        "no_qwen_tts": 503,
        "model_load_failed": 503,
        "vllm_unreachable": 502,
        "vllm_error": 502,
        "empty_text": 422,
        "no_reference": 422,
        "profile_no_audio": 422,
        "no_speaker": 422,
        "no_instruct": 422,
        "bad_model_kind": 409,
    }
    return HTTPException(
        status_code=code_map.get(exc.code, 500),
        detail={"error": str(exc), "code": exc.code, **(exc.details or {})},
    )


# --------------------------------------------------------------------------- #
# Основной эндпоинт: «всё в одном» (стриминг или цельный файл)
# --------------------------------------------------------------------------- #


@router.post("/api/tts", summary="Синтез речи (JSON-запрос, аудио в ответе)")
async def tts(body: TTSRequest, request: Request, svc: TTSService = Depends(get_service)):
    """Отдаёт готовое аудио. При `stream=true` — чанками (Transfer-Encoding: chunked).

    В заголовках ответа: X-Job-Id, X-Audio-Duration, X-TTS-RTF, X-TTS-TTFA-ms.
    """
    ip = client_ip(request)
    want_stream = body.stream if body.stream is not None else svc.cfg.generation.streaming
    req = build_request(svc, body, force_stream=bool(want_stream))
    job = svc.jobs.create(req, client=ip, endpoint="/api/tts")
    log.log(REQ, "запрос /api/tts job=%s client=%s chars=%d voice=%s stream=%s fmt=%s",
            job.id, ip, len(req.text), req.voice.describe(), req.streaming, req.format)

    if req.streaming:
        async def gen():
            sent = False
            try:
                async for item in svc.stream_synthesis(job, req):
                    if item["last"]:
                        # цельный файл в поток не дублируем — но если стриминг
                        # вырожден в одиночный файл (mp3/ogg/…), отдаём его целиком
                        if not sent:
                            yield item["data"]
                        continue
                    sent = True
                    yield item["data"]
            except EngineError as exc:
                log.error("стриминг job=%s прерван: %s", job.id, exc)
                yield json.dumps({"error": str(exc), "code": exc.code}).encode()

        return StreamingResponse(
            gen(),
            media_type=STREAM_CT.get(req.format, "audio/wav"),
            headers={**meta_headers({"sample_rate": svc.output_sample_rate(req)}, job.id)},
        )

    try:
        res = await svc.synthesize_full(req, client=ip, endpoint="/api/tts", job=job)
    except EngineError as exc:
        raise engine_error_to_http(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("ошибка /api/tts")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return Response(
        content=res["audio"],
        media_type=res["mime"],
        headers=meta_headers(res["meta"], job.id),
    )


@router.post("/api/tts/audio", summary="Только аудио (для стриминга в плеер)")
async def tts_audio(body: TTSRequest, request: Request, svc: TTSService = Depends(get_service)):
    ip = client_ip(request)
    body.stream = body.stream if body.stream is not None else True
    req = build_request(svc, body)
    job = svc.jobs.create(req, client=ip, endpoint="/api/tts/audio")
    fmt = req.format if req.format in ("pcm", "raw") else "wav"
    req.format = fmt
    req.streaming = True

    async def gen():
        try:
            async for item in svc.stream_synthesis(job, req):
                if item["last"]:
                    continue  # цельный файл не дублируем в поток
                yield item["data"]
        except EngineError as exc:
            log.error("стрим job=%s: %s", job.id, exc)

    return StreamingResponse(gen(), media_type=STREAM_CT.get(fmt, "audio/wav"),
                             headers=meta_headers({}, job.id))


@router.post("/api/tts/json", summary="Синтез с аудио в base64 + метаданные")
async def tts_json(body: TTSRequest, request: Request, svc: TTSService = Depends(get_service)):
    ip = client_ip(request)
    body.stream = False
    req = build_request(svc, body)
    job = svc.jobs.create(req, client=ip, endpoint="/api/tts/json")
    try:
        res = await svc.synthesize_full(req, client=ip, endpoint="/api/tts/json", job=job)
    except EngineError as exc:
        raise engine_error_to_http(exc) from exc
    return JSONResponse(
        {
            "job_id": job.id,
            "audio": base64.b64encode(res["audio"]).decode(),
            "format": res["meta"].get("format"),
            "mime": res["mime"],
            "sample_rate": res["meta"].get("sample_rate"),
            "audio_sec": res["meta"].get("audio_sec"),
            "metrics": {
                "elapsed_ms": res["meta"].get("elapsed_ms"),
                "ttfa_ms": res["meta"].get("ttfa_ms"),
                "rtf": res["meta"].get("rtf"),
                "chunks": res["meta"].get("chunks"),
            },
            "output_file": res["meta"].get("output_file"),
        }
    )


# --------------------------------------------------------------------------- #
# OpenAI-совместимый эндпоинт
# --------------------------------------------------------------------------- #


@router.post("/v1/audio/speech", summary="OpenAI-совместимый TTS-эндпоинт")
async def openai_speech(body: TTSRequest, request: Request, svc: TTSService = Depends(get_service)):
    """Совместим с OpenAI `POST /v1/audio/speech` (input, voice, response_format, speed, stream).

    Дополнительно принимает: language, instruct/instructions, ref_audio, ref_text, model, task_type.
    """
    ip = client_ip(request)
    if not body.text and body.input:
        body.text = body.input
    want_stream = bool(body.stream)
    req = build_request(svc, body, force_stream=want_stream)
    job = svc.jobs.create(req, client=ip, endpoint="/v1/audio/speech")
    log.log(REQ, "запрос /v1/audio/speech job=%s client=%s voice=%s stream=%s", job.id, ip,
            req.voice.describe(), want_stream)
    if want_stream:
        fmt = "pcm" if req.format in ("pcm", "raw") else req.format

        async def gen():
            sent = False
            try:
                async for item in svc.stream_synthesis(job, req):
                    if item["last"]:
                        if not sent:
                            yield item["data"]  # стриминг вырожден в одиночный файл (mp3/…)
                        continue
                    sent = True
                    yield item["data"]
            except EngineError as exc:
                log.error("стрим /v1/audio/speech job=%s: %s", job.id, exc)

        return StreamingResponse(gen(), media_type=audio_mod.MIME.get(fmt, "audio/wav"),
                                 headers=meta_headers({}, job.id))
    try:
        res = await svc.synthesize_full(req, client=ip, endpoint="/v1/audio/speech", job=job)
    except EngineError as exc:
        raise engine_error_to_http(exc) from exc
    return Response(content=res["audio"], media_type=res["mime"], headers=meta_headers(res["meta"], job.id))


# --------------------------------------------------------------------------- #
# Задания (async-режим)
# --------------------------------------------------------------------------- #


@router.post("/api/tts/async", status_code=202, summary="Поставить задачу в очередь и вернуть job_id")
async def tts_async(body: TTSRequest, request: Request, background: BackgroundTasks,
                    svc: TTSService = Depends(get_service)):
    ip = client_ip(request)
    body.stream = False
    req = build_request(svc, body)
    job = svc.jobs.create(req, client=ip, endpoint="/api/tts/async")

    async def run() -> None:
        try:
            async for _ in svc.stream_synthesis(job, req):
                pass
        except EngineError as exc:
            log.warning("фоновая задача %s завершилась ошибкой: %s", job.id, exc)
        except Exception:  # noqa: BLE001
            log.exception("фоновая задача %s упала", job.id)

    background.add_task(run)
    return JSONResponse({"job_id": job.id, "state": job.state, "status_url": f"/api/jobs/{job.id}"}, status_code=202)


@router.get("/api/jobs", summary="Список задач синтеза")
async def list_jobs(limit: int = Query(50, ge=1, le=500), state: Optional[str] = None,
                    svc: TTSService = Depends(get_service)):
    return {"jobs": svc.jobs.list(limit=limit, state=state), "stats": svc.jobs.stats()}


@router.get("/api/jobs/active", summary="Активные задачи (для индикаторов GUI)")
async def active_jobs(svc: TTSService = Depends(get_service)):
    jobs = [j for j in svc.jobs.list(limit=200) if j["state"] in ("queued", "running", "streaming")]
    return {"active": jobs[:20], "count": len(jobs)}


@router.get("/api/jobs/{job_id}", summary="Статус и метрики задачи")
async def get_job(job_id: str, events: bool = False, svc: TTSService = Depends(get_service)):
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"задача {job_id} не найдена")
    return job.to_dict(with_events=events, with_text=True)


@router.get("/api/jobs/{job_id}/audio", summary="Скачать аудио завершённой задачи")
async def job_audio(job_id: str, svc: TTSService = Depends(get_service)):
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"задача {job_id} не найдена")
    if job.state != "done":
        raise HTTPException(status_code=409, detail=f"задача в состоянии '{job.state}', аудио ещё нет")
    if not job.output_file or not Path(job.output_file).exists():
        raise HTTPException(
            status_code=404,
            detail="файл не сохранён (audio.save_outputs=false или уже удалён при ротации)",
        )
    data = Path(job.output_file).read_bytes()
    return Response(content=data, media_type=audio_mod.MIME.get(job.format, "audio/wav"),
                    headers={"Content-Disposition": f'inline; filename="{Path(job.output_file).name}"'})


@router.post("/api/jobs/{job_id}/cancel", summary="Отменить задачу")
async def cancel_job(job_id: str, svc: TTSService = Depends(get_service)):
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"задача {job_id} не найдена")
    job.cancel_event.set()
    if job.state in ("queued",):
        svc.jobs.update(job, state="cancelled", finished_at=time.time())
    log.warning("задача %s отменена клиентом", job_id)
    return {"job_id": job_id, "state": job.state, "cancel_requested": True}


@router.delete("/api/jobs", summary="Очистить историю задач")
async def clear_jobs(svc: TTSService = Depends(get_service)):
    return {"cleared": svc.jobs.clear()}


# --------------------------------------------------------------------------- #
# WebSocket: стриминг аудио + событий
# --------------------------------------------------------------------------- #


@router.websocket("/ws/tts")
async def ws_tts(ws: WebSocket, _auth: None = Depends(ws_require_auth)):
    """WebSocket-стриминг синтеза.

    Клиент → сервер (JSON):
        {"text": "...", "voice": "mama", "language": "Russian", "format": "pcm", "stream": true}
        {"cancel": "<job_id>"}
    Сервер → клиент:
        {"event":"job","job_id":"..."}                     — задача принята
        {"event":"progress", ...}                           — этапы/чанки (type: header|audio)
        <binary frame>                                      — байты единого аудиопотока:
                                                              для format=pcm это сырые PCM16-чанки,
                                                              для format=wav первый кадр — RIFF-заголовок
                                                              (размеры 0xFFFFFFFF), далее PCM16-чанки
        {"event":"done", "audio_sec":..., "rtf":...}        — готово (без дубля аудио)
        {"event":"error","error":"...","code":"..."}        — ошибка
    """
    await ws.accept()
    svc: TTSService = ws.app.state.service
    ip = ws.client.host if ws.client else "?"
    log.log(REQ, "WS /ws/tts подключён client=%s", ip)
    current: Optional[str] = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                await ws.send_json({"event": "error", "error": "некорректный JSON", "code": "bad_json"})
                continue
            if msg.get("cancel"):
                job = svc.jobs.get(str(msg["cancel"]))
                if job:
                    job.cancel_event.set()
                    await ws.send_json({"event": "cancelled", "job_id": job.id})
                continue
            try:
                body = TTSRequest.model_validate(msg)
            except ValidationError as exc:
                first = exc.errors()[0] if exc.errors() else {}
                field = ".".join(str(x) for x in first.get("loc", ()))
                await ws.send_json({
                    "event": "error",
                    "error": f"некорректный запрос: {first.get('msg', 'ошибка валидации')}"
                             + (f" (поле {field})" if field else ""),
                    "code": "bad_request",
                })
                continue
            try:
                req = build_request(svc, body, force_stream=True)
            except HTTPException as exc:
                await ws.send_json({"event": "error", "error": exc.detail, "code": "bad_request"})
                continue
            job = svc.jobs.create(req, client=f"ws:{ip}", endpoint="/ws/tts")
            current = job.id
            await ws.send_json({"event": "job", "job_id": job.id, "chunks": None})
            log.log(REQ, "WS-запрос job=%s chars=%d voice=%s", job.id, len(req.text), req.voice.describe())
            try:
                async for item in svc.stream_synthesis(job, req):
                    if item["meta"]["type"] == "final":
                        await ws.send_json({"event": "done", **item["meta"]})
                    else:
                        await ws.send_json({"event": "progress", **item["meta"]})
                        await ws.send_bytes(item["data"])
            except EngineError as exc:
                await ws.send_json({"event": "error", "error": str(exc), "code": exc.code})
            except WebSocketDisconnect:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("ошибка WS-синтеза")
                await ws.send_json({"event": "error", "error": f"{type(exc).__name__}: {exc}", "code": "internal"})
    except WebSocketDisconnect:
        log.log(REQ, "WS /ws/tts отключён client=%s", ip)
    except Exception:  # noqa: BLE001
        log.exception("сбой WS-соединения")
    finally:
        if current:
            job = svc.jobs.get(current)
            if job and job.state in ("queued", "running", "streaming"):
                job.cancel_event.set()
