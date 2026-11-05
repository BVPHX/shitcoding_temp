"""Роуты профилей голосов (образцы для клонирования) и списка доступных голосов."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from .. import audio as audio_mod
from ..engine.base import EngineError, SynthesisRequest
from ..logging_setup import get_logger
from ..profiles import ProfileError
from ..service import PRESET_SPEAKERS, TTSService
from .deps import client_ip, get_service, require_auth
from .routes_tts import engine_error_to_http
from .schemas import ProfileCreate, ProfileUpdate, TestRequest

log = get_logger("api.profiles")
router = APIRouter(dependencies=[Depends(require_auth)])

MAX_UPLOAD = 64 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Список голосов
# --------------------------------------------------------------------------- #


@router.get("/api/voices", summary="Все доступные голоса: пресеты + профили")
async def list_voices(svc: TTSService = Depends(get_service)):
    engine_speakers = []
    try:
        engine_speakers = svc.engine.speakers()
    except Exception as exc:  # noqa: BLE001
        log.debug("список пресетов недоступен (модель не загружена?): %s", exc)
    presets = sorted({*(engine_speakers or []), *(PRESET_SPEAKERS if not engine_speakers else [])})
    profiles = [svc.profiles.describe(p) for p in svc.profiles.list()]
    return {
        "presets": presets,
        "profiles": profiles,
        "default_profile": next((p["id"] for p in profiles if p["kind"] == "clone"), None),
        "count": len(presets) + len(profiles),
    }


@router.get("/api/voices/presets", summary="Встроенные тембры модели CustomVoice")
async def presets(svc: TTSService = Depends(get_service)):
    try:
        spk = svc.engine.speakers()
    except Exception:  # noqa: BLE001
        spk = []
    return {"speakers": spk or PRESET_SPEAKERS, "loaded": bool(spk)}


@router.get("/api/voices/languages", summary="Поддерживаемые языки")
async def languages(svc: TTSService = Depends(get_service)):
    try:
        langs = svc.engine.languages()
    except Exception:  # noqa: BLE001
        langs = []
    return {
        "languages": langs
        or ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian",
            "Portuguese", "Spanish", "Italian"],
    }


# --------------------------------------------------------------------------- #
# Профили
# --------------------------------------------------------------------------- #


@router.get("/api/profiles", summary="Список профилей голосов")
async def get_profiles(svc: TTSService = Depends(get_service)):
    return {"profiles": [svc.profiles.describe(p) for p in svc.profiles.list()]}


@router.get("/api/profiles/{pid}", summary="Профиль целиком")
async def get_profile(pid: str, svc: TTSService = Depends(get_service)):
    try:
        p = svc.profiles.get(pid)
    except ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return svc.profiles.describe(p)


def _tags(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [t.strip() for t in re.split(r"[,;]", v) if t.strip()]
    return []


@router.post(
    "/api/profiles",
    summary="Создать профиль (multipart с файлом ИЛИ JSON с audio=base64/URL/путь)",
    response_model=None,
)
async def create_profile(request: Request, svc: TTSService = Depends(get_service)):
    """Создать профиль голоса.

    Два формата тела на выбор (определяется по Content-Type):

    * **multipart/form-data** — как это делает веб-GUI: поля `name`, `kind`,
      `ref_text`, `instruct`, `language`, `speaker`, `x_vector_only`, `tags`,
      `notes`, `model_key` + файл `file` (wav/mp3/ogg/flac).
    * **application/json** — модель `ProfileCreate`, где `audio` это base64,
      data-URI, http(s)-URL или путь к файлу **на сервере**.

    Для профилей типа `preset` файл не нужен — достаточно `speaker`.
    """
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith("multipart/") or ctype.startswith("application/x-www-form-urlencoded"):
        data, audio_src, file_name = await _read_form(request)
    elif ctype.startswith("application/json"):
        data, audio_src, file_name = await _read_json(request)
    else:
        raise HTTPException(
            status_code=415,
            detail=f"непонятный Content-Type '{ctype or '(пусто)'}'. "
                   "Ожидается multipart/form-data (файл) или application/json",
        )

    name = str(data.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="поле 'name' обязательно")
    kind = str(data.get("kind") or "clone").strip().lower()
    if kind not in ("clone", "preset", "designed"):
        raise HTTPException(status_code=422, detail=f"неизвестный тип профиля: {kind}")
    if kind == "clone" and audio_src is None:
        raise HTTPException(
            status_code=422,
            detail="для профиля-клона нужен образец голоса: файл в поле 'file' (multipart) "
                   "или 'audio' = base64/URL/путь (JSON)",
        )

    try:
        p, info = svc.profiles.create(
            name=name,
            kind=kind,
            audio_src=audio_src,
            ref_text=_opt_str(data.get("ref_text")),
            instruct=_opt_str(data.get("instruct")),
            language=_opt_str(data.get("language")),
            speaker=_opt_str(data.get("speaker")),
            x_vector_only=_as_bool(data.get("x_vector_only")),
            tags=_tags(data.get("tags")),
            notes=_opt_str(data.get("notes")) or "",
            model_key=_opt_str(data.get("model_key")),
            profile_id=_opt_str(data.get("profile_id")),
        )
    except ProfileError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (audio_mod.AudioError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    svc.engine.invalidate_cache()
    log.info("профиль '%s' создан (%s) от %s", p.id, file_name or kind, client_ip(request))
    return {"profile": svc.profiles.describe(p), **info}


def _opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("1", "true", "yes", "on", "да")


async def _read_form(request: Request) -> Tuple[Dict[str, Any], Optional[bytes], str]:
    """Разобрать multipart/urlencoded-тело. Файл → bytes, остальное → строки."""
    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001 — нет python-multipart / битый multipart
        raise HTTPException(
            status_code=400,
            detail=f"не удалось разобрать форму: {exc}. "
                   "Проверьте, что установлен python-multipart (pip install -e .)",
        ) from exc
    data: Dict[str, Any] = {}
    audio: Optional[bytes] = None
    fname = ""
    for key in form:                     # SIM118: не нужен .keys()
        val = form[key]
        if hasattr(val, "read"):  # UploadFile
            blob = await val.read()
            if len(blob) > MAX_UPLOAD:
                raise HTTPException(status_code=413, detail=f"файл больше {MAX_UPLOAD // (1024 * 1024)} МБ")
            if blob:
                audio = blob
                fname = getattr(val, "filename", "") or "file"
        else:
            data[key] = val
    return data, audio, fname


async def _read_json(request: Request) -> Tuple[Dict[str, Any], Any, str]:
    """Разобрать JSON-тело в ProfileCreate (audio остаётся строкой-источником)."""
    try:
        raw = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"некорректный JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="JSON-тело должно быть объектом")
    try:
        payload = ProfileCreate.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    data = payload.model_dump()
    audio = data.pop("audio", None) or None
    return data, audio, "json"


@router.patch("/api/profiles/{pid}", summary="Обновить профиль (JSON)")
@router.put("/api/profiles/{pid}", include_in_schema=False)
async def update_profile(pid: str, body: ProfileUpdate, svc: TTSService = Depends(get_service)):
    try:
        changes = body.model_dump(exclude_none=True)
        p = svc.profiles.update(pid, **changes)
    except ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except audio_mod.AudioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    svc.engine.invalidate_cache(pid)
    return {"profile": svc.profiles.describe(p)}


@router.delete("/api/profiles/{pid}", summary="Удалить профиль")
async def delete_profile(pid: str, svc: TTSService = Depends(get_service)):
    try:
        ok = svc.profiles.delete(pid)
    except ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail=f"профиль '{pid}' не найден")
    svc.engine.invalidate_cache(pid)
    return {"deleted": pid}


@router.post("/api/profiles/{pid}/duplicate", summary="Скопировать профиль")
async def duplicate_profile(pid: str, name: str = Query(..., min_length=1), svc: TTSService = Depends(get_service)):
    try:
        p = svc.profiles.duplicate(pid, name)
    except ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"profile": svc.profiles.describe(p)}


@router.get("/api/profiles/{pid}/audio", summary="Эталонное аудио профиля")
async def profile_audio(pid: str, svc: TTSService = Depends(get_service)):
    try:
        p = svc.profiles.get(pid)
    except ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = svc.profiles.audio_path(p)
    if path is None:
        raise HTTPException(status_code=404, detail="у профиля нет эталонного аудио")
    return FileResponse(str(path), media_type="audio/wav", filename=f"{pid}_ref.wav")


@router.post("/api/profiles/analyze", summary="Разобрать аудио без сохранения (проверка эталона)")
async def analyze_audio(request: Request, file: Optional[UploadFile] = File(None),
                        svc: TTSService = Depends(get_service)):
    if file is None:
        raise HTTPException(status_code=422, detail="пришлите файл (multipart, поле file)")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="пустой файл")
    try:
        wav, sr = audio_mod.load_audio_bytes(data)
        wav, sr = audio_mod.prepare_reference(wav, sr, max_sec=svc.cfg.profiles.ref_max_sec)
    except audio_mod.AudioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    stats = audio_mod.audio_stats(wav, sr)
    warnings: List[str] = []
    if stats["duration_sec"] < svc.cfg.profiles.ref_min_sec:
        warnings.append(f"короткий эталон ({stats['duration_sec']:.1f} с), нужно ≥ {svc.cfg.profiles.ref_min_sec:.0f} с")
    if stats["duration_sec"] > svc.cfg.profiles.ref_target_sec:
        warnings.append(
            f"эталон длиннее рекомендуемых {svc.cfg.profiles.ref_target_sec:.0f} с — "
            "лишнее обрезано, качество клона от этого не растёт"
        )
    if stats.get("clipping"):
        warnings.append("клиппинг (перегруз) — лучше перезаписать тише")
    if stats.get("silent"):
        warnings.append("почти тишина")
    return {"filename": file.filename, "stats": stats, "warnings": warnings, "target_sr": sr}


@router.post("/api/profiles/{pid}/test", summary="Тестовый синтез голосом профиля")
async def test_profile(pid: str, body: Optional[TestRequest] = None, request: Request = None,
                       svc: TTSService = Depends(get_service)):
    try:
        p = svc.profiles.get(pid)
    except ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    text = (body.text if body else None) or "Раз, два, три — проверка голоса. Меня слышно?"
    req = SynthesisRequest(
        text=text,
        voice=svc._voice_from_profile(p),  # noqa: SLF001
        language=(body.language if body else None) or p.language or svc.cfg.generation.language,
        format="wav",
        streaming=False,
    )
    try:
        res = await svc.synthesize_full(req, client=client_ip(request) if request else "?",
                                        endpoint=f"/api/profiles/{pid}/test")
    except EngineError as exc:
        raise engine_error_to_http(exc) from exc
    return Response(content=res["audio"], media_type=res["mime"],
                    headers={"X-Job-Id": str(res["job"].id), "X-Audio-Duration": str(res["meta"].get("audio_sec"))})
