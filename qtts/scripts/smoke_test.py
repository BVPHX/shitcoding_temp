#!/usr/bin/env python3
"""Сквозная проверка работающего сервера Qwen TTS Studio.

Только стандартная библиотека Python (curl/jq не нужны). Проходит по всем
ключевым эндпоинтам и печатает PASS/FAIL по каждому:

  * здоровье, веб-GUI, OpenAPI-спецификация;
  * голоса и пресеты;
  * профили голосов: создание (multipart) → эталон → тестовый синтез → дубликат → удаление;
  * синтез: цельный WAV, стриминг PCM, /api/tts/json, OpenAI-совместимый /v1/audio/speech;
  * async-режим: 202 → опрос задачи → скачивание аудио;
  * система: диагностика ROCm, готовность, модели, конфиг (маскировка api_key), журнал, результаты;
  * WebSocket /ws/tts (если установлен пакет `websockets`, иначе пропускается).

Запуск:
    python3 scripts/smoke_test.py                          # http://127.0.0.1:8770
    python3 scripts/smoke_test.py http://127.0.0.1:8799    # например, dummy-сервер
    QTTSS_API_KEY=секрет python3 scripts/smoke_test.py http://host:8770

Код возврата: 0 — всё прошло, 1 — есть провалы, 2 — сервер недоступен.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# вывод
# --------------------------------------------------------------------------- #


class Out:
    def __init__(self, color: bool = True) -> None:
        self.color = color and sys.stdout.isatty()
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def _c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def section(self, title: str) -> None:
        print("\n" + self._c("2", f"── {title} " + "─" * max(0, 58 - len(title))))

    def ok(self, msg: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  {self._c('32', 'PASS')}  {msg}" + (self._c("2", f"  {detail}") if detail else ""))

    def bad(self, msg: str, detail: str = "") -> None:
        self.failed += 1
        print(f"  {self._c('31', 'FAIL')}  {msg}" + (self._c("2", f"  ({detail})") if detail else ""))

    def skip(self, msg: str) -> None:
        self.skipped += 1
        print(f"  {self._c('33', 'SKIP')}  {msg}")

    def info(self, msg: str) -> None:
        print("        " + self._c("2", msg))

    def summary(self) -> int:
        print("\n" + "─" * 62)
        line = f"  итог: {self.passed} passed, {self.failed} failed"
        if self.skipped:
            line += f", {self.skipped} skipped"
        print(self._c("31" if self.failed else "32", line))
        print("─" * 62)
        return 1 if self.failed else 0


# --------------------------------------------------------------------------- #
# HTTP-клиент
# --------------------------------------------------------------------------- #


class Api:
    def __init__(self, base: str, api_key: str = "", timeout: float = 180.0) -> None:
        self.base = base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        multipart: Optional[Tuple[Dict[str, str], List[Tuple[str, str, bytes, str]]]] = None,
        raw_body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, bytes, Dict[str, str]]:
        url = self.base + path
        hdrs: Dict[str, str] = dict(headers or {})
        if self.api_key:
            hdrs.setdefault("X-API-Key", self.api_key)

        data: Optional[bytes] = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif multipart is not None:
            data, ct = encode_multipart(*multipart)
            hdrs["Content-Type"] = ct
        elif raw_body is not None:
            data = raw_body

        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                return resp.status, resp.read(), dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read() or b"", dict(exc.headers.items()) if exc.headers else {}

    def get(self, path: str, **kw: Any) -> Tuple[int, bytes, Dict[str, str]]:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> Tuple[int, bytes, Dict[str, str]]:
        return self.request("POST", path, **kw)


def encode_multipart(fields: Dict[str, str],
                     files: List[Tuple[str, str, bytes, str]]) -> Tuple[bytes, str]:
    """Поля + файлы в multipart/form-data (как это делает браузер)."""
    boundary = f"----qtts{uuid.uuid4().hex}"
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buf.write(f"{value}\r\n".encode())
    for name, filename, blob, ctype in files:
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        buf.write(blob)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def tone_wav(seconds: float = 3.0, freq: float = 220.0, sr: int = 24000) -> bytes:
    """Синтетический моно-WAV (тон) — заменяет настоящий образец голоса."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(int(sr * seconds)):
            frames += struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr)))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def is_wav(blob: bytes) -> bool:
    return blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"


def wav_seconds(blob: bytes) -> float:
    try:
        with wave.open(io.BytesIO(blob), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------- #
# проверки
# --------------------------------------------------------------------------- #


def check(out: Out, api: Api, name: str, method: str, path: str, want: int, **kw: Any) -> Tuple[int, bytes, Dict[str, str]]:
    code, body, hdrs = api.request(method, path, **kw)
    if code == want:
        out.ok(f"{method} {path} → {code}")
    else:
        out.bad(f"{method} {path} → {code} (ожидался {want})", body[:200].decode("utf-8", "replace").replace("\n", " "))
    return code, body, hdrs


def basics(out: Out, api: Api) -> None:
    out.section("Здоровье, GUI, спецификация")
    if api.api_key:
        anon = Api(api.base, "", api.timeout)
        code, _b, _h = anon.get("/api/config")
        (out.ok if code == 401 else out.bad)(f"GET /api/config без ключа → {code} (ожидался 401)")

    code, body, _h = check(out, api, "health", "GET", "/api/health", 200)
    if code == 200:
        d = json.loads(body)
        out.info(f"движок: {d.get('engine')}  модель загружена: {d.get('model_loaded')}  uptime: {d.get('uptime_sec')} с")

    code, body, _h = check(out, api, "index", "GET", "/", 200)
    if code == 200:
        html = body.decode("utf-8", "replace")
        (out.ok if "__QTTS_BOOT__" in html else out.bad)("в index.html есть __QTTS_BOOT__ (токен для GUI)")
    check(out, api, "app.js", "GET", "/ui/app.js", 200)
    check(out, api, "app.css", "GET", "/ui/app.css", 200)
    code, body, _h = check(out, api, "openapi", "GET", "/api/openapi.json", 200)
    if code == 200:
        paths = json.loads(body).get("paths", {})
        # WS-роуты в OpenAPI не попадают (спецификация описывает только HTTP) — проверяем HTTP-часть
        missing = [p for p in ("/api/tts", "/v1/audio/speech", "/api/profiles", "/api/tts/async") if p not in paths]
        (out.ok if not missing else out.bad)("в спецификации есть все основные эндпоинты",
                                             f"нет: {missing}" if missing else "")


def voices(out: Out, api: Api) -> None:
    out.section("Голоса")
    code, body, _h = check(out, api, "voices", "GET", "/api/voices", 200)
    if code == 200:
        d = json.loads(body)
        out.info(f"пресетов: {len(d.get('presets', []))}, профилей: {len(d.get('profiles', []))}")
        (out.ok if d.get("presets") else out.bad)("список пресетов не пуст")
    check(out, api, "presets", "GET", "/api/voices/presets", 200)
    code, body, _h = check(out, api, "languages", "GET", "/api/voices/languages", 200)
    if code == 200:
        langs = json.loads(body).get("languages", [])
        (out.ok if "Russian" in langs else out.bad)("русский язык поддерживается", str(langs[:6]))


def profiles(out: Out, api: Api) -> None:
    out.section("Профили голосов (клонирование)")
    ref = tone_wav(3.0)
    fields = {"name": "Smoke Test", "kind": "clone", "ref_text": "Проверочный эталон.", "language": "Russian"}
    code, body, _h = api.request("POST", "/api/profiles", multipart=(fields, [("file", "ref.wav", ref, "audio/wav")]))
    pid = ""
    if code == 200:
        out.ok("POST /api/profiles (multipart + файл) → 200")
        prof = json.loads(body).get("profile", {})
        pid = prof.get("id", "")
        out.info(f"id={pid}  длительность эталона={(prof.get('audio') or {}).get('duration_sec')} с")
        (out.ok if pid == "smoke-test" else out.bad)("id профиля — транслитерация имени (Smoke Test → smoke-test)", pid)
    else:
        out.bad(f"POST /api/profiles → {code} (ожидался 200)", body[:200].decode("utf-8", "replace"))

    if pid:
        check(out, api, "profile", "GET", f"/api/profiles/{pid}", 200)
        code, body, _h = check(out, api, "profile audio", "GET", f"/api/profiles/{pid}/audio", 200)
        if code == 200:
            (out.ok if is_wav(body) else out.bad)("эталон хранится как WAV", f"{len(body)} байт")
            sr = 0
            try:
                with wave.open(io.BytesIO(body), "rb") as w:
                    sr = w.getframerate()
            except Exception:  # noqa: BLE001
                pass
            (out.ok if sr == 24000 else out.bad)("эталон приведён к 24 кГц (нативная частота модели)", f"sr={sr}")

        # дубль имени → 409
        code, _b, _h = api.request("POST", "/api/profiles",
                                   json_body={"name": "Smoke Test", "kind": "clone",
                                              "audio": "data:audio/wav;base64," + base64.b64encode(ref).decode()})
        (out.ok if code == 409 else out.bad)(f"дубликат имени профиля → {code} (ожидался 409)")

        # JSON-вариант создания (base64)
        code, body, _h = api.request("POST", "/api/profiles", json_body={
            "name": "Smoke JSON", "kind": "clone",
            "audio": "data:audio/wav;base64," + base64.b64encode(ref).decode(),
            "ref_text": "Эталон из base64.", "tags": ["smoke"]})
        if code == 200:
            out.ok("POST /api/profiles (JSON + data-URI base64) → 200")
            pid2 = json.loads(body)["profile"]["id"]
            api.request("DELETE", f"/api/profiles/{pid2}")
        else:
            out.bad(f"POST /api/profiles (JSON) → {code}", body[:160].decode("utf-8", "replace"))

        # тестовый синтез голосом профиля
        code, body, _h = api.request("POST", f"/api/profiles/{pid}/test", json_body={"text": "Проверка голоса профиля."})
        if code == 200 and is_wav(body):
            out.ok(f"POST /api/profiles/{pid}/test → 200, {wav_seconds(body):.2f} с аудио")
        else:
            out.bad(f"POST /api/profiles/{pid}/test → {code}", body[:160].decode("utf-8", "replace"))

        # синтез через /api/tts голосом профиля
        code, body, hdrs = api.request("POST", "/api/tts",
                                       json_body={"text": "Голос профиля в обычном запросе.", "voice": pid,
                                                  "language": "ru", "stream": False})
        if code == 200 and is_wav(body):
            out.ok(f"POST /api/tts voice={pid} → 200 ({wav_seconds(body):.2f} с)")
        else:
            out.bad(f"POST /api/tts voice={pid} → {code}", body[:160].decode("utf-8", "replace"))

    # ошибки валидации
    code, _b, _h = api.request("POST", "/api/profiles", raw_body=b"name=x",
                               headers={"Content-Type": "text/plain"})
    (out.ok if code == 415 else out.bad)(f"неподдерживаемый Content-Type → {code} (ожидался 415)")
    code, _b, _h = api.request("POST", "/api/profiles", json_body={"kind": "clone", "audio": None})
    (out.ok if code == 422 else out.bad)(f"профиль без имени → {code} (ожидался 422)")
    check(out, api, "analyze", "POST", "/api/profiles/analyze", 200,
          multipart=({}, [("file", "ref.wav", ref, "audio/wav")]))
    check(out, api, "404 profile", "GET", "/api/profiles/no-such-profile", 404)

    if pid:
        check(out, api, "delete", "DELETE", f"/api/profiles/{pid}", 200)


def synthesis(out: Out, api: Api) -> None:
    out.section("Синтез")
    code, body, hdrs = api.request("POST", "/api/tts", json_body={
        "text": "Проверка связи. Сервер работает.", "stream": False})
    if code == 200 and is_wav(body):
        out.ok(f"POST /api/tts (цельный WAV) → 200, {wav_seconds(body):.2f} с, {len(body)} байт")
        for h in ("X-Job-Id", "X-Audio-Duration", "X-TTS-RTF", "X-TTS-Elapsed-ms", "X-Request-ID"):
            key = next((k for k in hdrs if k.lower() == h.lower()), None)
            (out.ok if key else out.bad)(f"заголовок {h}: {hdrs.get(key) if key else 'нет'}")
    else:
        out.bad(f"POST /api/tts → {code}", body[:160].decode("utf-8", "replace"))

    code, body, hdrs = api.request("POST", "/api/tts", json_body={
        "text": "Раз. Два. Три. Четыре.", "format": "pcm", "stream": True})
    ct = next((v for k, v in hdrs.items() if k.lower() == "content-type"), "")
    if code == 200 and "audio/L16" in ct and not is_wav(body):
        out.ok(f"POST /api/tts (стриминг PCM) → 200, {len(body)} байт, {ct}")
    else:
        out.bad(f"стриминг PCM → {code}, content-type={ct}")

    code, body, _h = api.request("POST", "/api/tts/json", json_body={"text": "JSON-вариант ответа."})
    if code == 200:
        d = json.loads(body)
        audio = base64.b64decode(d.get("audio", ""))
        if is_wav(audio) and d.get("audio_sec"):
            out.ok(f"POST /api/tts/json → 200, audio_sec={d['audio_sec']}, метрики={d.get('metrics')}")
        else:
            out.bad("/api/tts/json вернул нечитаемое аудио")
    else:
        out.bad(f"POST /api/tts/json → {code}", body[:160].decode("utf-8", "replace"))

    code, body, _h = api.request("POST", "/v1/audio/speech", json_body={
        "input": "OpenAI-совместимый вызов.", "voice": "vivian", "response_format": "wav"})
    (out.ok if code == 200 and is_wav(body) else out.bad)(
        f"POST /v1/audio/speech (только input, OpenAI-стиль) → {code}",
        "" if code == 200 else body[:160].decode("utf-8", "replace"))

    code, _b, _h = api.request("POST", "/api/tts", json_body={"text": "  "})
    (out.ok if code == 422 else out.bad)(f"пустой текст → {code} (ожидался 422)")
    code, _b, _h = api.request("POST", "/api/tts", json_body={"text": "формат", "format": "midi"})
    (out.ok if code == 422 else out.bad)(f"неизвестный формат → {code} (ожидался 422)")


def async_jobs(out: Out, api: Api) -> None:
    out.section("Async-режим и задачи")
    code, body, _h = api.request("POST", "/api/tts/async", json_body={
        "text": "Задача в очереди. Ждём завершения."})
    if code != 202:
        out.bad(f"POST /api/tts/async → {code} (ожидался 202)", body[:160].decode("utf-8", "replace"))
        return
    d = json.loads(body)
    job_id, status_url = d.get("job_id"), d.get("status_url")
    out.ok(f"POST /api/tts/async → 202, job_id={job_id}, status_url={status_url}")

    state, waited = "", 0.0
    t0 = time.time()
    while time.time() - t0 < 180:
        c2, b2, _h2 = api.get(status_url or f"/api/jobs/{job_id}")
        if c2 == 200:
            j = json.loads(b2)
            state = j.get("state", "")
            if state in ("done", "error", "cancelled"):
                waited = time.time() - t0
                break
        time.sleep(0.5)
    if state == "done":
        out.ok(f"задача завершилась за {waited:.1f} с (state=done)")
        c3, b3, _h3 = api.get(f"/api/jobs/{job_id}/audio")
        (out.ok if c3 == 200 and is_wav(b3) else out.bad)(
            f"GET /api/jobs/{job_id}/audio → {c3}, {wav_seconds(b3):.2f} с")
    else:
        out.bad(f"задача не завершилась (state={state or '?'})")

    code, body, _h = api.get("/api/jobs")
    if code == 200:
        d = json.loads(body)
        out.ok(f"GET /api/jobs → 200, задач: {d['stats']}")
    else:
        out.bad(f"GET /api/jobs → {code}")
    check(out, api, "active", "GET", "/api/jobs/active", 200)
    check(out, api, "404 job", "GET", "/api/jobs/no-such-job", 404)


def system(out: Out, api: Api) -> None:
    out.section("Система, диагностика ROCm, настройки")
    code, body, _h = check(out, api, "info", "GET", "/api/system/info", 200)
    if code == 200:
        d = json.loads(body)
        v = d.get("versions", {})
        out.info(f"приложение {v.get('app')}, torch {v.get('torch')}, qwen_tts {v.get('qwen_tts')}, "
                 f"transformers {v.get('transformers')}")
        out.info(f"конфиг: {d.get('config_path')}")
        out.info(f"данные: {d.get('paths', {}).get('data')}")
    check(out, api, "status", "GET", "/api/system/status", 200)

    code, body, _h = check(out, api, "diagnostics", "GET", "/api/system/diagnostics?bench=false", 200)
    if code == 200:
        d = json.loads(body)
        out.info(f"статус диагностики: {d.get('health')}")
        for c in d.get("checks", [])[:8]:
            mark = {"ok": "+", "warn": "!", "error": "X"}.get(c.get("status"), "?")
            out.info(f"[{mark}] {c.get('name')}: {c.get('message', '')[:90]}")
    check(out, api, "diagnostics/text", "GET", "/api/system/diagnostics/text", 200)

    code, body, _h = check(out, api, "readiness", "GET", "/api/system/readiness", 200)
    if code == 200:
        d = json.loads(body)
        out.info(f"ready={d.get('ready')} gpu_visible={d.get('gpu_visible')} rocm={d.get('rocm_version')} "
                 f"gpu={d.get('gpu_name')} gfx={d.get('gfx')}")
        if not d.get("ready"):
            out.info("проблемы: " + "; ".join(p.get("message", "")[:70] for p in d.get("problems", [])) or "—")
    check(out, api, "gpu", "GET", "/api/system/gpu", 200)
    check(out, api, "models", "GET", "/api/models", 200)

    code, body, _h = check(out, api, "config", "GET", "/api/config", 200)
    if code == 200:
        text = body.decode("utf-8", "replace")
        cfg = json.loads(text)
        if api.api_key:
            (out.ok if cfg.get("server", {}).get("api_key") == "***" else out.bad)(
                "GET /api/config маскирует api_key")
        out.info(f"backend={cfg.get('device', {}).get('backend')} dtype={cfg.get('device', {}).get('dtype')} "
                 f"attn={cfg.get('device', {}).get('attn')} streaming={cfg.get('generation', {}).get('streaming')}")

    code, body, _h = api.request("PATCH", "/api/config", json_body={"log": {"level": "INFO"}})
    (out.ok if code == 200 else out.bad)(f"PATCH /api/config (уровень журнала) → {code}")

    code, body, _h = api.get("/api/logs?limit=20")
    if code == 200:
        d = json.loads(body)
        out.ok(f"GET /api/logs → 200, записей: {len(d.get('logs', []))}, счётчики: {d.get('counters')}")
    else:
        out.bad(f"GET /api/logs → {code}")
    check(out, api, "outputs", "GET", "/api/outputs", 200)


def websocket(out: Out, api: Api) -> None:
    out.section("WebSocket /ws/tts")
    try:
        import websockets  # noqa: F401
    except ImportError:
        out.skip("пакет `websockets` не установлен — pip install websockets (проверяется в pytest)")
        return

    url = api.base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/tts"
    if api.api_key:
        url += f"?api_key={api.api_key}"

    async def run() -> Dict[str, Any]:
        import websockets

        res: Dict[str, Any] = {"audio": 0, "events": [], "ttfa": None}
        async with websockets.connect(url, open_timeout=15) as ws:
            await ws.send(json.dumps({"text": "Раз. Два. Три.", "format": "pcm", "stream": True}))
            t0 = time.time()
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                if isinstance(msg, bytes):
                    if res["ttfa"] is None:
                        res["ttfa"] = (time.time() - t0) * 1000
                    res["audio"] += len(msg)
                    continue
                d = json.loads(msg)
                res["events"].append(d.get("event"))
                if d.get("event") in ("done", "error"):
                    res["final"] = d
                    break
        return res

    try:
        res = asyncio.run(asyncio.wait_for(run(), timeout=180))
    except Exception as exc:  # noqa: BLE001
        out.bad(f"WS-синтез упал: {type(exc).__name__}: {exc}")
        return

    if res.get("final", {}).get("event") == "done" and res["audio"] > 0:
        out.ok(f"WS: {res['audio']} байт аудио, события={res['events']}, TTFA={res['ttfa']:.0f} мс")
    else:
        out.bad(f"WS: неожиданный результат {res.get('final')}")


# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Сквозная проверка сервера Qwen TTS Studio")
    ap.add_argument("base", nargs="?", default=os.environ.get("QTTSS_BASE", "http://127.0.0.1:8770"),
                    help="адрес сервера (по умолчанию http://127.0.0.1:8770)")
    ap.add_argument("--api-key", default=os.environ.get("QTTSS_API_KEY", ""), help="ключ API (или QTTSS_API_KEY)")
    ap.add_argument("--timeout", type=float, default=180.0, help="таймаут одного запроса, с")
    ap.add_argument("--no-color", action="store_true", help="без ANSI-цветов")
    ap.add_argument("--skip-ws", action="store_true", help="не проверять WebSocket")
    args = ap.parse_args(argv)

    out = Out(color=not args.no_color)
    api = Api(args.base, args.api_key, args.timeout)
    print(f"Проверяю сервер: {api.base}" + ("  (с API-ключом)" if api.api_key else ""))

    try:
        code, _b, _h = api.get("/api/health")
    except Exception as exc:  # noqa: BLE001
        print(f"\nСервер недоступен: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Запустите: qtts serve --backend dummy --port 8799", file=sys.stderr)
        return 2
    if code != 200:
        print(f"\n/api/health вернул {code} — дальнейшие проверки бессмысленны", file=sys.stderr)
        return 2

    basics(out, api)
    voices(out, api)
    profiles(out, api)
    synthesis(out, api)
    async_jobs(out, api)
    system(out, api)
    if not args.skip_ws:
        websocket(out, api)
    return out.summary()


if __name__ == "__main__":
    raise SystemExit(main())
