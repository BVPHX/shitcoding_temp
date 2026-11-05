"""Тесты HTTP/WS API (FastAPI TestClient, движок `dummy` — без torch и GPU).

Проверяем: авторизацию, все эндпоинты синтеза (обычный, OpenAI-совместимый,
стриминг, async+задания, WebSocket), профили голосов, системные роуты,
настройки и служебные страницы GUI.
"""
from __future__ import annotations

import base64
import io
import json
import re
import struct
from pathlib import Path

import pytest
import soundfile as sf
from conftest import b64_wav, wav_bytes
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------- #
# вспомогательное
# --------------------------------------------------------------------------- #

TEXT = "Первое предложение. Второе предложение."


def riff_ok(body: bytes) -> bool:
    return body[:4] == b"RIFF" and body[8:12] == b"WAVE"


def wav_info(body: bytes):
    return sf.read(io.BytesIO(body))


def make_profile(client, name="Мама", seconds=3.0, **extra) -> dict:
    data = {"name": name, "kind": "clone", "ref_text": "Привет, это эталон.", **extra}
    r = client.post("/api/profiles", data=data,
                    files={"file": ("ref.wav", wav_bytes(seconds), "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()["profile"]


# --------------------------------------------------------------------------- #
# Авторизация
# --------------------------------------------------------------------------- #


def test_health_is_public(secure_client):
    anon, _auth = secure_client
    r = anon.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["engine"] == "dummy"


def test_protected_requires_key(secure_client):
    anon, _auth = secure_client
    for url in ("/api/config", "/api/system/info", "/api/voices", "/api/jobs", "/api/logs"):
        r = anon.get(url)
        assert r.status_code == 401, url
        assert r.headers.get("www-authenticate") == "ApiKey"


def test_protected_ok_with_key(secure_client):
    _anon, auth = secure_client
    assert auth.get("/api/config").status_code == 200
    assert auth.get("/api/system/info").status_code == 200


def test_auth_via_bearer_and_query(secure_client, tmp_path):
    from conftest import make_cfg

    from qtts.app.api.server import create_app

    cfg = make_cfg(tmp_path / "auth2", api_key="secret-key-123")
    with TestClient(create_app(cfg)) as c:
        assert c.get("/api/jobs", headers={"Authorization": "Bearer secret-key-123"}).status_code == 200
        assert c.get("/api/jobs", params={"api_key": "secret-key-123"}).status_code == 200
        assert c.get("/api/jobs", headers={"Authorization": "Bearer wrong-key"}).status_code == 401
        assert c.get("/api/jobs", params={"api_key": "wrong-key"}).status_code == 401


def test_no_key_configured_means_open(client):
    assert client.get("/api/config").status_code == 200
    assert client.get("/api/system/info").status_code == 200


def test_ui_token_from_index_authenticates(secure_client):
    anon, _auth = secure_client
    html = anon.get("/").text
    token = re.search(r'"uiToken"\s*:\s*"([^"]+)"', html)
    assert token, "в index.html нет ui_token"
    r = anon.get("/api/config", headers={"X-API-Key": token.group(1)})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# POST /api/tts
# --------------------------------------------------------------------------- #


def test_tts_returns_wav_with_metric_headers(client):
    r = client.post("/api/tts", json={"text": TEXT, "voice": "vivian", "stream": False})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert riff_ok(r.content)
    assert r.headers["x-job-id"]
    assert float(r.headers["x-audio-duration"]) > 0
    assert float(r.headers["x-tts-rtf"]) > 0
    assert float(r.headers["x-tts-elapsed-ms"]) > 0
    assert r.headers["x-request-id"]
    assert r.headers["cache-control"] == "no-store"


def test_tts_streaming_chunked(client):
    text = TEXT + " Третье предложение. Четвёртое предложение."
    with client.stream("POST", "/api/tts",
                       json={"text": text, "stream": True,
                             "format": "wav", "voice": "vivian"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")
        assert r.headers["x-job-id"]
        assert int(r.headers["x-sample-rate"]) == 24000
        chunks = list(r.iter_bytes())
    body = b"".join(chunks)
    assert riff_ok(body)
    assert len(body) > 10_000
    # ЕДИНЫЙ валидный WAV: один RIFF-заголовок, размеры «стримовые» (0xFFFFFFFF),
    # libsndfile читает до конца потока
    assert body.count(b"RIFF") == 1
    assert struct.unpack_from("<I", body, 4)[0] == 0xFFFFFFFF
    assert struct.unpack_from("<I", body, 40)[0] == 0xFFFFFFFF
    wav, sr = wav_info(body)
    assert sr == 24000 and len(wav) == (len(body) - 44) // 2
    # без дубля аудио: стрим примерно равен цельному файлу (а не в 2 раза больше)
    full = client.post("/api/tts", json={"text": text, "stream": False, "voice": "vivian"})
    assert abs(len(body) - len(full.content)) <= 44 + 4


def test_tts_pcm_stream_has_raw_content_type(client):
    r = client.post("/api/tts", json={"text": "Раз. Два. Три. Четыре.", "format": "pcm", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/L16;rate=24000")
    assert not riff_ok(r.content) and len(r.content) % 2 == 0
    # PCM-поток не дублирует аудио финальным файлом
    full = client.post("/api/tts", json={"text": "Раз. Два. Три. Четыре.", "format": "pcm", "stream": False})
    assert len(r.content) == len(full.content)


def test_tts_json_variant(client):
    r = client.post("/api/tts/json", json={"text": TEXT, "voice": "vivian"})
    assert r.status_code == 200
    d = r.json()
    assert d["mime"] == "audio/wav" and d["sample_rate"] == 24000
    assert d["audio_sec"] > 0 and d["job_id"]
    assert riff_ok(base64.b64decode(d["audio"]))
    assert d["metrics"]["elapsed_ms"] > 0 and d["metrics"]["chunks"] >= 1
    assert d["output_file"]


def test_tts_audio_endpoint_streams(client):
    r = client.post("/api/tts/audio", json={"text": "Коротко.", "format": "pcm"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/L16")
    assert len(r.content) % 2 == 0 and r.content


def test_tts_audio_endpoint_wav_is_single_valid_file(client):
    r = client.post("/api/tts/audio",
                    json={"text": "Первое предложение. Второе предложение. Третье предложение.",
                          "format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert riff_ok(r.content)
    assert r.content.count(b"RIFF") == 1          # ровно один заголовок на весь поток
    wav, sr = wav_info(r.content)                 # и файл целиком читается
    assert sr == 24000 and len(wav) == (len(r.content) - 44) // 2


def test_tts_sample_rate_override(client):
    r = client.post("/api/tts", json={"text": "Частота.", "sample_rate": 16000, "stream": False})
    assert r.status_code == 200
    _wav, sr = wav_info(r.content)
    assert sr == 16000
    assert r.headers["x-sample-rate"] == "16000"


def test_tts_with_speaker_and_instruct(client):
    r = client.post("/api/tts", json={"text": "Hello there.", "speaker": "ryan",
                                      "instruct": "Say it excitedly", "language": "en"})
    assert r.status_code == 200 and riff_ok(r.content)


def test_tts_with_profile_voice(client):
    prof = make_profile(client)
    r = client.post("/api/tts", json={"text": "Голос мамы.", "voice": prof["id"], "language": "ru"})
    assert r.status_code == 200 and riff_ok(r.content)

    job = client.get(f"/api/jobs/{r.headers['x-job-id']}").json()
    assert prof["id"] in job["voice"]
    assert job["language"] == "Russian"          # «ru» → «Russian»


def test_tts_with_inline_ref_audio_base64(client):
    """Регрессия: длинный base64 в ref_audio раньше ронял сервер в OSError(ENAMETOOLONG)."""
    r = client.post("/api/tts", json={
        "text": "Клон на лету.",
        "ref_audio": b64_wav(2.5),
        "ref_text": "Эталонная фраза.",
    })
    assert r.status_code == 200, r.text
    assert riff_ok(r.content)


def test_tts_with_inline_ref_audio_data_uri(client):
    r = client.post("/api/tts", json={
        "text": "Клон из data-URI.",
        "ref_audio": "data:audio/wav;base64," + b64_wav(2.0),
        "x_vector_only_mode": True,
    })
    assert r.status_code == 200 and riff_ok(r.content)


def test_tts_bad_ref_audio_gives_422(client):
    r = client.post("/api/tts", json={"text": "Нет эталона.", "ref_audio": "/tmp/нет-файла.wav"})
    assert r.status_code == 422


def test_tts_gen_overrides_accepted(client):
    r = client.post("/api/tts", json={"text": "Параметры.", "temperature": 0.4, "top_p": 0.9,
                                      "top_k": 20, "max_new_tokens": 512, "seed": 7})
    assert r.status_code == 200 and riff_ok(r.content)


@pytest.mark.parametrize("payload", [
    {},
    {"text": ""},
    {"text": "   "},
    {"input": ""},
])
def test_tts_requires_text(client, payload):
    r = client.post("/api/tts", json=payload)
    assert r.status_code == 422
    assert "text" in r.text.lower() or "input" in r.text.lower()


def test_tts_unknown_format_422(client):
    r = client.post("/api/tts", json={"text": "Формат.", "format": "midi"})
    assert r.status_code == 422


def test_tts_compressed_format_without_ffmpeg_501(client):
    if pytest.importorskip("shutil").which("ffmpeg"):
        pytest.skip("ffmpeg установлен — 501 не ожидается")
    r = client.post("/api/tts", json={"text": "Нужен ffmpeg.", "format": "mp3"})
    assert r.status_code == 501
    assert "ffmpeg" in r.text


def test_tts_bad_sample_rate_422(client):
    assert client.post("/api/tts", json={"text": "х", "sample_rate": 100}).status_code == 422


def test_tts_output_saved_and_listed(client, cfg):
    r = client.post("/api/tts", json={"text": "Сохранить.", "stream": False})
    assert r.status_code == 200
    job_id = r.headers["x-job-id"]

    outs = client.get("/api/outputs").json()
    names = [f["name"] for f in outs["files"]]
    assert any(job_id in n for n in names), names
    assert outs["dir"] == str(cfg.outputs_path)

    name = next(n for n in names if job_id in n)
    got = client.get(f"/api/outputs/{name}")
    assert got.status_code == 200 and riff_ok(got.content)


def test_outputs_path_traversal_blocked(client):
    assert client.get("/api/outputs/..%2f..%2fconfig.json").status_code in (400, 404)
    assert client.get("/api/outputs/нет-такого.wav").status_code == 404


# --------------------------------------------------------------------------- #
# OpenAI-совместимый эндпоинт
# --------------------------------------------------------------------------- #


def test_openai_speech_with_input_only(client):
    r = client.post("/v1/audio/speech", json={"input": "Привет, мир.", "voice": "vivian"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert riff_ok(r.content)
    assert r.headers["x-job-id"]


def test_openai_speech_response_format_and_model(client):
    r = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts", "input": "Формат pcm.", "voice": "ryan",
        "response_format": "pcm", "instructions": "Спокойно", "language": "ru",
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/L16")
    assert not riff_ok(r.content)


def test_openai_speech_stream(client):
    r = client.post("/v1/audio/speech", json={"input": "Стрим. Стрим. Стрим.",
                                              "voice": "vivian", "stream": True})
    assert r.status_code == 200 and r.content
    assert riff_ok(r.content) and r.content.count(b"RIFF") == 1
    wav, sr = wav_info(r.content)
    assert sr == 24000 and len(wav) == (len(r.content) - 44) // 2


def test_openai_speech_clone_via_ref(client):
    r = client.post("/v1/audio/speech", json={
        "input": "Клон через OpenAI-эндпоинт.",
        "ref_audio": "data:audio/wav;base64," + b64_wav(2.0),
        "ref_text": "Эталон.",
        "task_type": "Base",
    })
    assert r.status_code == 200 and riff_ok(r.content)


def test_openai_speech_empty_input_422(client):
    assert client.post("/v1/audio/speech", json={"input": "  "}).status_code == 422


# --------------------------------------------------------------------------- #
# Async-режим и задания
# --------------------------------------------------------------------------- #


def test_async_job_flow(client):
    r = client.post("/api/tts/async", json={"text": TEXT, "voice": "vivian"})
    assert r.status_code == 202
    d = r.json()
    assert d["state"] == "queued" and d["job_id"]
    assert d["status_url"] == f"/api/jobs/{d['job_id']}"

    # фоновая задача выполняется до возврата ответа клиенту (Starlette BackgroundTasks)
    job = client.get(d["status_url"]).json()
    assert job["state"] == "done"
    assert job["audio_sec"] > 0 and job["chunks_done"] >= 1
    assert job["endpoint"] == "/api/tts/async"

    audio = client.get(f"{d['status_url']}/audio")
    assert audio.status_code == 200 and riff_ok(audio.content)
    assert "attachment" in audio.headers["content-disposition"] or "inline" in audio.headers["content-disposition"]


def test_jobs_list_and_stats(client):
    client.post("/api/tts", json={"text": "Один.", "stream": False})
    client.post("/api/tts", json={"text": "Два.", "stream": False})
    d = client.get("/api/jobs").json()
    assert len(d["jobs"]) == 2
    assert d["stats"]["total"] == 2
    assert d["stats"]["by_state"]["done"] == 2
    assert d["stats"]["errors"] == 0
    assert d["stats"]["avg_rtf"] > 0
    assert client.get("/api/jobs", params={"limit": 1}).json()["jobs"].__len__() == 1
    assert client.get("/api/jobs", params={"state": "error"}).json()["jobs"] == []


def test_jobs_active_and_events(client):
    r = client.post("/api/tts", json={"text": "События.", "stream": False})
    job_id = r.headers["x-job-id"]
    assert client.get("/api/jobs/active").json() == {"active": [], "count": 0}

    job = client.get(f"/api/jobs/{job_id}", params={"events": True}).json()
    assert job["events"] and job["text"] == "События."
    assert "done" in [e["stage"] for e in job["events"]]


def test_job_not_found(client):
    assert client.get("/api/jobs/нет-такой").status_code == 404
    assert client.get("/api/jobs/нет-такой/audio").status_code == 404
    assert client.post("/api/jobs/нет-такой/cancel").status_code == 404


def test_cancel_finished_job_is_noop(client):
    r = client.post("/api/tts", json={"text": "Готово.", "stream": False})
    job_id = r.headers["x-job-id"]
    d = client.post(f"/api/jobs/{job_id}/cancel").json()
    assert d == {"job_id": job_id, "state": "done", "cancel_requested": True}


def test_clear_jobs_history(client):
    client.post("/api/tts", json={"text": "История.", "stream": False})
    assert client.delete("/api/jobs").json()["cleared"] >= 1
    assert client.get("/api/jobs").json()["jobs"] == []


def test_job_audio_requires_saved_output(cfg):
    from conftest import make_cfg

    from qtts.app.api.server import create_app

    cfg2 = make_cfg(cfg.data_path.parent / "nosave", backend="dummy")
    cfg2.audio.save_outputs = False
    with TestClient(create_app(cfg2)) as c:
        r = c.post("/api/tts", json={"text": "Без сохранения.", "stream": False})
        job_id = r.headers["x-job-id"]
        got = c.get(f"/api/jobs/{job_id}/audio")
        assert got.status_code == 404
        assert "save_outputs" in got.text


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #


def test_ws_streaming_flow(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"text": "Раз. Два. Три.", "voice": "vivian", "format": "pcm", "stream": True})
        first = ws.receive_json()
        assert first["event"] == "job" and first["job_id"]

        audio = b""
        progress = 0
        while True:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                audio += msg["bytes"]
                continue
            data = json.loads(msg["text"])
            if data["event"] == "progress":
                progress += 1
            elif data["event"] == "done":
                assert data["audio_sec"] > 0 and data["chunks"] == progress
                break
            elif data["event"] == "error":
                pytest.fail(f"WS вернул ошибку: {data}")
        assert audio and len(audio) % 2 == 0


def test_ws_bad_json_reports_error(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_text("{это не json")
        d = ws.receive_json()
        assert d == {"event": "error", "error": "некорректный JSON", "code": "bad_json"}


def test_ws_invalid_request_reports_error(client):
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"text": ""})
        d = ws.receive_json()
        assert d["event"] == "error" and d["code"] == "bad_request"


def test_ws_requires_key(secure_client):
    from starlette.websockets import WebSocketDisconnect

    anon, auth = secure_client
    with pytest.raises(WebSocketDisconnect) as ei, anon.websocket_connect("/ws/tts"):
        pass
    assert ei.value.code == 1008

    with auth.websocket_connect("/ws/tts") as ws:
        ws.send_json({"text": "С ключом.", "stream": False})
        assert ws.receive_json()["event"] == "job"


# --------------------------------------------------------------------------- #
# Голоса и профили
# --------------------------------------------------------------------------- #


def test_voices_listing(client):
    d = client.get("/api/voices").json()
    assert "vivian" in d["presets"]
    assert d["profiles"] == [] and d["count"] == len(d["presets"])
    assert d["default_profile"] is None

    make_profile(client)
    d = client.get("/api/voices").json()
    assert len(d["profiles"]) == 1 and d["default_profile"] == d["profiles"][0]["id"]


def test_presets_and_languages(client):
    p = client.get("/api/voices/presets").json()
    assert "vivian" in p["speakers"]
    langs = client.get("/api/voices/languages").json()["languages"]
    assert "Russian" in langs and "Auto" in langs


def test_create_profile_multipart(client):
    prof = make_profile(client, name="Тёплый голос", ref_text="Эталонный текст.")
    assert prof["id"] == "teplyy-golos"
    assert prof["name"] == "Тёплый голос"
    assert prof["kind"] == "clone"
    assert prof["audio"]["duration_sec"] == pytest.approx(3.0, abs=0.2)
    assert prof["audio"]["sample_rate"] == 24000
    assert prof["url"] == f"/api/profiles/{prof['id']}/audio"

    full = client.get(f"/api/profiles/{prof['id']}").json()
    assert full["ref_text"] == "Эталонный текст."
    assert client.get("/api/profiles").json()["profiles"][0]["id"] == prof["id"]


def test_create_profile_json_base64(client):
    r = client.post("/api/profiles", json={
        "name": "From Base64", "kind": "clone",
        "audio": b64_wav(2.5), "ref_text": "Текст.", "tags": ["тест"],
    })
    assert r.status_code == 200, r.text
    prof = r.json()["profile"]
    assert prof["id"] == "from-base64" and prof["tags"] == ["тест"]


def test_create_profile_json_data_uri_and_path(client, tmp_path):
    r = client.post("/api/profiles", json={
        "name": "Data URI", "audio": "data:audio/wav;base64," + b64_wav(2.0)})
    assert r.status_code == 200, r.text

    f = tmp_path / "sample.wav"
    f.write_bytes(wav_bytes(2.0))
    r = client.post("/api/profiles", json={"name": "С диска", "audio": str(f)})
    assert r.status_code == 200, r.text
    assert r.json()["profile"]["id"] == "s-diska"


def test_create_preset_profile_without_audio(client):
    r = client.post("/api/profiles", json={"name": "Мой Райан", "kind": "preset",
                                           "speaker": "ryan", "instruct": "Бодро"})
    assert r.status_code == 200
    assert r.json()["profile"]["speaker"] == "ryan"


def test_profile_create_errors(client):
    assert client.post("/api/profiles", content=b"name=x",
                       headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.post("/api/profiles", json={"kind": "clone",
                                              "audio": b64_wav(2.0)}).status_code == 422   # нет name
    assert client.post("/api/profiles", json={"name": "Без аудио",
                                              "kind": "clone"}).status_code == 422
    assert client.post("/api/profiles", json={"name": "Х", "kind": "magic"}).status_code == 422
    assert client.post("/api/profiles", json={"name": "Пресет без speaker",
                                              "kind": "preset"}).status_code == 409

    make_profile(client, name="Дубль")
    dup = client.post("/api/profiles", data={"name": "Дубль", "kind": "clone"},
                      files={"file": ("r.wav", wav_bytes(3.0), "audio/wav")})
    assert dup.status_code == 409


def test_profile_get_missing_404(client):
    assert client.get("/api/profiles/нет").status_code == 404


def test_profile_update_and_delete(client):
    prof = make_profile(client, name="Обновляемый")
    pid = prof["id"]

    r = client.patch(f"/api/profiles/{pid}", json={"name": "Новое имя", "notes": "заметка",
                                                    "tags": ["a", "b"]})
    assert r.status_code == 200
    assert r.json()["profile"]["name"] == "Новое имя"
    assert r.json()["profile"]["tags"] == ["a", "b"]

    assert client.put(f"/api/profiles/{pid}", json={"instruct": "Шёпотом"}).status_code == 200
    assert client.delete(f"/api/profiles/{pid}").json() == {"deleted": pid}
    assert client.get(f"/api/profiles/{pid}").status_code == 404
    assert client.delete(f"/api/profiles/{pid}").status_code == 404


def test_profile_duplicate(client):
    prof = make_profile(client, name="Оригинал")
    r = client.post(f"/api/profiles/{prof['id']}/duplicate", params={"name": "Копия"})
    assert r.status_code == 200
    new = r.json()["profile"]
    assert new["id"] != prof["id"] and new["name"] == "Копия"
    assert client.post("/api/profiles/нет/duplicate", params={"name": "Х"}).status_code == 404


def test_profile_audio_download_is_24k_mono(client):
    prof = make_profile(client, seconds=3.0)
    r = client.get(f"/api/profiles/{prof['id']}/audio")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    wav, sr = wav_info(r.content)
    assert sr == 24000 and wav.ndim == 1
    assert client.get("/api/profiles/нет/audio").status_code == 404


def test_profile_analyze(client):
    r = client.post("/api/profiles/analyze",
                    files={"file": ("ref.wav", wav_bytes(3.0), "audio/wav")})
    assert r.status_code == 200
    d = r.json()
    assert d["target_sr"] == 24000
    assert d["stats"]["duration_sec"] == pytest.approx(3.0, abs=0.1)
    assert isinstance(d["warnings"], list)

    assert client.post("/api/profiles/analyze").status_code == 422
    short = client.post("/api/profiles/analyze",
                        files={"file": ("s.wav", wav_bytes(0.5), "audio/wav")}).json()
    assert any("короткий" in w for w in short["warnings"])


def test_profile_test_synthesis(client):
    prof = make_profile(client, name="Тестируемый")
    r = client.post(f"/api/profiles/{prof['id']}/test", json={"text": "Проверка голоса."})
    assert r.status_code == 200 and riff_ok(r.content)
    assert r.headers["x-job-id"]
    assert client.post("/api/profiles/нет/test").status_code == 404


# --------------------------------------------------------------------------- #
# Система, модели, настройки
# --------------------------------------------------------------------------- #


def test_system_status_and_info(client, cfg):
    st = client.get("/api/system/status").json()
    assert st["backend"] == "dummy" and st["profiles"] == 0
    assert st["uptime_sec"] >= 0 and "jobs" in st

    info = client.get("/api/system/info").json()
    assert info["versions"]["app"]
    assert info["versions"]["fastapi"]
    assert info["paths"]["data"] == str(cfg.data_path)
    assert info["config_path"] == str(cfg.config_path)
    assert "dummy" in info["backends"]
    assert info["engine"]["name"] == "dummy"


def test_system_diagnostics_and_readiness(client):
    rep = client.get("/api/system/diagnostics", params={"bench": False}).json()
    assert "checks" in rep and rep["checks"]
    assert {"status", "name"} <= set(rep["checks"][0])

    txt = client.get("/api/system/diagnostics/text")
    assert txt.status_code == 200 and txt.text.strip()

    ready = client.get("/api/system/readiness").json()
    assert ready["engine"] == "dummy"
    assert "gpu_visible" in ready and isinstance(ready["problems"], list)

    gpu = client.get("/api/system/gpu").json()
    assert isinstance(gpu, dict)


def test_system_test_endpoint(client):
    r = client.post("/api/system/test", json={"text": "Проверка."})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["test"]["ok"] is True and d["test"]["audio_bytes"] > 0
    assert d["diagnostics"]["checks"]


def test_models_crud(client, cfg):
    key = cfg.models[0].key
    d = client.get("/api/models").json()
    assert [m["key"] for m in d["models"]] == [key]
    assert d["backend"] == "dummy"

    loaded = client.post("/api/models/load", json={})
    assert loaded.status_code == 200
    assert loaded.json()["model"]["key"] == key
    assert client.get("/api/models").json()["models"][0]["loaded"] is True

    un = client.post("/api/models/unload", json={})
    assert un.status_code == 200 and un.json()["ok"] is True
    assert un.json()["status"]["ready"] is False

    add = client.post("/api/models/add", json={"key": "second", "source": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"})
    assert add.status_code == 200
    assert [m["key"] for m in add.json()["models"]] == [key, "second"]
    assert client.post("/api/models/add", json={"key": "second", "source": "x"}).status_code == 409
    assert [m.key for m in cfg.models] == [key, "second"]           # конфиг сервиса обновлён

    rem = client.delete("/api/models/second")
    assert rem.status_code == 200 and [m["key"] for m in rem.json()["models"]] == [key]
    assert client.delete("/api/models/нет").status_code == 404


def test_models_delete_last_restores_default(client, cfg):
    key = cfg.models[0].key
    d = client.delete(f"/api/models/{key}").json()
    assert len(d["models"]) == 1            # пустой список моделей недопустим → дефолт


def test_config_get_masks_api_key(secure_client):
    _anon, auth = secure_client
    d = auth.get("/api/config").json()
    assert d["server"]["api_key"] == "***"
    assert d["device"]["backend"] == "dummy"


def test_config_patch_applies_and_persists(client, cfg):
    r = client.patch("/api/config", json={"generation": {"chunk_target_chars": 120,
                                                         "temperature": 0.5}})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True and d["restart_required"] == []
    assert d["config"]["generation"]["chunk_target_chars"] == 120

    on_disk = json.loads(Path(cfg.config_path).read_text(encoding="utf-8"))
    assert on_disk["generation"]["chunk_target_chars"] == 120
    assert on_disk["generation"]["temperature"] == 0.5

    # применилось к живому сервису
    assert client.get("/api/config").json()["generation"]["temperature"] == 0.5


def test_config_patch_reports_restart(client):
    d = client.patch("/api/config", json={"device": {"backend": "qwen_tts"}}).json()
    assert "backend" in d["restart_required"]


def test_config_patch_validation(client):
    assert client.patch("/api/config", json={}).status_code == 422


def test_config_save_and_reload(client, cfg):
    client.patch("/api/config", json={"generation": {"warmup_text": "Своя фраза."}})
    saved = client.post("/api/config/save")
    assert saved.status_code == 200
    assert saved.json()["path"] == str(cfg.config_path)

    Path(cfg.config_path).write_text(
        json.dumps({"generation": {"chunk_target_chars": 33}}), encoding="utf-8")
    reloaded = client.post("/api/config/reload").json()
    assert reloaded["config"]["generation"]["chunk_target_chars"] == 33
    assert client.get("/api/config").json()["generation"]["chunk_target_chars"] == 33


def test_backend_switch(client, cfg):
    r = client.post("/api/backend", json={"backend": "dummy"})
    assert r.status_code == 200
    assert r.json()["backend"] == "dummy"
    assert client.get("/api/health").json()["engine"] == "dummy"
    assert client.post("/api/backend", json={"backend": "cuda_tts"}).status_code == 422


def test_env_endpoints(client):
    got = client.get("/api/system/env").json()
    assert isinstance(got, dict) and got

    r = client.post("/api/system/env", json={"env": {"HSA_OVERRIDE_GFX_VERSION": "12.0.0"}})
    assert r.status_code == 200
    r = client.post("/api/system/env", json={"preset": "rdna4_rocm10"})
    assert r.status_code == 200


def test_logs_endpoints(client):
    # в тестах уровень WARNING: REQ-записи (26) ниже порога — включаем INFO «на лету»
    assert client.patch("/api/config", json={"log": {"level": "INFO"}}).status_code == 200

    client.get("/api/voices")                       # порождаем запись журнала
    client.post("/api/tts", json={"text": "Лог.", "stream": False})

    d = client.get("/api/logs", params={"limit": 100}).json()
    assert sum(d["counters"].values()) > 0
    assert d["logs"] and {"ts", "time", "level", "name", "msg", "rid"} <= set(d["logs"][0])
    assert any("/api/tts" in rec["msg"] for rec in d["logs"])
    assert any(rec["level"] == "REQ" for rec in d["logs"])
    assert any("job=" in rec["msg"] and "chars=4" in rec["msg"] for rec in d["logs"])
    # REQ-записи одного запроса связаны общим request-id
    rids = [rec["rid"] for rec in d["logs"] if rec["level"] == "REQ" and rec["rid"] not in ("-", "")]
    assert rids

    assert client.get("/api/logs/file").status_code == 404    # файловый лог в тестах выключен


def test_logs_level_filter(client):
    client.patch("/api/config", json={"log": {"level": "INFO"}})
    client.post("/api/tts", json={"text": "Фильтр.", "stream": False})
    client.post("/api/tts", json={"text": "Ошибка.", "format": "midi"})   # → WARNING в журнале

    reqs = client.get("/api/logs", params={"level": "REQ"}).json()["logs"]
    assert reqs and all(rec["level"] == "REQ" for rec in reqs)
    warns = client.get("/api/logs", params={"level": "WARNING"}).json()["logs"]
    assert all(rec["level"] == "WARNING" for rec in warns)


# --------------------------------------------------------------------------- #
# GUI и служебные страницы
# --------------------------------------------------------------------------- #


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "__QTTS_BOOT__" in r.text
    assert "Синтез" in r.text

    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/app.css").status_code == 200
    assert client.get("/ui/нет.js").status_code == 404


def test_docs_and_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}
    red = client.get("/docs", follow_redirects=False)
    assert red.status_code in (302, 307) and red.headers["location"] == "/api/docs"
    assert client.get("/docs").status_code == 200
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/openapi.json").status_code == 200
    spec = client.get("/api/openapi.json").json()
    assert "/api/tts" in spec["paths"] and "/v1/audio/speech" in spec["paths"]


def test_request_id_header_on_errors(client):
    r = client.post("/api/tts", json={})
    assert r.status_code == 422
    assert r.headers.get("x-request-id")


def test_method_not_allowed(client):
    assert client.get("/api/tts").status_code == 405


def test_cors_headers(client):
    r = client.options("/api/tts", headers={"Origin": "http://example.com",
                                            "Access-Control-Request-Method": "POST"})
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin")
