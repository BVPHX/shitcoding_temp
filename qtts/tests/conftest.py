"""Общие фикстуры тестов.

Все тесты работают на движке `dummy`: он не требует torch/весов/ROCm и выдаёт
детерминированное «аудио», поэтому API можно проверять где угодно (CI, ноутбук без GPU).
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import wave
from pathlib import Path

# ВАЖНО: окружение выставляется ДО импорта qtts — пути по умолчанию (HOME_DIR,
# DEFAULT_CONFIG_PATH) вычисляются один раз при импорте qtts.app.config.
_TEST_HOME = Path(os.environ.get("QTTSS_HOME") or tempfile.mkdtemp(prefix="qtts-test-home-"))
_TEST_HOME.mkdir(parents=True, exist_ok=True)
os.environ["QTTSS_HOME"] = str(_TEST_HOME)
os.environ["QTTSS_CONFIG"] = str(_TEST_HOME / "config.json")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from qtts.app.config import Config, save_config  # noqa: E402

SR = 24000


def make_cfg(tmp_path: Path, backend: str = "dummy", api_key: str = "", **over) -> Config:
    """Собрать конфиг с изолированным каталогом данных в tmp_path."""
    cfg = Config()
    cfg.data.dir = str(tmp_path / "data")
    cfg.device.backend = backend
    cfg.generation.warmup = False          # прогрев не нужен: dummy и так мгновенный
    cfg.generation.chunk_target_chars = 60  # чаще режем на чанки → проверяем стриминг
    cfg.log.level = "WARNING"
    cfg.log.console = False
    cfg.log.file = ""                       # без файлового лога в тестах
    cfg.audio.save_outputs = True
    cfg.server.api_key = api_key
    cfg.server.host = "127.0.0.1"
    cfg.server.port = 8931
    for key, val in over.items():
        section, _, field = key.partition("__")
        getattr(cfg, section).__setattr__(field, val)
    # конфиг теста лежит рядом с данными в tmp_path: update_config() пишет именно сюда
    cfg.config_path = tmp_path / "config.json"
    save_config(cfg, cfg.config_path)
    return cfg


def wav_bytes(seconds: float = 3.0, freq: float = 220.0, sr: int = SR) -> bytes:
    """Синтетический моно-WAV 16 бит (тон) — заменяет настоящий образец голоса."""
    n = int(seconds * sr)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def b64_wav(seconds: float = 3.0) -> str:
    return base64.b64encode(wav_bytes(seconds)).decode()


@pytest.fixture
def cfg(tmp_path) -> Config:
    return make_cfg(tmp_path)


@pytest.fixture
def service(cfg):
    from qtts.app.service import TTSService

    return TTSService(cfg)


@pytest.fixture
def client(cfg):
    from fastapi.testclient import TestClient

    from qtts.app.api.server import create_app

    with TestClient(create_app(cfg)) as c:
        yield c


@pytest.fixture
def secure_client(tmp_path):
    """Клиент + сервер с включённым api_key (проверка авторизации)."""
    from fastapi.testclient import TestClient

    from qtts.app.api.server import create_app

    cfg = make_cfg(tmp_path / "secure", api_key="s3cr3t-key")
    app = create_app(cfg)
    with TestClient(app) as anon, TestClient(app, headers={"X-API-Key": "s3cr3t-key"}) as auth:
        yield anon, auth
