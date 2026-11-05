"""Конфиг: загрузка, переменные окружения, CLI-переопределения, сохранение."""

from __future__ import annotations

import json
from pathlib import Path

from qtts.app.config import Config, apply_cli_overrides, load_config, save_config


def test_defaults_are_rocm_friendly():
    cfg = Config()
    assert cfg.device.backend == "qwen_tts"
    assert cfg.device.dtype == "bfloat16"          # RDNA4 умеет bf16 — это самый быстрый режим
    assert cfg.device.attn == "auto"               # на ROCm выберется sdpa (flash-attn только NVIDIA)
    assert cfg.generation.compile is False         # torch.compile на ROCm = долгие рекомпиляции
    assert cfg.generation.max_concurrent == 1      # одна модель, одна GPU
    assert cfg.models and cfg.models[0].source.startswith("Qwen/Qwen3-TTS")


def test_save_and_load_roundtrip(tmp_path):
    cfg = Config()
    cfg.server.port = 9911
    cfg.generation.temperature = 0.42
    p = save_config(cfg, tmp_path / "config.json")
    assert p.exists()

    loaded = load_config(p)
    assert loaded.server.port == 9911
    assert loaded.generation.temperature == 0.42


def test_env_overrides_use_double_underscore(tmp_path, monkeypatch):
    monkeypatch.setenv("QTTSS_SERVER__PORT", "9123")
    monkeypatch.setenv("QTTSS_DEVICE__DTYPE", "float16")
    monkeypatch.setenv("QTTSS_GENERATION__CHUNK_TARGET_CHARS", "140")
    cfg = load_config(tmp_path / "absent.json")
    assert cfg.server.port == 9123                 # строка привелась к int
    assert cfg.device.dtype == "float16"
    assert cfg.generation.chunk_target_chars == 140


def test_partial_json_merged_with_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"port": 8123}, "audio": {"format": "mp3"}}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.server.port == 8123
    assert cfg.server.host == "127.0.0.1"          # остальное — из дефолтов
    assert cfg.audio.format == "mp3"
    assert cfg.generation.streaming is True


def test_broken_json_falls_back_to_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{ это не json", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.server.port == 8770


def test_cli_overrides(tmp_path):
    class Args:  # минимум полей, которые читает apply_cli_overrides
        host = "0.0.0.0"
        port = 9001
        api_key = "k"
        model = "/models/qwen3-tts"
        device = "cpu"
        cpu = True
        dtype = "float32"
        backend = "dummy"
        attn = "sdpa"
        no_stream = True
        preload = True
        no_browser = True
        browser = False
        data_dir = str(tmp_path / "d2")
        env = ["HSA_OVERRIDE_GFX_VERSION=12.0.0"]
        preset = "rdna4_rocm10"

    cfg = apply_cli_overrides(Config(), Args())
    assert cfg.server.host == "0.0.0.0" and cfg.server.port == 9001
    assert cfg.device.backend == "dummy"
    assert cfg.device.device == "cpu" and cfg.device.dtype == "float32"
    assert cfg.generation.streaming is False
    assert cfg.models[0].source == "/models/qwen3-tts"
    assert cfg.data.dir.endswith("d2")


def test_paths_are_absolute(cfg):
    assert cfg.data_path.is_absolute()
    assert cfg.profiles_path.is_absolute()
    assert cfg.outputs_path.is_absolute()
    assert cfg.profiles_path.parent == cfg.data_path


# --------------------------------------------------------------------------- #
# Домашний каталог приложения (XDG) — туда пишутся конфиг, профили, логи
# --------------------------------------------------------------------------- #


def test_home_dir_prefers_qtts_home(monkeypatch, tmp_path):
    from qtts.app.config import home_dir

    monkeypatch.setenv("QTTSS_HOME", str(tmp_path / "custom"))
    assert home_dir() == (tmp_path / "custom").resolve()


def test_home_dir_uses_xdg(monkeypatch, tmp_path):
    from qtts.app.config import home_dir

    monkeypatch.delenv("QTTSS_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert home_dir() == (tmp_path / "xdg" / "qtts").resolve()


def test_home_dir_falls_back_to_local_share(monkeypatch, tmp_path):
    from qtts.app.config import home_dir

    monkeypatch.delenv("QTTSS_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert home_dir() == (tmp_path / "home" / ".local" / "share" / "qtts").resolve()


def test_relative_log_path_resolved_against_home(monkeypatch, tmp_path):
    from qtts.app import config as config_mod

    monkeypatch.setattr(config_mod, "HOME_DIR", tmp_path)
    cfg = Config()
    cfg.log.file = "logs/qtts.log"
    assert cfg.log_path == tmp_path / "logs/qtts.log"
    cfg.log.file = "/var/log/qtts.log"
    assert cfg.log_path == Path("/var/log/qtts.log")
    cfg.log.file = ""
    assert cfg.log_path is None


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QTTSS_DATA_DIR", str(tmp_path / "dd"))
    cfg = load_config(tmp_path / "absent.json")
    assert cfg.data_path == (tmp_path / "dd").resolve()
    assert cfg.profiles_path == (tmp_path / "dd" / "profiles").resolve()
    assert cfg.outputs_path.exists()
