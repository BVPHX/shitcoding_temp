#!/usr/bin/env python3
"""Точка входа qwen-tts-studio.

    qtts serve            # запустить сервер + веб-GUI
    qtts doctor           # диагностика ROCm/GPU в терминале
    qtts gpu              # быстрый снимок VRAM/загрузки
    qtts say "текст"      # синтез из консоли в файл
    qtts config init|show|path

ВАЖНО: переменные окружения ROCm (device.env из config.json) применяются ЗДЕСЬ,
до импорта torch — иначе HSA_*/GPU_* не подействуют.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qtts",
        description="Локальный TTS-сервер и веб-GUI для Qwen3-TTS (оптимизирован под AMD ROCm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  qtts serve --host 0.0.0.0 --port 8770 --preload\n"
            "  qtts serve --backend dummy          # проверить API/GUI без GPU и весов\n"
            "  qtts doctor                          # определилась ли карта через ROCm\n"
            "  qtts say 'Привет, мир!' -o hello.wav --voice mama\n"
        ),
    )
    p.add_argument("--config", type=str, default=None, help="путь к config.json (по умолчанию $QTTSS_HOME/config.json, ~/.local/share/qtts)")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    p.add_argument("--json-logs", action="store_true", help="логи в терминал в формате JSON (для journalctl/локагрегатора)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="запустить сервер и GUI")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--api-key", default=None)
    s.add_argument("--model", default=None, help="repo id HuggingFace или локальный каталог")
    s.add_argument("--device", default=None, help="auto | cuda:0 | cpu")
    s.add_argument("--cpu", action="store_true", help="принудительно CPU")
    s.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    s.add_argument("--backend", default=None, choices=["qwen_tts", "vllm_omni", "dummy"])
    s.add_argument("--attn", default=None, choices=["auto", "sdpa", "flash_attention_2", "eager"])
    s.add_argument("--no-stream", action="store_true", help="отключить чанковый стриминг")
    s.add_argument("--preload", action="store_true", help="загрузить модель сразу при старте")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--browser", action="store_true", help="открыть GUI в браузере")
    s.add_argument("--data-dir", default=None)
    s.add_argument("--env", action="append", default=[], metavar="K=V", help="доп. переменная окружения ROCm")
    s.add_argument("--preset", default=None, choices=["rdna4_rocm10", "rdna3", "legacy_override", "none"],
                   help="дополнительный пресет переменных окружения")

    d = sub.add_parser("doctor", help="диагностика системы/ROCm/GPU")
    d.add_argument("--json", action="store_true", help="вывести JSON")
    d.add_argument("--no-bench", action="store_true", help="без микро-бенчмарка")
    d.add_argument("--refresh", action="store_true")

    g = sub.add_parser("gpu", help="быстрый снимок состояния GPU")
    g.add_argument("--watch", type=float, default=0.0, help="обновлять каждые N секунд")

    sy = sub.add_parser("say", help="синтез из консоли")
    sy.add_argument("text")
    sy.add_argument("-o", "--output", default="output.wav")
    sy.add_argument("--voice", default=None, help="профиль или встроенный тембр")
    sy.add_argument("--ref-audio", default=None)
    sy.add_argument("--ref-text", default=None)
    sy.add_argument("--language", default=None)
    sy.add_argument("--format", default="wav", choices=["wav", "pcm", "mp3", "ogg", "flac"])
    sy.add_argument("--no-stream", action="store_true")

    c = sub.add_parser("config", help="операции с конфигом")
    c.add_argument("action", choices=["init", "show", "path"])
    return p


def parse_env_pairs(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" in item:
            k, _, v = item.partition("=")
            out[k.strip()] = v.strip()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv or ["serve"])
    if args.cmd is None:
        args.cmd = "serve"

    # 1) конфиг (без torch!)
    if args.config:
        os.environ["QTTSS_CONFIG"] = args.config
    from qtts.app.config import apply_cli_overrides, load_config

    cfg = load_config()
    if args.json_logs:
        cfg.log.json_console = True
    if args.log_level:
        cfg.log.level = args.log_level
    cfg = apply_cli_overrides(cfg, args)

    from qtts.app.logging_setup import get_logger, setup_logging

    setup_logging(cfg.log)
    log = get_logger("main")

    # 2) окружение ROCm — ДО импорта torch
    from qtts.app.env import PRESETS, apply_env

    extra = parse_env_pairs(getattr(args, "env", []))
    if getattr(args, "preset", None) and args.preset in PRESETS:
        extra = {**PRESETS[args.preset], **extra}
    applied = apply_env({**cfg.device.env, **extra})
    if applied:
        log.debug("env: %s", ", ".join(f"{k}={v}({a})" for k, v, a in applied))

    # 3) команда
    if args.cmd == "config":
        return cmd_config(args, cfg, log)
    if args.cmd == "doctor":
        return cmd_doctor(args, cfg)
    if args.cmd == "gpu":
        return cmd_gpu(args)
    if args.cmd == "say":
        return cmd_say(args, cfg, log)
    return cmd_serve(args, cfg, log)


# --------------------------------------------------------------------------- #
# Команды
# --------------------------------------------------------------------------- #


def cmd_config(args, cfg, log) -> int:
    from qtts.app.config import DEFAULT_CONFIG_PATH, save_config

    if args.action == "path":
        print(DEFAULT_CONFIG_PATH)
        return 0
    if args.action == "show":
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if DEFAULT_CONFIG_PATH.exists():
        print(f"{DEFAULT_CONFIG_PATH} уже существует — не перезаписываю", file=sys.stderr)
        return 1
    save_config(cfg, DEFAULT_CONFIG_PATH)
    print(f"создан {DEFAULT_CONFIG_PATH}")
    return 0


def cmd_doctor(args, cfg) -> int:
    from qtts.app import diagnostics

    if args.json:
        rep = diagnostics.collect(force=args.refresh, bench=not args.no_bench)
        print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
        return 0 if rep.get("health") != "error" else 2
    print()
    print("=== Диагностика qwen-tts-studio ===")
    rep = diagnostics.collect(force=args.refresh, bench=not args.no_bench)
    sysinfo = rep["system"]
    print(f"ОС:            {sysinfo.get('distro')} ({sysinfo.get('os')})")
    print(f"CPU:           {sysinfo.get('cpu_model')} [{sysinfo.get('cpu_threads')} потоков]")
    print(f"RAM:           {diagnostics.human_bytes(sysinfo.get('ram_total'))}")
    print(f"ROCm:          {rep['rocm'].get('version') or 'не найден в /opt/rocm'}")
    print(f"драйвер:       amdgpu {'загружен' if sysinfo.get('kernel_driver_amdgpu') else 'НЕТ'}"
          f" {sysinfo.get('kernel_driver_version') or ''}")
    print(f"/dev/kfd:      {'есть' if rep['groups']['kfd_exists'] else 'НЕТ'}, "
          f"доступ: {'да' if rep['groups']['kfd_readable'] else 'НЕТ'}; "
          f"группы: {','.join(rep['groups']['groups']) or '—'}")
    if rep["sysfs_gpus"]:
        for g in rep["sysfs_gpus"]:
            print(f"GPU (sysfs):   {g.get('card')} {g.get('gfx') or ''} VRAM {g.get('vram_total')} "
                  f"(занято {g.get('vram_used')}) загрузка {g.get('gpu_busy_percent')}%")
    for a in rep["rocminfo"]:
        print(f"GPU (rocminfo):{a.get('marketing_name') or a.get('name')} [{a.get('gfx')}] "
              f"{a.get('compute_units')} CU")
    t = rep["torch"]
    print(f"PyTorch:       {t.get('torch_version', 'НЕ УСТАНОВЛЕН')} "
          f"(HIP {t.get('hip_version') or '—'}, CUDA {t.get('cuda_version') or '—'})")
    print(f"GPU виден:     {t.get('gpu_visible')}")
    for d in t.get("devices", []):
        print(f"  cuda:{d.get('index')} {d.get('name')} [{d.get('gfx')}] {d.get('compute_units')} CU, "
              f"VRAM {d.get('vram_total_h')} (свободно {d.get('vram_free_h')})")
    if t.get("sdpa"):
        print(f"SDPA backends: {t['sdpa']}")
    if t.get("benchmark"):
        b = t["benchmark"]
        print(f"Бенчмарк:      GEMM bf16 {b.get('gemm_tflops')} TFLOPS, H2D {b.get('h2d_gbps')} ГБ/с, "
              f"запуск ядра {b.get('kernel_launch_us')} мкс")
    print()
    print(diagnostics.short_report(rep))
    env = rep.get("env") or {}
    if env:
        print("\nПеременные окружения: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    print()
    return 0 if rep.get("health") != "error" else 2


def cmd_gpu(args) -> int:
    import time

    from qtts.app import diagnostics

    while True:
        st = diagnostics.gpu_live_stats()
        line = []
        for g in st.get("sysfs", []):
            busy = g.get("gpu_busy_percent", -1)
            line.append(
                f"{g.get('card')}: busy {busy if busy >= 0 else '?'}% "
                f"VRAM {g.get('vram_used')}/{g.get('vram_total')}"
            )
        if st.get("gpu_visible"):
            line.append(
                f"torch: {st.get('name')} alloc {diagnostics.human_bytes(st.get('vram_allocated'))} "
                f"reserv {diagnostics.human_bytes(st.get('vram_reserved'))} "
                f"free {diagnostics.human_bytes(st.get('vram_free'))}"
            )
        print(f"[{time.strftime('%H:%M:%S')}] " + (" | ".join(line) or "GPU не найден"))
        if not args.watch:
            return 0
        time.sleep(args.watch)


def cmd_say(args, cfg, log) -> int:
    import asyncio

    from qtts.app.engine.base import SynthesisRequest
    from qtts.app.service import TTSService

    svc = TTSService(cfg)
    try:
        voice = svc.resolve_voice(args.voice, ref_audio=args.ref_audio, ref_text=args.ref_text)
    except Exception as exc:  # noqa: BLE001
        log.error("не удалось разобрать голос: %s", exc)
        return 2
    req = SynthesisRequest(
        text=args.text,
        voice=voice,
        language=svc.normalize_language(args.language),
        format=args.format,
        streaming=not args.no_stream,
    )
    log.info("синтез %d симв., голос %s, формат %s", len(args.text), voice.describe(), args.format)

    async def run() -> bytes:
        res = await svc.synthesize_full(req, client="cli", endpoint="qtts-say")
        return res["audio"], res["meta"]

    try:
        data, meta = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        log.error("ошибка синтеза: %s", exc)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    log.info("готово: %s (%.2f с аудио, %s, RTF %s)", out, meta.get("audio_sec") or 0.0,
             meta.get("format"), meta.get("rtf"))
    print(out.resolve())
    return 0


def cmd_serve(args, cfg, log) -> int:
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        log.error("uvicorn не установлен: pip install 'uvicorn[standard]'")
        return 2

    from qtts.app.api.server import create_app  # noqa: PLC0415

    app = create_app(cfg)
    host, port = cfg.server.host, cfg.server.port
    show_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    log.info("поднимаю uvicorn на http://%s:%d", host, port)
    if not cfg.server.api_key and host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "сервер открыт наружу (%s) БЕЗ api_key — любой в вашей сети может им пользоваться. "
            "Задайте server.api_key в config.json", host,
        )
    if cfg.device.backend == "qwen_tts":
        log.info("движок qwen_tts: модель будет загружена при первом запросе "
                 "(или сразу, если --preload)")
    log.info("GUI: http://%s:%d/   API-доки: http://%s:%d/api/docs", show_host, port, show_host, port)

    if args.browser or (cfg.server.open_browser and not args.no_browser):
        _open_browser(f"http://{show_host}:{port}/")

    log_level = "debug" if cfg.log.level.upper() == "DEBUG" else "info"
    ws_max = int(cfg.server.max_upload_mb * 1024 * 1024 * 2)
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            # log_config=None КРИТИЧНО: иначе uvicorn переопределяет логирование
            # (свои обработчики/формат) и дублирует access-лог поверх нашего REQ-лога.
            log_config=None,
            log_level=log_level,
            access_log=False,        # запросы печатает RequestContextMiddleware
            ssl_certfile=cfg.server.ssl_certfile or None,
            ssl_keyfile=cfg.server.ssl_keyfile or None,
            server_header=False,
            proxy_headers=bool(cfg.server.proxy_headers),
            ws_max_size=ws_max,
            ws_ping_interval=20,
            ws_ping_timeout=30,
            timeout_graceful_shutdown=cfg.server.shutdown_timeout_sec,
        )
    except OSError as exc:
        if getattr(exc, "errno", None) == 98 or "Address already in use" in str(exc):
            log.error(
                "порт %s:%d уже занят. Освободите его или запустите на другом: "
                "qtts serve --port %d", host, port, port + 1,
            )
            return 3
        log.error("не удалось запустить HTTP-сервер: %s", exc)
        return 1
    except ImportError as exc:
        log.error("не хватает зависимости: %s", exc)
        log.error("установите пакет: bash scripts/install_venv.sh (или pip install -e .)")
        return 1
    return 0


def _open_browser(url: str) -> None:
    try:
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    raise SystemExit(main())
