"""Диагностика системы и ROCm.

Собирает:
  * уровень ОС: драйвер amdgpu, PCI-устройства, sysfs (VRAM/загрузка/частоты), rocminfo, rocm-smi,
    версия ROCm, членство пользователя в группах render/video;
  * уровень PyTorch: сборка (hip/rocm), видимость GPU, имя, gfx-архитектура, VRAM, доступные
    backends SDPA (AOTriton/flash/mem-efficient), наличие flash-attn;
  * микро-бенчмарк: GEMM bf16 (TFLOPS), H2D-копия (ГБ/с), задержка запуска ядра;
  * итоговые проверки (checks) с подсказками что исправить.

Всё сериализуется в JSON — используется и в GUI («Система»), и в /api/system/diagnostics.
"""

from __future__ import annotations

import getpass
import grp
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logging_setup import GPU as GPU_LEVEL
from .logging_setup import get_logger

log = get_logger("diag")

# Официально поддержанные в ROCm 10.x потребительские GPU (Radeon RX 9000 = RDNA4).
KNOWN_GFX: Dict[str, str] = {
    "gfx1200": "RDNA4 (RX 9050 / RX 9060 / RX 9060 XT)",
    "gfx1201": "RDNA4 (RX 9070 / RX 9070 XT / RX 9070 GRE / AI PRO R9700)",
    "gfx1151": "RDNA3.5 (Ryzen AI Max, Radeon 8060S/8050S)",
    "gfx1150": "RDNA3.5 (Radeon 890M/880M)",
    "gfx1100": "RDNA3 (RX 7900 XTX/XT/GRE)",
    "gfx1101": "RDNA3 (RX 7800 XT / 7700)",
    "gfx1102": "RDNA3 (RX 7600)",
    "gfx1030": "RDNA2 (RX 6800/6900)",
    "gfx942": "CDNA3 (MI300)",
    "gfx950": "CDNA4 (MI350/MI355)",
    "gfx90a": "CDNA2 (MI250)",
}
RDNA4_GFX = {"gfx1200", "gfx1201"}

_CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #


def _read(path: str | Path) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        return None


def _run(cmd: List[str], timeout: float = 15.0) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


def _int(v: Optional[str], default: int = 0) -> int:
    try:
        return int(str(v).strip())
    except Exception:  # noqa: BLE001
        return default


def human_bytes(n: Optional[float]) -> str:
    if not n:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PiB"


# --------------------------------------------------------------------------- #
# Уровень ОС
# --------------------------------------------------------------------------- #


def amdgpu_sysfs_devices() -> List[Dict[str, Any]]:
    """Перебор /sys/class/drm/card*/device — самый надёжный способ без root."""
    out: List[Dict[str, Any]] = []
    base = Path("/sys/class/drm")
    if not base.exists():
        return out
    for card in sorted(base.glob("card[0-9]")):
        dev = card / "device"
        if not dev.exists():
            continue
        vendor = _read(dev / "vendor") or ""
        if vendor.lower() not in ("0x1002", "0x1022"):  # AMD / ATI
            continue
        vram_total = _int(_read(dev / "mem_info_vram_total"))
        vram_used = _int(_read(dev / "mem_info_vram_used"))
        gpu_busy = _read(dev / "gpu_busy_percent")
        info: Dict[str, Any] = {
            "card": card.name,
            "pci": os.path.basename(os.readlink(str(dev))) if dev.is_symlink() else str(dev),
            "vendor_id": vendor,
            "device_id": _read(dev / "device") or "",
            "vram_total_bytes": vram_total,
            "vram_used_bytes": vram_used,
            "vram_total": human_bytes(vram_total),
            "vram_used": human_bytes(vram_used),
            "gpu_busy_percent": _int(gpu_busy, -1),
            "driver": _read(dev / "driver/module/version") or (
                os.path.basename(os.readlink(str(dev / "driver"))) if (dev / "driver").is_symlink() else ""
            ),
            "power_cap_w": _int(_read(dev / "hwmon/hwmon0/power1_cap")) or None,
            "power_now_w": (_int(_read(dev / "hwmon/hwmon0/power1_input")) or 0) / 1_000_000 or None,
        }
        # gfx-архитектура: sysfs отдаёт число вида MAJOR*10000 + MINOR*100 + STEP (120000 -> gfx1200)
        gfx_raw = _int(_read(dev / "gfx_target_version"), -1)
        if gfx_raw > 0:
            major, rest = divmod(gfx_raw, 10000)
            minor, step = divmod(rest, 100)
            info["gfx"] = f"gfx{major}{minor}{step}" if (minor or step) else f"gfx{major}"
        out.append(info)
    return out


def rocm_version() -> Dict[str, Any]:
    info: Dict[str, Any] = {"found": False, "path": None, "version": None, "version_tuple": None}
    candidates = []
    env_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if env_path:
        candidates.append(Path(env_path))
    for p in Path("/opt").glob("rocm*"):
        candidates.append(p)
    candidates.append(Path("/opt/rocm"))
    for c in candidates:
        vf = c / ".info" / "version"
        if vf.exists():
            ver = _read(vf)
            info.update(found=True, path=str(c), version=ver)
            m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", ver or "")
            if m:
                info["version_tuple"] = [int(x) for x in m.groups() if x]
            break
    if not info["found"]:
        # HIP из PyTorch — тоже индикатор
        hipcc = shutil.which("hipconfig") or shutil.which("hipcc")
        if hipcc:
            rc, out, _ = _run([hipcc, "--version"] if hipcc.endswith("hipcc") else [hipcc, "--full"])
            info.update(found=True, path=hipcc, version=(out.splitlines() or [""])[0][:80])
    return info


def rocminfo_agents() -> List[Dict[str, Any]]:
    """rocminfo: точное имя маркетингового GPU + gfx + число CU."""
    bin_ = shutil.which("rocminfo")
    if not bin_:
        return []
    rc, out, _ = _run([bin_], timeout=25)
    if rc != 0:
        return []
    agents: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("Agent "):
            if cur.get("type") == "GPU":
                agents.append(cur)
            cur = {"agent": line.split()[1] if len(line.split()) > 1 else ""}
        elif ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            lk = k.lower()
            if lk == "type":
                cur["type"] = v
            elif lk == "name":
                cur["name"] = v
            elif lk == "uuid":
                cur["uuid"] = v
            elif lk == "marketing name":
                cur["marketing_name"] = v
            elif lk == "compute unit":
                cur["compute_units"] = _int(v)
            elif lk == "max clock freq. (mhz)":
                cur["max_clock_mhz"] = _int(v)
            elif lk == "pool info" and cur.get("type") == "GPU":
                pass
    if cur.get("type") == "GPU":
        agents.append(cur)
    for a in agents:
        name = a.get("name") or ""
        m = re.match(r"gfx\d+", name)
        if m:
            a["gfx"] = m.group(0)
    return agents


def rocm_smi_snapshot() -> Optional[List[Dict[str, Any]]]:
    """rocm-smi --showmeminfo vram --showuse --json (если установлен)."""
    bin_ = shutil.which("rocm-smi")
    if not bin_:
        return None
    rc, out, _ = _run([bin_, "--showmeminfo", "vram", "--showuse", "--showpower", "--json"], timeout=25)
    if rc != 0 or not out.strip():
        return None
    try:
        data = json.loads(out)
    except Exception:  # noqa: BLE001
        return None
    res: List[Dict[str, Any]] = []
    for k, v in data.items():
        if not k.lower().startswith("card") and not k.lower().startswith("gpu"):
            continue
        item: Dict[str, Any] = {"id": k}
        for kk, vv in (v or {}).items():
            kl = kk.lower()
            if "total" in kl and "vram" in kl:
                item["vram_total_bytes"] = _int(str(vv).replace("B", "").strip())
            elif "used" in kl and "vram" in kl:
                item["vram_used_bytes"] = _int(str(vv).replace("B", "").strip())
            elif "gpu use" in kl or kl.endswith("gpu use (%)"):
                item["gpu_busy_percent"] = _int(str(vv).replace("%", "").strip(), -1)
            elif "power" in kl and "watt" in kl.lower() or kl.endswith("current socket graphics card power (w)"):
                item["power_w"] = _int(str(vv).split()[0], 0)
        res.append(item)
    return res or None


def user_groups() -> Dict[str, Any]:
    user = getpass.getuser()
    try:
        groups = sorted({g.gr_name for g in grp.getgrall() if user in g.gr_mem})
        gi = os.getgrouplist(user, os.getgid()) if hasattr(os, "getgrouplist") else [os.getgid()]
        for gid in gi:
            try:
                name = grp.getgrgid(gid).gr_name
                if name not in groups:
                    groups.append(name)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        groups = []
    return {
        "user": user,
        "groups": sorted(groups),
        "render": "render" in groups,
        "video": "video" in groups,
        "kfd_readable": os.access("/dev/kfd", os.R_OK | os.W_OK) if Path("/dev/kfd").exists() else False,
        "kfd_exists": Path("/dev/kfd").exists(),
        "dri_render": [p.name for p in Path("/dev/dri").glob("renderD*")] if Path("/dev/dri").exists() else [],
    }


def system_info() -> Dict[str, Any]:
    cpu_model = "unknown"
    cores = os.cpu_count() or 0
    try:
        txt = _read("/proc/cpuinfo") or ""
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            cpu_model = m.group(1).strip()
    except Exception:  # noqa: BLE001
        pass
    mem: Dict[str, Any] = {}
    try:
        for line in (_read("/proc/meminfo") or "").splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal"):
                mem[k] = _int(v.split()[0]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return {
        "os": f"{platform.system()} {platform.release()}",
        "distro": _read("/etc/os-release").split('PRETTY_NAME="')[-1].split('"')[0]
        if _read("/etc/os-release")
        else "",
        "python": platform.python_version(),
        "arch": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_threads": cores,
        "ram_total": mem.get("MemTotal", 0),
        "ram_available": mem.get("MemAvailable", 0),
        "ffmpeg": shutil.which("ffmpeg") or None,
        "kernel_driver_amdgpu": bool(Path("/sys/module/amdgpu").exists()),
        "kernel_driver_version": _read("/sys/module/amdgpu/version") or None,
    }


# --------------------------------------------------------------------------- #
# Уровень PyTorch
# --------------------------------------------------------------------------- #


def _torch_meta() -> Dict[str, Any]:
    import torch  # локальный импорт: до вызова должны быть применены env-переменные

    meta: Dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "is_rocm_build": bool(getattr(torch.version, "hip", None)),
        "gpu_visible": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        meta["arch_list"] = torch.cuda.get_arch_list()
    except Exception:  # noqa: BLE001
        meta["arch_list"] = []
    if not meta["gpu_visible"]:
        return meta

    devs: List[Dict[str, Any]] = []
    for i in range(meta["device_count"]):
        try:
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            d = {
                "index": i,
                "name": p.name,
                "gfx": getattr(p, "gcnArchName", None),
                "compute_units": getattr(p, "multi_processor_count", None),
                "capability": f"{p.major}.{p.minor}",
                "vram_total": total,
                "vram_free": free,
                "vram_total_h": human_bytes(total),
                "vram_free_h": human_bytes(free),
                "l2_cache": getattr(p, "L2_cache_size", None),
                "uuid": getattr(p, "uuid", None),
            }
            if isinstance(d["gfx"], str):
                d["gfx"] = d["gfx"].split(":")[0].split("(")[0].strip()
            devs.append(d)
        except Exception as exc:  # noqa: BLE001
            devs.append({"index": i, "error": str(exc)})
    meta["devices"] = devs
    return meta


def _sdpa_backends(device: str) -> Dict[str, Any]:
    """Проверяем, какие backends SDPA реально доступны на этом GPU (AOTriton и пр.)."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    res: Dict[str, Any] = {}
    q = torch.randn(1, 8, 128, 64, device=device, dtype=torch.bfloat16)
    for name, be in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("math", SDPBackend.MATH),
    ):
        if be is None:
            continue
        try:
            with sdpa_kernel(be):
                o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            res[name] = bool(o is not None and torch.isfinite(o.float()).all().item())
        except Exception as exc:  # noqa: BLE001
            res[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    return res


def _bench(device: str, dtype_name: str = "bfloat16") -> Dict[str, Any]:
    import torch

    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    out: Dict[str, Any] = {"dtype": dtype_name}
    try:
        torch.cuda.synchronize()
        n = 4096
        a = torch.randn(n, n, device=device, dtype=dt)
        b = torch.randn(n, n, device=device, dtype=dt)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        dt_s = (time.perf_counter() - t0) / iters
        out["gemm_ms"] = round(dt_s * 1000, 2)
        out["gemm_tflops"] = round(2 * n**3 / dt_s / 1e12, 2)

        # H2D bandwidth
        host = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.int16, pin_memory=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        host = host.to(device)
        torch.cuda.synchronize()
        dt_s = time.perf_counter() - t0
        out["h2d_gbps"] = round(host.numel() * 2 / dt_s / 1e9, 2)

        # latency of tiny kernel launch
        x = torch.zeros(1, device=device, dtype=dt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            x.add_(1.0)
        torch.cuda.synchronize()
        out["kernel_launch_us"] = round((time.perf_counter() - t0) / 200 * 1e6, 1)
        del a, b, host, x
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _torch_probe(force_bench: bool = True) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    try:
        meta = _torch_meta()
    except ImportError as exc:
        return {"available": False, "error": f"torch не установлен: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    res.update(meta)
    if not meta.get("gpu_visible"):
        res["sdpa"] = {}
        return res
    dev = "cuda:0"
    try:
        res["sdpa"] = _sdpa_backends(dev)
    except Exception as exc:  # noqa: BLE001
        res["sdpa"] = {"error": str(exc)}
    for mod in ("flash_attn", "aotriton", "triton", "transformers", "qwen_tts", "vllm"):
        try:
            m = __import__(mod)
            res[f"pkg_{mod}"] = getattr(m, "__version__", "ok")
        except Exception:  # noqa: BLE001
            res[f"pkg_{mod}"] = None
    if force_bench:
        res["benchmark"] = _bench(dev)
    try:
        import torch

        free, total = torch.cuda.mem_get_info(0)
        res["vram_free"] = free
        res["vram_total"] = total
        res["vram_allocated"] = torch.cuda.memory_allocated(0)
        res["vram_reserved"] = torch.cuda.memory_reserved(0)
    except Exception:  # noqa: BLE001
        pass
    return res


# --------------------------------------------------------------------------- #
# Сборка отчёта
# --------------------------------------------------------------------------- #


def collect(force: bool = False, ttl: float = 5.0, bench: bool = True) -> Dict[str, Any]:
    with _LOCK:
        if not force and _CACHE["data"] is not None and (time.time() - _CACHE["ts"]) < ttl:
            return _CACHE["data"]
    t0 = time.perf_counter()
    log.log(GPU_LEVEL, "Диагностика: собираю данные системы и ROCm...")
    report: Dict[str, Any] = {
        "generated_at": time.time(),
        "system": system_info(),
        "groups": user_groups(),
        "rocm": rocm_version(),
        "sysfs_gpus": amdgpu_sysfs_devices(),
        "rocminfo": rocminfo_agents(),
        "rocm_smi": rocm_smi_snapshot(),
        "env": {
            k: os.environ.get(k)
            for k in (
                "HSA_OVERRIDE_GFX_VERSION",
                "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
                "GPU_MAX_ALLOC_PERCENT",
                "PYTORCH_HIP_ALLOC_CONF",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
            if os.environ.get(k)
        },
    }
    report["torch"] = _torch_probe(force_bench=bench)
    report["checks"] = _evaluate(report)
    report["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    statuses = [c["status"] for c in report["checks"]]
    report["health"] = (
        "error" if "error" in statuses else "warn" if "warn" in statuses else "ok"
    )
    with _LOCK:
        _CACHE["data"] = report
        _CACHE["ts"] = time.time()
    for c in report["checks"]:
        lvl = {"ok": GPU_LEVEL, "info": GPU_LEVEL, "warn": 30, "error": 40}.get(c["status"], 20)
        log.log(lvl, "[check] %s — %s", c["name"], c["message"])
    log.log(GPU_LEVEL, "Диагностика завершена за %.0f мс, статус: %s", report["took_ms"], report["health"].upper())
    return report


def _evaluate(rep: Dict[str, Any]) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []

    def add(name: str, status: str, message: str, hint: str = "") -> None:
        checks.append({"name": name, "status": status, "message": message, "hint": hint})

    sysinfo = rep["system"]
    add(
        "ОС / драйвер amdgpu",
        "ok" if sysinfo.get("kernel_driver_amdgpu") else "error",
        f"драйвер amdgpu {'загружен' if sysinfo.get('kernel_driver_amdgpu') else 'НЕ найден'}"
        + (f", версия {sysinfo['kernel_driver_version']}" if sysinfo.get("kernel_driver_version") else ""),
        "" if sysinfo.get("kernel_driver_amdgpu") else "sudo modprobe amdgpu / установите amdgpu-dkms",
    )

    rocm = rep["rocm"]
    if rocm.get("found"):
        vt = rocm.get("version_tuple") or []
        major = vt[0] if vt else 0
        status = "ok" if major >= 10 else ("warn" if major >= 7 else "warn")
        add(
            "ROCm runtime",
            status,
            f"ROCm {rocm.get('version')} в {rocm.get('path')}",
            "" if major >= 10 else "для RX 9060 XT рекомендуется ROCm 7.0.2+ (лучше 10.x): pip-индекс rocm10.0",
        )
    else:
        add("ROCm runtime", "warn", "/opt/rocm не найден — работаем через pip-сборку PyTorch+HIP",
            "это нормально: pip torch+rocm10.0 несёт собственный HIP-рантайм")

    groups = rep["groups"]
    if groups["kfd_exists"] and not groups["kfd_readable"]:
        add("Права на GPU", "error", "/dev/kfd недоступен для записи",
            f"sudo usermod -aG render,video {groups['user']} && newgrp render (или перезаход в сессию)")
    elif not groups["kfd_exists"]:
        add("Права на GPU", "warn", "/dev/kfd отсутствует (нет KFD-рантайма)", "проверьте драйвер amdgpu")
    else:
        add("Права на GPU", "ok", f"/dev/kfd доступен, группы: {','.join(g for g in ('render','video') if groups[g]) or '—'}", "")

    t = rep["torch"]
    if not t.get("available"):
        add("PyTorch", "error", t.get("error", "torch недоступен"),
            "python3 -m venv .venv && .venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm10.0")
        return checks
    add("PyTorch", "ok", f"torch {t['torch_version']}"
        + (f", HIP {t['hip_version']}" if t.get("hip_version") else ", БЕЗ HIP (CUDA/CPU-сборка!)"),
        "" if t.get("is_rocm_build") else "нужна ROCm-сборка: --index-url https://download.pytorch.org/whl/rocm10.0")

    if not t.get("gpu_visible"):
        add("Видимость GPU", "error", "torch.cuda.is_available() == False — видеокарта НЕ определена через ROCm",
            "проверьте /dev/kfd, группы render/video, HIP_VISIBLE_DEVICES и HSA_OVERRIDE_GFX_VERSION")
        return checks

    dev = (t.get("devices") or [{}])[0]
    gfx = (dev.get("gfx") or "").split(":")[0]
    name = dev.get("name") or "?"
    known = gfx in KNOWN_GFX
    is_rdna4 = gfx in RDNA4_GFX
    add(
        "GPU определён",
        "ok" if known else "warn",
        f"{name} [{gfx or 'gfx?'}] — {KNOWN_GFX.get(gfx, 'архитектура не в списке известных')}, "
        f"{dev.get('compute_units')} CU, VRAM {dev.get('vram_total_h')}",
        "" if known else "GPU не в списке официально поддержанных — может потребоваться HSA_OVERRIDE_GFX_VERSION",
    )
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") and is_rdna4:
        add("HSA_OVERRIDE_GFX_VERSION", "warn",
            f"установлен override {os.environ['HSA_OVERRIDE_GFX_VERSION']} для нативно поддержанного {gfx}",
            "для RX 9060 XT на ROCm 7.0.2+/10.x override не нужен — уберите его, иначе возможны артефакты")
    arch_list = t.get("arch_list") or []
    if arch_list and gfx and not any(gfx in a for a in arch_list):
        add("Поддержка gfx в сборке torch", "warn",
            f"в arch_list={arch_list} нет {gfx}",
            "обновите PyTorch: pip install -U torch --index-url https://download.pytorch.org/whl/rocm10.0")
    else:
        add("Поддержка gfx в сборке torch", "ok", f"{gfx} присутствует в arch_list={arch_list or 'n/a'}", "")

    sdpa = t.get("sdpa") or {}
    flash_ok = sdpa.get("flash") is True
    mem_ok = sdpa.get("mem_efficient") is True
    if flash_ok or mem_ok:
        add("SDPA backends", "ok",
            f"flash={sdpa.get('flash')}, mem_efficient={sdpa.get('mem_efficient')}, math={sdpa.get('math')}"
            + (" (AOTriton активен)" if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1" else ""),
            "" if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") else
            "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 — включает быстрые AOTriton-ядра attention")
    else:
        add("SDPA backends", "warn", f"нет быстрых ядер attention: {sdpa}",
            "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1; иначе будет медленный math-backend")

    if not t.get("pkg_qwen_tts"):
        add("Пакет qwen-tts", "error", "не установлен",
            "pip install qwen-tts==0.1.1 (тянет transformers==4.57.3) — см. scripts/setup_rocm.sh")
    else:
        add("Пакет qwen-tts", "ok", f"qwen-tts {t.get('pkg_qwen_tts')}", "")

    if t.get("pkg_transformers") and t["pkg_transformers"] != "4.57.3":
        add("Версия transformers", "warn", f"установлена {t['pkg_transformers']}, qwen-tts требует 4.57.3",
            "pip install transformers==4.57.3")

    bench = t.get("benchmark") or {}
    if bench.get("error"):
        add("Бенчмарк GPU", "error", bench["error"], "")
    elif bench:
        tf = bench.get("gemm_tflops") or 0
        # RX 9060 XT: ~20+ TFLOPS bf16 в GEMM; <5 = явный признак CPU/fallback или проблем с ядрами
        status = "ok" if tf >= 8 else "warn"
        add("Бенчмарк GPU", status,
            f"GEMM bf16 {tf} TFLOPS ({bench.get('gemm_ms')} мс на 4096³), H2D {bench.get('h2d_gbps')} ГБ/с, "
            f"запуск ядра {bench.get('kernel_launch_us')} мкс",
            "" if status == "ok" else "очень низкая производительность: проверьте, что тензоры на GPU, и arch/gfx")

    if not sysinfo.get("ffmpeg"):
        add("ffmpeg", "warn", "не найден — mp3/ogg/flac и перекодирование записи голоса недоступны",
            "sudo apt install ffmpeg (или используйте format=wav)")
    else:
        add("ffmpeg", "ok", sysinfo["ffmpeg"], "")

    vram_total = t.get("vram_total") or 0
    need = 7 * 1024**3
    add("VRAM под модель 1.7B", "ok" if vram_total >= need else "warn",
        f"свободно {human_bytes(t.get('vram_free'))} из {human_bytes(vram_total)}; "
        f"для 1.7B bf16 + токенайзер нужно ~6–7 ГБ",
        "" if vram_total >= need else "закройте другие GPU-приложения или возьмите 0.6B-модель")
    return checks


def gpu_live_stats() -> Dict[str, Any]:
    """Быстрый снимок (без бенчмарка) — для автообновления в GUI."""
    out: Dict[str, Any] = {"ts": time.time(), "sysfs": amdgpu_sysfs_devices()}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            out.update(
                gpu_visible=True,
                name=torch.cuda.get_device_properties(0).name,
                vram_total=total,
                vram_free=free,
                vram_allocated=torch.cuda.memory_allocated(0),
                vram_reserved=torch.cuda.memory_reserved(0),
            )
        else:
            out["gpu_visible"] = False
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    smi = rocm_smi_snapshot()
    if smi:
        out["rocm_smi"] = smi
    return out


def short_report(rep: Optional[Dict[str, Any]] = None) -> str:
    """Компактная текстовая сводка для терминала при старте."""
    rep = rep or collect()
    lines = ["Диагностика ROCm / GPU:"]
    for c in rep["checks"]:
        mark = {"ok": "+", "info": "i", "warn": "!", "error": "X"}.get(c["status"], "?")
        lines.append(f"  [{mark}] {c['name']}: {c['message']}")
        if c.get("hint") and c["status"] in ("warn", "error"):
            lines.append(f"      → {c['hint']}")
    return "\n".join(lines)
