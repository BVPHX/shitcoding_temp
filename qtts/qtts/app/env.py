"""Применение переменных окружения ДО импорта torch/HIP.

Для ROCm это критично: HSA_*, GPU_*, PYTORCH_HIP_ALLOC_CONF читаются рантаймом один раз
при инициализации, поэтому менять их «на лету» бесполезно.

Пресеты подобраны под связку AMD Ryzen + Radeon RX 9060 XT (gfx1200, RDNA4) и ROCm 10.x.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .logging_setup import get_logger

log = get_logger("env")

# Пресет для RDNA4 (RX 9050/9060/9070, gfx1200/gfx1201) на ROCm 10.x.
# Всё это — «безопасные» ускорители, они не подменяют архитектуру GPU.
PRESET_RDNA4_ROCM10: Dict[str, str] = {
    # AOTriton-ядра attention (mem-efficient/flash) в SDPA: на RDNA4 дают основной прирост.
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
    # Разрешить аллокатору занять всю VRAM (по умолчанию HIP оставляет запас).
    "GPU_MAX_ALLOC_PERCENT": "100",
    # Меньше фрагментации при длинных сессиях синтеза.
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    # Отладка HIP-ошибок (0 = выкл; ставьте 4 при диагностике падений).
    "AMD_LOG_LEVEL": "0",
}

# Только для GPU, которых НЕТ в официальном списке поддержки ROCm (старые драйверы,
# iGPU gfx1151 и т.п.). Для RX 9060 XT на ROCm 7.0.2+/10.x НЕ ТРЕБУЕТСЯ.
PRESET_LEGACY_OVERRIDE: Dict[str, str] = {
    "HSA_OVERRIDE_GFX_VERSION": "12.0.0",
}

PRESETS: Dict[str, Dict[str, str]] = {
    "rdna4_rocm10": PRESET_RDNA4_ROCM10,
    "rdna3": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "GPU_MAX_ALLOC_PERCENT": "100",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    },
    "legacy_override": PRESET_LEGACY_OVERRIDE,
    "none": {},
}


def apply_env(extra: Dict[str, str] | None = None, overwrite: bool = False) -> List[Tuple[str, str, str]]:
    """Выставить переменные окружения.

    Возвращает список (имя, значение, действие), где действие:
      set    — установлено
      kept   — уже было в окружении, оставлено как есть
      forced — перезаписано (overwrite=True)
    """
    applied: List[Tuple[str, str, str]] = []
    merged: Dict[str, str] = {}
    merged.update(PRESET_RDNA4_ROCM10)
    merged.update(extra or {})
    for k, v in merged.items():
        if v is None or v == "":
            if k in os.environ:
                del os.environ[k]
                applied.append((k, "", "unset"))
            continue
        v = str(v)
        if k in os.environ and not overwrite:
            applied.append((k, os.environ[k], "kept"))
            continue
        action = "forced" if k in os.environ else "set"
        os.environ[k] = v
        applied.append((k, v, action))
    for k, v, action in applied:
        log.debug("env %s=%s (%s)", k, v, action)
    return applied


def current_relevant_env() -> Dict[str, str]:
    keys = [
        "HSA_OVERRIDE_GFX_VERSION",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
        "GPU_MAX_ALLOC_PERCENT",
        "GPU_MAX_HW_QUEUES",
        "PYTORCH_HIP_ALLOC_CONF",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "AMD_LOG_LEVEL",
        "HSA_ENABLE_SDMA",
        "PYTORCH_CUDA_ALLOC_CONF",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TORCHINDUCTOR_CACHE_DIR",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}
