#!/usr/bin/env bash
# ============================================================================
#  Qwen TTS Studio — установка под AMD ROCm 10 (RDNA3/RDNA4)
#
#  Что делает скрипт:
#    1. проверяет драйвер amdgpu, /dev/kfd, группы render/video (без этого GPU не виден);
#    2. создаёт виртуальное окружение;
#    3. ставит PyTorch/torchaudio из индекса ROCm (rocm10.0 по умолчанию);
#    4. ставит сам проект и ядро Qwen3-TTS;
#    5. проверяет, что карта определилась (torch.cuda.is_available + gfx-версия);
#    6. запускает `qtts doctor`.
#
#  Проверено для: RX 9060 XT (gfx1200, RDNA4) + Ryzen, ROCm 10.x, Python 3.10–3.13.
#  Системный пакет ROCm ставить НЕ обязательно: pip-колесо torch+rocm10.0 несёт
#  собственный HIP-рантайм, от системы нужны только ядро amdgpu и права на /dev/kfd.
#
#  Примеры:
#    ./scripts/install_rocm.sh                      # полная установка в .venv
#    ./scripts/install_rocm.sh --minimal            # без gradio/sox/onnxruntime
#    ./scripts/install_rocm.sh --torch-index https://download.pytorch.org/whl/rocm10.1
#    ./scripts/install_rocm.sh --venv ~/venvs/qtts --with-vllm
# ============================================================================
set -euo pipefail

# ------------------------------------------------------------------ аргументы
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
TORCH_INDEX="https://download.pytorch.org/whl/rocm10.0"
MINIMAL=0
WITH_VLLM=0
SKIP_GPU_CHECK=0
EXTRA_PIP_ARGS=()

usage() {
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--python)      PYTHON_BIN="$2"; shift 2 ;;
        -v|--venv)        VENV_DIR="$2"; shift 2 ;;
        --torch-index)    TORCH_INDEX="$2"; shift 2 ;;
        --minimal)        MINIMAL=1; shift ;;
        --with-vllm)      WITH_VLLM=1; shift ;;
        --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
        -h|--help)        usage ;;
        *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
ok()   { echo "${C_GREEN}[ OK ]${C_OFF} $*"; }
warn() { echo "${C_YELLOW}[ !! ]${C_OFF} $*"; }
err()  { echo "${C_RED}[FAIL]${C_OFF} $*" >&2; }
step() { echo; echo "${C_GREEN}==>${C_OFF} ${C_DIM}$*${C_OFF}"; }

# ------------------------------------------------------- 1. система и драйвер
step "Проверяю систему и драйвер amdgpu"

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    ok "ОС: ${PRETTY_NAME:-$ID}"
else
    warn "/etc/os-release не найден — продолжаю"
fi

PYVER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")"
if [[ "$PYVER" == "?" ]]; then
    err "не удалось запустить $PYTHON_BIN — установите python3 (нужен 3.10+)"
    exit 1
fi
case "$PYVER" in
    3.1[0-9]) ok "Python $PYVER ($PYTHON_BIN)" ;;
    *) warn "Python $PYVER — поддерживаются 3.10–3.13, возможны проблемы" ;;
esac

if lsmod 2>/dev/null | grep -q '^amdgpu'; then
    ok "модуль ядра amdgpu загружен"
else
    warn "модуль amdgpu не найден в lsmod"
    echo "        → Ubuntu/Debian: sudo apt install amdgpu-dkms (или linux-modules-extra-\$(uname -r))"
    echo "        → затем: sudo modprobe amdgpu && sudo reboot"
fi

if [[ -e /dev/kfd ]]; then
    ok "/dev/kfd присутствует (KFD-рантайм ROCm)"
else
    warn "/dev/kfd отсутствует — PyTorch не увидит GPU"
    echo "        → sudo modprobe amdgpu; проверьте, что ядро собрано с CONFIG_HSA_AMD"
fi

RENDER_OK=0
for d in /dev/dri/renderD*; do
    [[ -e "$d" ]] || continue
    RENDER_OK=1
    if [[ -r "$d" && -w "$d" ]]; then
        ok "$d доступен на чтение/запись"
    else
        warn "$d без прав на запись для текущего пользователя"
    fi
done
[[ "$RENDER_OK" == 1 ]] || warn "/dev/dri/renderD* не найдено — нет DRM-узла GPU"

NEED_GROUPS=""
for g in render video; do
    if id -nG "$USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$g"; then
        ok "пользователь $USER в группе '$g'"
    else
        NEED_GROUPS="$NEED_GROUPS $g"
    fi
done
if [[ -n "$NEED_GROUPS" ]]; then
    warn "не хватает групп:$NEED_GROUPS"
    echo "        → sudo usermod -aG$NEED_GROUPS $USER   (затем перелогиньтесь)"
fi

if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram 2>/dev/null | sed 's/^/        /' || true
else
    echo "${C_DIM}        rocm-smi не установлен (не обязательно: pip-torch несёт свой HIP)${C_OFF}"
fi

if command -v lspci >/dev/null 2>&1; then
    GPU_LINE="$(lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display' | head -1 || true)"
    [[ -n "$GPU_LINE" ]] && ok "GPU: $GPU_LINE"
fi

if [[ "$SKIP_GPU_CHECK" == 1 ]]; then
    warn "проверка GPU пропущена (--skip-gpu-check)"
fi

# ------------------------------------------------------------- 2. venv + pip
step "Создаю виртуальное окружение: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "создано $REPO_DIR/$VENV_DIR"
else
    ok "уже существует — переиспользую"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null
ok "pip $(python -m pip --version | awk '{print $2}')"

# --------------------------------------------------- 3. PyTorch из индекса ROCm
step "Ставлю PyTorch + torchaudio из $TORCH_INDEX"
# shellcheck disable=SC2086
python -m pip install --index-url "$TORCH_INDEX" torch torchaudio ${EXTRA_PIP_ARGS[@]+"${EXTRA_PIP_ARGS[@]}"}
ok "torch $(python -c 'import torch; print(torch.__version__)')"

step "Проверяю, что ROCm-сборка видит GPU"
python - <<'PY' || warn "GPU не определился — смотрите подсказки выше (amdgpu, /dev/kfd, группы) и docs/ROCM_SETUP.md"
import torch
print(f"        torch:      {torch.__version__}")
print(f"        HIP/ROCm:   {getattr(torch.version, 'hip', None) or 'нет (это НЕ ROCm-сборка!)'}")
print(f"        GPU виден:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"        устройство: {p.name}")
    print(f"        gfx:        {getattr(p, 'gcnArchName', '?')}")
    print(f"        VRAM:       {p.total_memory / 1024**3:.1f} ГиБ")
    raise SystemExit(0)
raise SystemExit(1)
PY

# ------------------------------------------------------- 4. проект и Qwen3-TTS
step "Устанавливаю Qwen TTS Studio (editable) + зависимости"
python -m pip install -e ".[dev]"

if [[ "$MINIMAL" == 1 ]]; then
    step "Ставлю ядро Qwen3-TTS в минимальном составе (без gradio/sox/onnxruntime)"
    python -m pip install -e ".[tts-min]"
    python -m pip install --no-deps "qwen-tts==0.1.1"
else
    step "Ставлю ядро Qwen3-TTS (qwen-tts==0.1.1, transformers==4.57.3)"
    python -m pip install -e ".[tts]"
fi
ok "qwen-tts $(python -c 'import qwen_tts; print(getattr(qwen_tts, "__version__", "установлен"))' 2>/dev/null || echo 'установлен')"

if [[ "$WITH_VLLM" == 1 ]]; then
    step "Ставлю поддержку vLLM-Omni (режим внешнего сервера)"
    python -m pip install -e ".[vllm]" || warn "vllm-omni не установился — режим qwen_tts остаётся рабочим"
fi

# ffmpeg нужен только для mp3/ogg/flac/opus; wav/pcm работают без него
if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "ffmpeg не найден: форматы mp3/ogg/flac/opus будут недоступны (wav/pcm — работают)"
    echo "        → sudo apt install ffmpeg   /   sudo dnf install ffmpeg"
fi

# ---------------------------------------------------------------- 5. проверка
step "Диагностика: qtts doctor"
qtts doctor --no-bench || warn "doctor сообщил о проблемах — это не обязательно ошибка установки"

cat <<EOF

${C_GREEN}Готово.${C_OFF} Дальше:

  source $VENV_DIR/bin/activate

  # 1) скачать веса (≈4 ГБ) и запустить сервер + веб-GUI:
  qtts serve --host 127.0.0.1 --port 8770 --preload
  #    GUI:  http://127.0.0.1:8770/
  #    API:  http://127.0.0.1:8770/api/docs

  # 2) если весов ещё нет, а проверить API/GUI хочется прямо сейчас:
  qtts serve --backend dummy --port 8799

  # 3) быстрый сквозной тест API:
  python3 scripts/smoke_test.py http://127.0.0.1:8770

  # 4) автозапуск как сервис:
  sudo cp scripts/qtts.service /etc/systemd/system/qtts.service
  #    отредактируйте WorkingDirectory/ExecStart/User, затем:
  sudo systemctl daemon-reload && sudo systemctl enable --now qtts

  Конфиг и данные: \${QTTSS_HOME:-\$HOME/.local/share/qtts}
  Подробно про ROCm: docs/ROCM_SETUP.md
EOF
