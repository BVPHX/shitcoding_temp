#!/usr/bin/env bash
# ============================================================================
#  Qwen TTS Studio — быстрая установка «без GPU» (venv + сервер + GUI + тесты)
#
#  Ставит только лёгкие зависимости: FastAPI/uvicorn/numpy/soundfile + сам проект.
#  torch и веса моделей НЕ нужны — сервер работает на движке `dummy`
#  (вместо речи генерирует тоны): этого достаточно, чтобы поднять API, веб-GUI,
#  профили голосов и прогнать тесты на любой машине (в т.ч. в CI).
#
#  Для боевого синтеза на AMD-карте используйте scripts/install_rocm.sh.
#
#  Примеры:
#    ./scripts/install_venv.sh                 # venv + пакет + pytest
#    ./scripts/install_venv.sh --venv ~/v/qtts # в другой каталог
#    ./scripts/install_venv.sh --serve         # и сразу запустить dummy-сервер
# ============================================================================
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
RUN_TESTS=1
SERVE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--python)  PYTHON_BIN="$2"; shift 2 ;;
        -v|--venv)    VENV_DIR="$2"; shift 2 ;;
        --no-tests)   RUN_TESTS=0; shift ;;
        --serve)      SERVE=1; shift ;;
        -h|--help)    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

C_GREEN=$'\033[32m'; C_OFF=$'\033[0m'
ok()   { echo "${C_GREEN}[ OK ]${C_OFF} $*"; }
step() { echo; echo "${C_GREEN}==>${C_OFF} $*"; }

step "Виртуальное окружение: $VENV_DIR"
[[ -d "$VENV_DIR" ]] || "$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null
ok "python $(python -V 2>&1 | awk '{print $2}'), pip $(python -m pip --version | awk '{print $2}')"

step "Установка пакета (editable) + dev-зависимости"
python -m pip install -e ".[dev]"
ok "qtts установлен: $(command -v qtts)"

if [[ "$RUN_TESTS" == 1 ]]; then
    step "Тесты (pytest, движок dummy — без torch и GPU)"
    python -m pytest tests/ -q
    ok "тесты пройдены"
fi

cat <<EOF

${C_GREEN}Готово.${C_OFF}

  source $VENV_DIR/bin/activate
  qtts serve --backend dummy --port 8799      # API + веб-GUI без GPU
  #   GUI:  http://127.0.0.1:8799/
  #   Swagger: http://127.0.0.1:8799/api/docs
  python3 scripts/smoke_test.py http://127.0.0.1:8799

  Боевой режим на AMD: ./scripts/install_rocm.sh
EOF

if [[ "$SERVE" == 1 ]]; then
    step "Запускаю dummy-сервер на :8799 (Ctrl-C — остановить)"
    exec qtts serve --backend dummy --port 8799 --host 127.0.0.1
fi
