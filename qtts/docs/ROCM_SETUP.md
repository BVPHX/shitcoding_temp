# ROCm для Qwen TTS Studio: AMD Ryzen + Radeon RX 9060 XT

Пошаговая настройка синтеза речи на GPU AMD: драйвер, права, PyTorch с ROCm 10,
переменные окружения, выбор реализации attention, оценка памяти и скорости, типовые проблемы.

Целевая конфигурация: **AMD Ryzen (AM4/AM5) + Radeon RX 9060 XT 8/16 ГБ (RDNA4, gfx1200), ROCm 10.x, Python 3.11**.

---

## 0. Коротко: что реально нужно

1. Драйвер ядра **`amdgpu`** (входит в состав ядра Linux, для RDNA4 нужно ядро ≥ 6.14, лучше 6.15+).
2. Устройство **`/dev/kfd`** и права на него (группы `render`, `video`).
3. **PyTorch из индекса ROCm** (`rocm10.0`) — он несёт собственный HIP-рантайм,
   поэтому ставить системный пакет ROCm **не обязательно**.
4. Переменные окружения ROCm, выставленные **до** импорта torch (приложение делает это само).

Всё остальное (attention, dtype, чанкование) уже подобрано в настройках по умолчанию.

| Карта | Архитектура | Поддержка ROCm | Нужен ли `HSA_OVERRIDE_GFX_VERSION` |
|---|---|---|---|
| RX 9060 / 9060 XT | RDNA4, **gfx1200** | официально с ROCm 7.0.2, в ROCm 10.x — да | **нет** |
| RX 9070 / 9070 XT | RDNA4, gfx1201 | официально | нет |
| RX 7900 XTX / 7800 XT | RDNA3, gfx1100/gfx1101 | официально | нет |
| iGPU Radeon 780M/890M | RDNA3.5, gfx1151 | частично | иногда (пресет `legacy_override`) |

Свою точную строку `gfx` покажет `qtts doctor` или GUI → Система → Диагностика.

---

## 1. Проверка железа и драйвера

```bash
# карта видна в PCIe?
lspci -nn | grep -Ei 'vga|3d|display'
#   1002:7480 ... VGA compatible controller: Advanced Micro Devices [AMD/ATI] Navi 44 [Radeon RX 9060 XT]

# драйвер ядра загружен?
lsmod | grep amdgpu
sudo dmesg | grep -iE 'amdgpu|ring gfx|hsa|kfd' | tail -20

# устройства ROCm
ls -l /dev/kfd /dev/dri/renderD*
#   crw-rw---- 1 root render /dev/kfd
#   crw-rw---- 1 root render /dev/dri/renderD128

# текущий пользователь в нужных группах?
id -nG | tr ' ' '\n' | grep -E 'render|video'

# версия ядра (для RDNA4 желательно ≥ 6.14)
uname -r
```

**Если `/dev/kfd` нет** — не загружен модуль `amdgpu`:

```bash
sudo modprobe amdgpu
# Ubuntu/Debian: sudo apt install linux-modules-extra-$(uname -r) amdgpu-dkms
# Fedora:        sudo dnf install kernel-modules-extra akmod-amdgpu   (или amdgpu-dkms)
# Arch:          sudo pacman -S linux-firmware-amdgpu
sudo reboot
```

**Если нет прав** (`Permission denied`, `HSA_STATUS_ERROR`):

```bash
sudo usermod -aG render,video $USER
# затем полностью выйти из сессии и зайти снова (или newgrp render)
```

**rocm-smi** (необязательно, но удобно для наблюдения):

```bash
sudo apt install rocm-smi-lib python3-pyamdgpuinfo   # или из репозитория ROCm
rocm-smi --showproductname --showmeminfo vram --showuse
```

> Приложение работает и без `rocm-smi`: состояние GPU читается через PyTorch/HIP
> (`qtts gpu --watch 2`, GUI → Система).

---

## 2. Виртуальное окружение и PyTorch с ROCm 10

```bash
cd qtts
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel

# PyTorch под ROCm 10.0 (cp310–cp313)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm10.0
```

Проверка — самая важная команда всей инструкции:

```bash
python - <<'PY'
import torch
print("torch:      ", torch.__version__)          # 2.15.0+rocm10.0
print("HIP/ROCm:   ", torch.version.hip)          # не None → это ROCm-сборка
print("GPU виден:  ", torch.cuda.is_available())  # True
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("устройство:", p.name)                  # AMD Radeon RX 9060 XT
    print("gfx:       ", p.gcnArchName)           # gfx1200:sramecc-:xnack-
    print("VRAM:      ", round(p.total_memory / 1024**3, 1), "ГиБ")
    x = torch.randn(4096, 4096, device="cuda:0", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print("bf16 matmul:", (x @ x).sum().item() is not None)
PY
```

Соответствие версий (если собираете вручную):

| ROCm | PyTorch |
|---|---|
| 10.0.x | 2.10–2.12 (в индексе `rocm10.0` доступно и 2.15) |
| 10.1.x | 2.12–2.14 |
| 7.0.x | 2.8–2.9 (индекс `rocm7.0`) |

---

## 3. Установка приложения

```bash
./scripts/install_rocm.sh                 # всё сразу: проверки + venv + torch(rocm) + qwen-tts + doctor
./scripts/install_rocm.sh --minimal       # без gradio/sox/onnxruntime (qwen-tts ставится --no-deps)
```

или вручную:

```bash
pip install -e ".[tts]"          # qwen-tts==0.1.1 (transformers==4.57.3, accelerate==1.12.0)
# или лёгкий вариант:
pip install -e ".[tts-min]" && pip install --no-deps qwen-tts==0.1.1

qtts doctor                      # диагностика: драйвер, /dev/kfd, torch, gfx, VRAM, env
qtts serve --preload             # запуск: GUI http://127.0.0.1:8770
```

`qtts doctor` печатает список проверок с подсказками (`ok` / `warn` / `error`), например:

```
[+] ОС / драйвер amdgpu: модуль загружен
[+] Права на GPU: /dev/kfd доступен, группы render,video
[+] PyTorch 2.15.0+rocm10.0, HIP 6.4.xxxxx, GPU виден
[+] Устройство: AMD Radeon RX 9060 XT, gfx1200, VRAM 15.9 ГиБ
[!] ROCm runtime: /opt/rocm не найден — работаем через pip-сборку PyTorch+HIP (это нормально)
```

Тот же отчёт доступен в GUI (вкладка **Система → Диагностика ROCm**) и по API:
`GET /api/system/diagnostics`, `GET /api/system/diagnostics/text`, `GET /api/system/readiness`.

---

## 4. Переменные окружения ROCm

HIP читает эти переменные **один раз при инициализации**, поэтому менять их «на лету» бесполезно.
Приложение выставляет их **до импорта torch** — из `device.env`, `--env K=V` или пресета `--preset`.

### Пресет `rdna4_rocm10` (по умолчанию для RX 9060 XT)

| Переменная | Значение | Зачем |
|---|---|---|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | включает AOTriton-ядра (flash/mem-efficient attention) внутри SDPA — **основной прирост скорости на RDNA4** |
| `GPU_MAX_ALLOC_PERCENT` | `100` | разрешает аллокатору HIP занять всю VRAM (по умолчанию остаётся запас) |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | меньше фрагментации памяти при длинных сессиях синтеза |
| `AMD_LOG_LEVEL` | `0` | отладка HIP выключена (ставьте `4` при диагностике падений) |

### Пресет `rdna3`

То же без `AMD_LOG_LEVEL` — для RX 7000 (gfx1100/gfx1101).

### Пресет `legacy_override`

```
HSA_OVERRIDE_GFX_VERSION=12.0.0
```

**Только** для карт, которых нет в официальном списке поддержки ROCm (старые драйверы,
некоторые iGPU). Для RX 9060 XT на ROCm 7.0.2+/10.x **не требуется** и может ухудшить работу:
подмена версии архитектуры отключает специфичные для gfx1200 ядра.

### Как задать свои значения

```jsonc
// config.json
"device": {
  "env": {
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
    "GPU_MAX_ALLOC_PERCENT": "100",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    "HIP_VISIBLE_DEVICES": "0"
  }
}
```

```bash
qtts serve --env HSA_ENABLE_SDMA=0 --env AMD_LOG_LEVEL=4 --preset rdna4_rocm10
```

```ini
# /etc/systemd/system/qtts.service
Environment=PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

Проверить, какие переменные реально заданы в процессе: `GET /api/system/env`
или GUI → Система (блок «Переменные окружения»).

---

## 5. Выбор декодера и реализации attention

Qwen3-TTS — двухэтапная модель: **talker** (авторегрессионная генерация аудио-кодов 12.5 Гц)
и **speech tokenizer/декодер** (коды → волна 24 кГц). Узкое место на GPU — именно attention
в talker'е, поэтому выбор его реализации даёт основную разницу в скорости.

| Реализация | AMD/ROCm | Комментарий |
|---|---|---|
| `flash_attention_2` | ✗ практически недоступно | пакет `flash-attn` собирается под CUDA; на ROCm — только вручную и не для RDNA4 |
| **`sdpa` + AOTriton** | ✓ **рекомендуется** | `attn_implementation="sdpa"` + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`: PyTorch вызывает AOTriton-ядра flash/mem-efficient attention |
| `sdpa` без AOTriton | ✓ медленнее | SDPA уходит в mem-efficient или math-backend |
| `eager` | ✓ самый медленный | запасной вариант, если SDPA падает |

Что делает приложение:

* `device.attn = "auto"` → пробуем `flash_attention_2`, если пакет реально установлен, иначе `sdpa`;
* если загрузка модели с выбранным `attn` не удалась — автоматический фолбэк по цепочке
  `flash_attention_2 → sdpa → eager` (каждая попытка логируется);
* при загрузке модели выполняется **проба SDPA-бэкендов** (`_probe_sdpa`): видно, какие из
  `flash` / `mem_efficient` / `math` работают на вашей карте — результат в
  `GET /api/system/status` → `info.sdpa` и в GUI → Система;
* если `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL != 1`, сервер честно предупреждает в журнале,
  что SDPA может уйти в медленный math-backend.

Другие решения по умолчанию (и почему):

| Параметр | Значение | Причина |
|---|---|---|
| `device.dtype` | `bfloat16` | RDNA4 поддерживает bf16 нативно: вдвое меньше памяти и выше пропускная способность, чем у float32; стабильнее float16 по диапазону значений |
| `generation.compile` | `false` | `torch.compile` на ROCm для TTS даёт долгие рекомпиляции на каждом новом размере чанка и нестабильность — прирост не окупается |
| `generation.non_streaming_mode` | `true` | отключает «симуляцию» потокового текстового входа Dual-Track (она медленнее), так как стриминг у нас организован чанками по предложениям |
| `generation.max_concurrent` | `1` | одна карта и одна копия модели: параллельные запросы встают в очередь, а не конкурируют за VRAM |
| `device.allow_tf32`, `matmul_precision=high` | вкл | дополнительные разрешения на быстрые пути матричных умножений |
| кэш prompt'а клонирования | вкл (`profiles.prompt_cache_size`) | эталонный образец кодируется один раз и переиспользуется для всех чанков и запросов |

---

## 6. Память (VRAM)

| Модель | Веса | Реально в VRAM при синтезе |
|---|---|---|
| 1.7B, bfloat16 | ~3.4 ГБ | ~5–6 ГБ (+ KV-кэш и буферы декодера) |
| 1.7B, float32 | ~6.8 ГБ | ~9–10 ГБ |
| 0.6B, bfloat16 | ~1.2 ГБ | ~2.5–3 ГБ |

Практические советы для 8-гигабайтной карты:

* держите `dtype=bfloat16`;
* `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (в пресете уже есть);
* ограничьте `generation.max_new_tokens` (по умолчанию 4096 — потолок на один чанк);
* уменьшите `generation.chunk_max_chars` (700 → 400): короче чанк — меньше KV-кэш;
* закройте браузер с видео и игры: они отъедают VRAM;
* при нехватке памяти возьмите `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

Текущее потребление: `qtts gpu --watch 2`, GUI → Система, `GET /api/system/gpu`.

---

## 7. Скорость и задержки

Метрики, которые отдаёт сервер: **RTF** (время счёта / длительность аудио, < 1 = быстрее реального
времени), **TTFA** (время до первого аудио-чанка), `tok_per_sec`, число чанков.
Смотреть: заголовки `X-TTS-RTF` / `X-TTS-TTFA-ms`, `GET /api/jobs/{id}`, `GET /api/jobs` → `stats`
(`avg_rtf`, `avg_ttfa_ms`), GUI → Синтез.

Что влияет сильнее всего:

| Параметр | Эффект |
|---|---|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | основной прирост attention на RDNA4 |
| `dtype=bfloat16` | ~2× к float32 по памяти и скорости |
| `generation.warmup=true` + `--preload` | убирают «долгий первый запрос» (компиляция HIP/Triton-ядер) |
| `generation.chunking=sentence`, `chunk_target_chars≈180–260` | короткий чанк → меньший TTFA; слишком короткий → больше накладных расходов и хуже просодия |
| `inter_chunk_silence_ms=90` | естественная пауза между чанками при склейке |
| `non_streaming_mode=true` | быстрее на длинных фразах |
| `compile=false` | стабильность важнее гипотетического прироста на ROCm |
| `max_concurrent=1` | предсказуемая задержка; для нескольких параллельных клиентов лучше режим vLLM-Omni |

Порядок величин для 1.7B на RDNA4: RTF ≈ 0.3–0.7 (то есть 10 секунд речи за 3–7 секунд счёта),
TTFA первого чанка — около 0.5–1.5 с после прогрева. Точные числа для вашей карты:

```bash
qtts serve --preload --backend qwen_tts
# в другом терминале
python3 scripts/smoke_test.py http://127.0.0.1:8770     # покажет RTF/TTFA по каждому запросу
```

---

## 8. Альтернатива: vLLM-Omni на ROCm

Если нужна максимальная пропускная способность (несколько клиентов, непрерывный батчинг),
поднимите отдельный сервер vLLM-Omni и переключите приложение в режим прокси:

```bash
pip install "vllm-omni>=0.20.0"
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml --omni --port 8091

qtts serve --backend vllm_omni --port 8770
# device.vllm_base_url = "http://127.0.0.1:8091"
```

Особенности: двухэтапный пайплайн (talker + code2wav), `gpu_memory_utilization ≈ 0.3` на этап,
на ROCm используется Model Runner V1. Наш сервер при этом остаётся единой точкой входа
(GUI, профили, задачи, журнал, OpenAI-совместимый API).

---

## 9. Типовые проблемы

| Симптом | Причина | Решение |
|---|---|---|
| `torch.cuda.is_available() == False`, `/dev/kfd` отсутствует | не загружен `amdgpu` | `sudo modprobe amdgpu`; установите `amdgpu-dkms`/`linux-modules-extra`; перезагрузка |
| `HIP error: no ROCm-capable device is detected` | нет прав на `/dev/kfd`, `/dev/dri/renderD*` | `sudo usermod -aG render,video $USER` + перелогин |
| `torch.version.hip is None` | стоит CPU/CUDA-сборка torch | `pip uninstall -y torch torchaudio && pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm10.0` |
| `No module named 'torch'` (ошибка `no_torch`, HTTP 503) | venv не активирован или зависимости не установлены | `./scripts/install_rocm.sh` |
| `gfx1200 not supported` / `Unknown GPU arch` | старый ROCm-стек | обновите индекс torch до `rocm10.0`; при необходимости ядро ≥ 6.14 |
| `HIP out of memory` | не хватает VRAM | см. раздел [Память](#6-память-vram); освободите карту, возьмите 0.6B |
| Синтез работает, но медленно; в статусе `sdpa.math: true`, `flash: ошибка` | AOTriton не включён | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` **до запуска** (пресет `rdna4_rocm10`); перезапустите сервер |
| Первый запрос висит 30–60 с | загрузка весов + компиляция ядер | `--preload`, `generation.warmup=true` |
| Каждый запрос начинается с долгой паузы, в логе `recompiling` | включён `torch.compile` | `generation.compile=false` |
| `amdgpu: ring gfx timeout` в `dmesg`, карта «отваливается` | перегрев/недостаток питания/агрессивный разгон | проверьте температуры (`rocm-smi`), блок питания, отключите разгон; обновите ядро |
| `rocm-smi` не видит карту | не установлен системный ROCm | не обязательно: используйте `qtts gpu` / GUI |
| Python 3.14, torch не ставится | колёс под вашу версию Python нет в индексе ROCm | используйте Python 3.11–3.13 |
| `501 формат 'mp3' требует ffmpeg` | нет ffmpeg | `sudo apt install ffmpeg` или используйте `wav`/`pcm` |

---

## 10. Отчёт о проблеме

Соберите три вещи — этого достаточно для диагностики:

```bash
qtts doctor --json > doctor.json
AMD_LOG_LEVEL=4 qtts serve --preload > server.log 2>&1    # подробный лог HIP
# и через GUI/API:
#   GET /api/system/diagnostics/text  → diagnostics.txt
```

В отчёте видны: версия ядра и драйвера, наличие `/dev/kfd`, группы пользователя,
версии torch/HIP/transformers/qwen-tts, имя карты и `gfx`, объём VRAM, заданные переменные
окружения ROCm, какие SDPA-бэкенды работают, и результаты каждой проверки с подсказками.
