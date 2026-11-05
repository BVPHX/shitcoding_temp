# Qwen TTS Studio

Локальный сервер синтеза речи на **Qwen3-TTS (1.7B / 0.6B)** для Linux с оптимизацией под
**AMD ROCm 10** (связка AMD Ryzen + Radeon RX 9060 XT, gfx1200/RDNA4), веб-GUI для управления
и API, которое можно вызывать из любых приложений.

* **Голоса:** клонирование по образцу 3–10 с, встроенные тембры CustomVoice, дизайн голоса по описанию.
* **Скорость:** чанковый стриминг по предложениям — первое аудио приходит до конца генерации
  (TTFA ≈ 0.3–1.5 с на dummy, на GPU — как только модель выдаст первый чанк).
* **Интерфейсы:** HTTP (включая OpenAI-совместимый `/v1/audio/speech`), WebSocket, SSE-журнал, веб-GUI.
* **Отладка:** сквозной `request-id`, этапы генерации в терминале, диагностика ROCm/GPU из GUI и API.
* **Без GPU:** движок `dummy` позволяет поднять весь стек (API + GUI + профили + тесты) на любой машине.

```
┌───────────────────────────────────────────────────────────────────────────┐
│  qtts serve                                                               │
│    ├─ HTTP API:  /api/tts  /api/tts/json  /v1/audio/speech  /api/jobs     │
│    ├─ Streaming: chunked HTTP · WS /ws/tts · SSE /api/logs/stream          │
│    ├─ Web GUI:   /  (Синтез · Профили · Система · Настройки · Журнал)      │
│    └─ Service ── Engine: qwen_tts (ROCm/CPU) │ vllm_omni │ dummy           │
│                  Profiles · Jobs · Pipeline · Audio · Diagnostics · Logs   │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Содержание

1. [Требования](#требования)
2. [Быстрый старт](#быстрый-старт)
3. [Установка под ROCm (AMD GPU)](#установка-под-rocm-amd-gpu)
4. [Веб-GUI](#веб-gui)
5. [Командная строка](#командная-строка)
6. [API](#api)
7. [Профили голосов и клонирование](#профили-голосов-и-клонирование)
8. [Конфигурация](#конфигурация)
9. [Журналирование и отладка](#журналирование-и-отладка)
10. [Запуск как сервис (systemd)](#запуск-как-сервис-systemd)
11. [Оптимизация под ROCm и RX 9060 XT](#оптимизация-под-rocm-и-rx-9060-xt)
12. [Режим vLLM-Omni](#режим-vllm-omni)
13. [Тесты и разработка](#тесты-и-разработка)
14. [Структура проекта](#структура-проекта)
15. [Устранение неполадок](#устранение-неполадок)

---

## Требования

| Компонент | Минимум | Рекомендуется |
|---|---|---|
| ОС | Linux (x86_64) | Ubuntu 24.04 / Fedora 41+ / Arch |
| Python | 3.10 | 3.11–3.13 |
| GPU | не обязателен (CPU/`dummy`) | AMD RDNA3/RDNA4 + ROCm 10.x |
| VRAM | 4 ГБ (0.6B, float16) | 8 ГБ (1.7B, bfloat16) — RX 9060 XT подходит |
| Диск | 2 ГБ | ~12 ГБ (веса 1.7B ≈ 4 ГБ + кэш HF + venv) |
| RAM | 4 ГБ | 16 ГБ |
| ffmpeg | не нужен для `wav`/`pcm` | нужен для `mp3`/`ogg`/`flac`/`opus` |

Для AMD: достаточно **драйвера ядра `amdgpu`** и прав на `/dev/kfd` — системный пакет ROCm
ставить не обязательно, PyTorch из индекса `rocm10.0` несёт собственный HIP-рантайм.
Подробности: [docs/ROCM_SETUP.md](docs/ROCM_SETUP.md).

---

## Быстрый старт

### Вариант A — боевой режим на AMD GPU

```bash
cd qtts
./scripts/install_rocm.sh          # venv + torch(rocm10.0) + qwen-tts + проверка GPU
source .venv/bin/activate

qtts doctor                        # определилась ли карта через ROCm
qtts serve --preload               # GUI: http://127.0.0.1:8770   API: /api/docs
```

Первый запуск скачает веса `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~4 ГБ) в кэш HuggingFace.
Чтобы использовать локальную копию весов — укажите каталог: `qtts serve --model /models/Qwen3-TTS-12Hz-1.7B-Base`.

### Вариант B — без GPU (проверить API и GUI за минуту)

```bash
cd qtts
./scripts/install_venv.sh          # лёгкие зависимости + тесты
source .venv/bin/activate

qtts serve --backend dummy --port 8799
# в другом терминале:
python3 scripts/smoke_test.py http://127.0.0.1:8799
```

`dummy` вместо речи генерирует тоны, но весь тракт (запросы, стриминг, WebSocket, профили,
задачи, журнал, GUI) работает как в боевом режиме.

### Первый синтез

```bash
curl -s http://127.0.0.1:8770/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привет! Это локальный синтез речи.","language":"ru"}' -o hello.wav
```

---

## Установка под ROCm (AMD GPU)

`scripts/install_rocm.sh` делает всё сам и печатает подсказки, если чего-то не хватает:

```bash
./scripts/install_rocm.sh                       # полная установка в .venv
./scripts/install_rocm.sh --minimal             # qwen-tts --no-deps: без gradio/sox/onnxruntime
./scripts/install_rocm.sh --venv ~/venvs/qtts   # venv в другом месте
./scripts/install_rocm.sh --torch-index https://download.pytorch.org/whl/rocm10.1
./scripts/install_rocm.sh --with-vllm           # + режим прокси к vLLM-Omni
```

Вручную то же самое:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm10.0
pip install -e ".[tts]"          # или: pip install -e ".[tts-min]" && pip install --no-deps qwen-tts==0.1.1
qtts doctor
```

Проверка, что ROCm-сборка видит карту:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), \
torch.cuda.get_device_properties(0).gcnArchName)"
# ожидаемо: 2.15.0+rocm10.0 6.x.xxxx True  AMD Radeon RX 9060 XT (gfx1200...)
```

Если `is_available() == False` или нет `/dev/kfd` — см. [docs/ROCM_SETUP.md](docs/ROCM_SETUP.md#9-типовые-проблемы).

---

## Веб-GUI

GUI встроен в сервер (тот же порт, отдельная сборка не нужна): `http://127.0.0.1:8770/`.
Одностороннее приложение на ванильном JS (без CDN и внешних ресурсов), тёмная тема, русский интерфейс.

| Вкладка | Что делает |
|---|---|
| **Синтез** | текст → речь: выбор голоса (профиль/пресет), язык, формат, скорость, стриминг; плеер, метрики (RTF, TTFA, чанки), скачивание результата; WebSocket-режим |
| **Профили голосов** | создание профиля по образцу (файл, запись с микрофона в браузере, URL/base64), транскрипт эталона, предварительный анализ аудио (длительность, клиппинг, тишина), тестовый синтез, дублирование, удаление |
| **Система** | **проверка режима ROCm**: видна ли карта, архитектура (gfx), версия ROCm/HIP, PyTorch, VRAM, драйвер, список проверок с подсказками; статус движка и очереди; кнопка «Проверить режим работы» (диагностика + живой синтез) |
| **Настройки** | правка `config.json` «на лету»: устройство/dtype/attention, параметры генерации и чанкования, аудио, профиль по умолчанию, API-ключ, журнал; переключение движка, загрузка/выгрузка модели, переменные окружения ROCm (пресеты) |
| **Журнал** | живой поток записей (SSE `/api/logs/stream`), фильтр по уровню, поиск, `request-id` каждого запроса |

Если включён `server.api_key`, браузер получает одноразовый `ui_token`, встроенный в `index.html`
при выдаче страницы, — основной ключ в GUI вводить не нужно.

---

## Командная строка

```
qtts [--config PATH] [--log-level LEVEL] [--json-logs] <команда>
```

| Команда | Назначение |
|---|---|
| `qtts serve` | запустить сервер + GUI (по умолчанию, если команда не указана) |
| `qtts doctor [--json] [--no-bench] [--refresh]` | диагностика системы/ROCm/GPU с подсказками |
| `qtts gpu [--watch N]` | снимок состояния GPU (VRAM, загрузка), можно обновлять каждые N с |
| `qtts say "текст" [-o out.wav] [--voice mama] [--language ru] [--format wav\|pcm\|mp3] [--ref-audio ... --ref-text ...] [--no-stream]` | синтез из консоли |
| `qtts config init\|show\|path` | создать / показать / напечатать путь к `config.json` |

Основные флаги `serve`:

| Флаг | Смысл |
|---|---|
| `--host`, `--port` | адрес и порт (по умолчанию `127.0.0.1:8770`) |
| `--api-key` | ключ доступа к API (иначе сервер открыт всем в сети — будет предупреждение) |
| `--model` | repo id HuggingFace **или** локальный каталог с весами |
| `--backend` | `qwen_tts` (по умолчанию) \| `vllm_omni` \| `dummy` |
| `--device`, `--cpu` | `auto` \| `cuda:0` \| `cpu` |
| `--dtype` | `bfloat16` (по умолчанию, RDNA4 умеет) \| `float16` \| `float32` |
| `--attn` | `auto` \| `sdpa` \| `flash_attention_2` \| `eager` |
| `--preload` | загрузить модель при старте (иначе первый запрос ждёт загрузки) |
| `--no-stream` | выключить чанковый стриминг |
| `--data-dir` | каталог данных (профили, результаты, журнал) |
| `--env K=V` | переменная окружения ROCm (можно несколько; применяются **до** импорта torch) |
| `--preset` | `rdna4_rocm10` \| `rdna3` \| `legacy_override` \| `none` |
| `--browser`, `--no-browser` | открыть/не открывать GUI в браузере |

Коды возврата `serve`: `0` — штатная остановка, `1` — не хватает зависимостей, `3` — порт занят.

---

## API

Полный справочник с примерами запросов и ответов: **[docs/API.md](docs/API.md)**.
Интерактивная документация: `http://127.0.0.1:8770/api/docs` (Swagger UI).

### Основные эндпоинты

| Метод и путь | Назначение |
|---|---|
| `POST /api/tts` | синтез: цельный файл или стриминг чанками (`stream=true`) |
| `POST /api/tts/audio` | только аудио-поток (удобно для плеера) |
| `POST /api/tts/json` | аудио в base64 + метаданные и метрики |
| `POST /v1/audio/speech` | **OpenAI-совместимый** эндпоинт (`input`, `voice`, `response_format`, `speed`, `stream`) |
| `POST /api/tts/async` | поставить задачу в очередь → `202` + `job_id` |
| `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/audio`, `POST /api/jobs/{id}/cancel`, `DELETE /api/jobs` | задачи синтеза |
| `WS /ws/tts` | стриминг аудио + события (минимальная задержка) |
| `GET /api/voices`, `/api/voices/presets`, `/api/voices/languages` | доступные голоса и языки |
| `POST /api/profiles` | создать профиль голоса (multipart с файлом **или** JSON с `audio` = base64/URL/путь) |
| `GET/PATCH/DELETE /api/profiles/{id}`, `POST /api/profiles/{id}/duplicate`, `GET /api/profiles/{id}/audio`, `POST /api/profiles/{id}/test` | управление профилями |
| `POST /api/profiles/analyze` | разобрать аудио без сохранения (проверка эталона) |
| `GET /api/health` | жив ли сервер (публичный, без ключа) |
| `GET /api/system/info\|status\|diagnostics\|diagnostics/text\|gpu\|readiness\|env`, `POST /api/system/test\|env` | система и проверка ROCm |
| `GET /api/models`, `POST /api/models/load\|unload\|add`, `DELETE /api/models/{key}` | модели |
| `GET/PATCH /api/config`, `POST /api/config/save\|reload`, `POST /api/backend` | настройки |
| `GET /api/logs`, `GET /api/logs/file`, `GET /api/logs/stream` (SSE) | журнал |
| `GET /api/outputs`, `GET /api/outputs/{name}` | сохранённые результаты синтеза |

### Авторизация

Если задан `server.api_key`, ключ передаётся одним из способов:

```
X-API-Key: <ключ>
Authorization: Bearer <ключ>
?api_key=<ключ>            # удобно для WebSocket и EventSource
```

Без ключа защищённые эндпоинты возвращают `401`, WebSocket закрывается с кодом `1008`.

### Примеры

**HTTP (стриминг чанками):**

```bash
curl -N -s http://127.0.0.1:8770/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Длинный текст. Он будет разрезан на предложения. И отдаваться по мере готовности.",
       "voice":"mama","language":"ru","format":"pcm","stream":true}' \
  --output - | aplay -r 24000 -f S16_LE -c 1
```

**Python (цельный файл + метрики):**

```python
import requests

r = requests.post("http://127.0.0.1:8770/api/tts/json", json={
    "text": "Привет! Это локальный синтез речи.",
    "voice": "mama",              # id профиля клонированного голоса
    "language": "ru",
}, timeout=600)
d = r.json()
print(d["audio_sec"], d["metrics"])          # 2.29 {'rtf': 0.25, 'ttfa_ms': 573.0, ...}
open("out.wav", "wb").write(__import__("base64").b64decode(d["audio"]))
```

**OpenAI-совместимый вызов** (работает с существующими OpenAI-клиентами, если подменить `base_url`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8770/v1", api_key="любой-ключ")
with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts", voice="vivian", input="Hello from a local server!"
) as resp:
    resp.stream_to_file("speech.wav")
```

**WebSocket (первое аудио до конца генерации):**

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8770/ws/tts") as ws:
        await ws.send(json.dumps({"text": "Раз. Два. Три.", "format": "pcm", "stream": True}))
        pcm = b""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                pcm += msg                      # аудио-чанк — можно сразу играть
                continue
            ev = json.loads(msg)
            print(ev["event"], ev.get("type", ""))
            if ev["event"] in ("done", "error"):
                break
        open("out.pcm", "wb").write(pcm)

asyncio.run(main())
```

### Заголовки ответа с метриками

| Заголовок | Смысл |
|---|---|
| `X-Job-Id` | id задачи (по нему — статус, метрики, скачивание аудио) |
| `X-Audio-Duration` | длительность аудио, с |
| `X-Sample-Rate` | частота дискретизации ответа |
| `X-TTS-RTF` | real-time factor: время счёта / длительность аудио (<1 = быстрее реального времени) |
| `X-TTS-TTFA-ms` | time-to-first-audio, мс |
| `X-TTS-Elapsed-ms`, `X-TTS-Chunks` | полное время и число чанков |
| `X-Request-ID` | сквозной id запроса — тот же, что в журнале |

### Коды ошибок

| Код | Когда |
|---|---|
| `401` | нет/неверный API-ключ |
| `404` | задача, профиль или файл не найдены |
| `409` | профиль с таким именем уже существует; аудио задачи ещё нет |
| `413` | образец голоса больше `server.max_upload_mb` |
| `415` | неподдерживаемый `Content-Type` при создании профиля |
| `422` | пустой текст, неизвестный формат, битое аудио, неверные параметры |
| `429` | очередь задач переполнена (`generation.queue_limit`) |
| `499` | синтез отменён клиентом |
| `501` | формат требует ffmpeg, которого нет |
| `502/503` | проблема с движком (нет torch/qwen-tts, vLLM недоступен, модель не загрузилась, GPU не виден) |

---

## Профили голосов и клонирование

Профиль — это сохранённый образец голоса + метаданные. Хранится в
`<data>/profiles/<id>/{meta.json,ref.wav}`, где `id` — транслитерация имени
(«Тёплый голос» → `teplyy-golos`), а `ref.wav` всегда приводится к **24 кГц, моно, 16 бит**
(нативная частота модели).

**Как записать хороший эталон**

* 3–10 секунд чистой речи (длиннее не нужно — лишнее обрезается по `profiles.ref_target_sec`);
* один голос, без музыки, шума и реверберации;
* ровная громкость без клиппинга (сервер это проверяет и предупреждает);
* обязательно укажите **транскрипт** (`ref_text`) — с ним клон заметно точнее;
* если транскрипта нет, включите `x_vector_only` — клон только по тембру (чуть хуже просодия).

**Создание через GUI:** вкладка «Профили голосов» → «Новый профиль» → файл или запись с микрофона →
имя, транскрипт, язык → «Создать». Кнопка «Проанализировать» покажет длительность, уровень,
клиппинг и тишину ещё до сохранения.

**Создание через API:**

```bash
# multipart (как делает GUI)
curl -s -X POST http://127.0.0.1:8770/api/profiles \
  -F "name=Мама" -F "kind=clone" -F "ref_text=Привет, это мой эталонный голос." \
  -F "file=@/home/user/samples/mama.wav;type=audio/wav"

# JSON: base64 / data-URI / URL / путь на сервере
curl -s -X POST http://127.0.0.1:8770/api/profiles -H 'Content-Type: application/json' -d '{
  "name": "Диктор", "kind": "clone", "audio": "/srv/samples/dictor.mp3",
  "ref_text": "Текст этой записи.", "language": "Russian", "tags": ["работа"]
}'
```

Затем профиль используется как `voice` в любом запросе синтеза:

```bash
curl -s http://127.0.0.1:8770/api/tts -H 'Content-Type: application/json' \
  -d '{"text":"Сынок, ты поел?","voice":"mama","language":"ru"}' -o mama.wav
```

**Типы профилей:** `clone` (нужен образец), `preset` (встроенный тембр: vivian, serena, uncle_fu,
dylan, eric, ryan, aiden, ono_anna, sohee — + `instruct` для эмоции), `designed` (голос по описанию).

---

## Конфигурация

Приоритет (сверху вниз): **аргументы CLI → переменные окружения `QTTSS_*` → `config.json` → значения по умолчанию**.

**Где лежит конфиг и данные** (в каталог пакета ничего не пишется):

1. `$QTTSS_HOME` — если задан;
2. `$XDG_DATA_HOME/qtts`;
3. `~/.local/share/qtts`.

Внутри: `config.json`, `profiles/`, `outputs/`, `logs/qtts.log`, `tmp/`.
Печать путей: `qtts config path`, `qtts config show`.

**Переменные окружения:** `QTTSS_HOME`, `QTTSS_CONFIG`, `QTTSS_DATA_DIR`,
а также любая настройка в виде `QTTSS_<СЕКЦИЯ>__<ПОЛЕ>`, например:

```bash
QTTSS_SERVER__PORT=9000 QTTSS_DEVICE__DTYPE=float16 qtts serve
```

**Ключевые поля `config.json`**

```jsonc
{
  "server":     { "host": "127.0.0.1", "port": 8770, "api_key": "", "cors_origins": ["*"],
                  "preload_model": false, "open_browser": false, "max_upload_mb": 64,
                  "proxy_headers": true, "ssl_certfile": "", "ssl_keyfile": "" },
  "device":     { "backend": "qwen_tts",      // qwen_tts | vllm_omni | dummy
                  "device": "auto",           // auto | cuda:0 | cpu
                  "dtype": "bfloat16",        // bfloat16 | float16 | float32
                  "attn": "auto",             // auto | sdpa | flash_attention_2 | eager
                  "env": { "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1" },
                  "gpu_memory_utilization": 0.9, "vllm_base_url": "http://127.0.0.1:8091" },
  "generation": { "language": "Auto", "streaming": true, "non_streaming_mode": true,
                  "chunking": "sentence",     // off | sentence | paragraph
                  "chunk_target_chars": 220, "chunk_min_chars": 40, "chunk_max_chars": 700,
                  "inter_chunk_silence_ms": 90,
                  "temperature": 0.9, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
                  "max_new_tokens": 4096, "tokens_per_char": 4.0,
                  "max_concurrent": 1, "queue_limit": 32, "warmup": true, "compile": false },
  "audio":      { "format": "wav", "sample_rate": 0,   // 0 = нативная 24000 Гц
                  "normalize": true, "peak": 0.97,
                  "save_outputs": true, "keep_outputs": 300 },
  "profiles":   { "dir": "profiles", "ref_min_sec": 2.0, "ref_max_sec": 30.0,
                  "ref_target_sec": 10.0, "prompt_cache_size": 4 },
  "log":        { "level": "INFO", "file": "logs/qtts.log", "console": true, "color": true,
                  "json_console": false, "ring_size": 2000,
                  "log_requests": true, "log_generation": true, "log_devices": true },
  "data":       { "dir": "~/.local/share/qtts" },
  "models":     [ { "key": "model", "source": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "kind": "auto" } ]
}
```

Настройки меняются «на лету» — `PATCH /api/config` или вкладка «Настройки» в GUI
(сервер сам сообщит, что для применения нужен перезапуск движка).

**Модели.** `models[].source` — это repo id HuggingFace **или** путь к локальному каталогу:

| Модель | Назначение |
|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | клонирование голоса по эталону (по умолчанию) |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 встроенных тембров + эмоции |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | голос по текстовому описанию |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | то же, но легче и быстрее (4 ГБ VRAM) |

Локальную копию весов можно скачать заранее:
`hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /models/qwen3-tts-1.7b-base`.

---

## Журналирование и отладка

В терминале видны три специализированных потока записей (плюс обычные уровни):

| Уровень | Что показывает |
|---|---|
| `REQ` | внешние запросы: `→ POST /api/tts client=192.168.1.5 ua=curl/8.5.0` и `← 200 (578 мс)`, с `request-id` |
| `GEN` | этапы генерации: `первый аудио-чанк через 573 мс (job ...)`, `Готово: 2.29 с аудио, 1 чанк, RTF 0.251, TTFA 573 мс` |
| `GPU` | загрузка модели (device/dtype/attn/VRAM/gfx), диагностика ROCm, снимки GPU |

Пример живого вывода:

```
07:52:01 REQ     6facb5 → POST /api/tts  client=127.0.0.1  ua=Python-urllib/3.11
07:52:01 REQ     6facb5 запрос /api/tts job=f26f78637429 client=127.0.0.1 chars=32 voice=clone:mama stream=False fmt=wav
07:52:01 GEN          - Этап: первый аудио-чанк через 573 мс (job f26f78637429)
07:52:01 GEN          - Готово (job f26f78637429): 2.29 с аудио, 1 чанк(ов), 574 мс, RTF 0.251, TTFA 572.9 мс, 107.2 KiB
07:52:01 REQ     6facb5 ← POST /api/tts → 200 (578 мс)
```

* `request-id` проходит сквозь весь запрос: заголовок `X-Request-ID` в ответе, поле `rid` в журнале.
* Файл журнала: `<data>/logs/qtts.log` с ротацией (`log.max_bytes`, `log.backups`).
* В GUI вкладка «Журнал» — живой поток (`GET /api/logs/stream`, SSE) с фильтром по уровню.
* `qtts --json-logs serve` — машиночитаемый формат в stdout (для `journalctl`/Loki).
* Доступ к журналу через API: `GET /api/logs?limit=500&level=ERROR`, `GET /api/logs/file`.
* `uvicorn.access` намеренно приглушён: наши `REQ`-строки содержат то же плюс `request-id` и тайминг.

Уровень меняется без перезапуска: `PATCH /api/config {"log": {"level": "DEBUG"}}`.

---

## Запуск как сервис (systemd)

```bash
sudo mkdir -p /opt/qtts && sudo cp -r . /opt/qtts
cd /opt/qtts && sudo ./scripts/install_rocm.sh --venv /opt/qtts/.venv
sudo useradd --system --home-dir /var/lib/qtts --shell /usr/sbin/nologin qtts
sudo cp /opt/qtts/scripts/qtts.service /etc/systemd/system/qtts.service
#   поправьте WorkingDirectory / ExecStart / User под свою установку
sudo systemctl daemon-reload
sudo systemctl enable --now qtts
journalctl -u qtts -f
```

Юнит уже содержит: `QTTSS_HOME=/var/lib/qtts`, `StateDirectory=qtts`, доступ к `/dev/kfd`
и `/dev/dri/*`, группы `render`/`video`, перезапуск при сбое, лимиты файловых дескрипторов
и базовую изоляцию (`NoNewPrivileges`, `ProtectSystem`, `ProtectHome`).

---

## Оптимизация под ROCm и RX 9060 XT

RX 9060 XT — это **gfx1200 (RDNA4)**, официально поддерживается в ROCm 10.0/10.1,
поэтому `HSA_OVERRIDE_GFX_VERSION` **не нужен** (он предназначен для неподдерживаемых карт).

Что делает приложение по умолчанию (пресет `rdna4_rocm10`, применяется **до** импорта torch):

| Параметр | Значение | Зачем |
|---|---|---|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | включает AOTriton-ядра attention в SDPA — основной прирост на RDNA4 |
| `GPU_MAX_ALLOC_PERCENT` | `100` | разрешает аллокатору использовать всю VRAM |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | меньше фрагментации на длинных сессиях |
| `dtype` | `bfloat16` | нативная поддержка на RDNA4, в 2 раза меньше памяти чем float32 |
| `attn` | `sdpa` | flash-attn на AMD практически не собирается; SDPA + AOTriton — самый быстрый рабочий путь |
| `compile` | `false` | `torch.compile` на ROCm даёт долгие рекомпиляции и нестабильность на TTS |
| `non_streaming_mode` | `true` | отключает «симуляцию» потокового входа Dual-Track — быстрее при нашем чанковании |

Дополнительно: кэш prompt'а клонирования между чанками (эталон кодируется один раз),
budget токенов по длине текста (`tokens_per_char`), прогрев при старте (`generation.warmup`).

Подробный разбор установки, проверок, тюнинга и типовых ошибок — **[docs/ROCM_SETUP.md](docs/ROCM_SETUP.md)**.

---

## Режим vLLM-Omni

Для максимальной пропускной способности можно поднять отдельный сервер
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) и работать через него:

```bash
pip install "vllm-omni>=0.20.0"
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml --omni --port 8091

qtts serve --backend vllm_omni --port 8770
# или в конфиге: device.backend = "vllm_omni", device.vllm_base_url = "http://127.0.0.1:8091"
```

В этом режиме наше приложение остаётся точкой входа (GUI, профили, задачи, журнал),
а генерацию выполняет vLLM-Omni (два этапа talker + code2wav, непрерывный батчинг).

---

## Тесты и разработка

```bash
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/ -q                 # 218 тестов, движок dummy — torch и GPU не нужны
python3 scripts/smoke_test.py http://127.0.0.1:8799     # сквозная проверка живого сервера
ruff check qtts tests            # линтер
```

| Файл | Что покрывает |
|---|---|
| `tests/test_config.py` | загрузка/сохранение конфига, env- и CLI-переопределения, XDG-пути |
| `tests/test_pipeline.py` | нормализация текста, определение языка, резка на чанки |
| `tests/test_audio.py` | декодирование/ресемплинг/кодирование, обрезка тишины, base64/URL/data-URI |
| `tests/test_profiles.py` | хранилище профилей, транслитерация id, валидация эталонов |
| `tests/test_jobs.py` | жизненный цикл задач, статистика, подписки |
| `tests/test_service.py` | сервисный слой: голоса, чанкование, метрики, ошибки, горячая замена конфига |
| `tests/test_api.py` | HTTP/WS API: авторизация, все эндпоинты, коды ошибок, заголовки метрик |
| `tests/gui_smoke.mjs` | смоук веб-GUI в jsdom: вкладки, синтез через HTTP-стрим и WebSocket, создание профиля |

GUI-смоук требует Node.js и jsdom:

```bash
cd tests && npm i jsdom ws && node gui_smoke.mjs http://127.0.0.1:8799
```

---

## Структура проекта

```
qtts/
├── pyproject.toml              # пакет, зависимости, точки входа (qtts / qwen-tts-studio)
├── README.md
├── docs/
│   ├── API.md                  # полный справочник API
│   └── ROCM_SETUP.md           # установка и тюнинг под AMD ROCm
├── scripts/
│   ├── install_rocm.sh         # установка под AMD GPU (torch+rocm10.0, qwen-tts, проверки)
│   ├── install_venv.sh         # лёгкая установка без GPU (dummy-режим, тесты)
│   ├── smoke_test.py           # сквозная проверка работающего сервера (только stdlib)
│   └── qtts.service            # systemd-юнит
├── qtts/
│   ├── __main__.py             # CLI: serve | doctor | gpu | say | config
│   └── app/
│       ├── config.py           # dataclass-конфиг, env/CLI-переопределения, XDG-пути
│       ├── logging_setup.py    # уровни REQ/GEN/GPU, кольцевой буфер, SSE-шина, ротация файла
│       ├── env.py              # пресеты переменных ROCm (применяются до импорта torch)
│       ├── diagnostics.py      # проверка ROCm/GPU: драйвер, /dev/kfd, torch, gfx, VRAM, бенчмарк
│       ├── audio.py            # декодирование/ресемплинг/кодирование, base64/URL, обрезка тишины
│       ├── pipeline.py         # нормализация текста, определение языка, резка на чанки
│       ├── profiles.py         # хранилище профилей голосов (транслитерация id, валидация эталона)
│       ├── jobs.py             # задачи синтеза: состояния, метрики, подписчики
│       ├── service.py          # сервисный слой: очередь, чанковый стриминг, метрики
│       ├── engine/             # base | qwen_engine (ROCm) | vllm_engine | dummy_engine
│       ├── api/                # server, routes_tts, routes_profiles, routes_system, deps, middleware
│       └── web/                # index.html, app.css, app.js (SPA без сборки и CDN)
└── tests/                      # pytest (218 тестов) + gui_smoke.mjs
```

---

## Устранение неполадок

| Симптом | Причина и решение |
|---|---|
| `torch.cuda.is_available() == False` | Нет драйвера `amdgpu` или прав на `/dev/kfd`: `sudo modprobe amdgpu`, `sudo usermod -aG render,video $USER`, перелогин. Подробности в [docs/ROCM_SETUP.md](docs/ROCM_SETUP.md) |
| `No module named 'torch'` | Не установлены зависимости: `./scripts/install_rocm.sh` |
| torch без ROCm (`torch.version.hip is None`) | Поставлен CPU/CUDA-вариант: `pip uninstall -y torch torchaudio && pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm10.0` |
| `HIP out of memory` | Закройте другие GPU-приложения; возьмите модель 0.6B; включите `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`; уменьшите `generation.max_new_tokens` |
| Первый запрос очень медленный | Идёт загрузка весов и компиляция HIP/Triton-ядер: используйте `--preload` и `generation.warmup=true` |
| Медленно, SDPA уходит в math-backend | Не задан `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (пресет `rdna4_rocm10`); проверка: GUI → Система → Диагностика |
| `501 формат 'mp3' требует ffmpeg` | `sudo apt install ffmpeg` или используйте `wav`/`pcm` |
| Порт занят (`exit code 3`) | `--port 8771` или `ss -ltnp \| grep 8770` |
| `401` на все запросы | Задан `server.api_key`: передавайте `X-API-Key` (GUI получает токен сам) |
| Профиль не клонируется / похож, но «не тот» | Плохой эталон: шум, музыка, меньше 2 с; добавьте `ref_text` (транскрипт) |
| Сервер открыт наружу без ключа | Предупреждение при `host=0.0.0.0`: задайте `server.api_key`, держите порт за reverse proxy |

---

## Лицензия

Apache-2.0. Веса моделей Qwen3-TTS распространяются по лицензии Qwen — см. репозиторий модели
на HuggingFace.
