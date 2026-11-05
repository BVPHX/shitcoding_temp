# Справочник API — Qwen TTS Studio

Базовый адрес по умолчанию: `http://127.0.0.1:8770`
Интерактивная документация (Swagger UI): `/api/docs` · ReDoc: `/api/redoc` · спецификация: `/api/openapi.json`

* [Авторизация](#авторизация)
* [Синтез речи](#синтез-речи)
* [OpenAI-совместимый эндпоинт](#openai-совместимый-эндпоинт)
* [Задачи (async-режим)](#задачи-async-режим)
* [WebSocket /ws/tts](#websocket-wstts)
* [Голоса и профили](#голоса-и-профили)
* [Система и диагностика ROCm](#система-и-диагностика-rocm)
* [Модели](#модели)
* [Настройки](#настройки)
* [Журнал](#журнал)
* [Результаты синтеза](#результаты-синтеза)
* [Объекты данных](#объекты-данных)
* [Ошибки](#ошибки)

---

## Авторизация

Если `server.api_key` пустой — API открыт (при `host=0.0.0.0` сервер предупреждает об этом в журнале).
Если ключ задан, передайте его одним из способов:

```
X-API-Key: <ключ>
Authorization: Bearer <ключ>
?api_key=<ключ>          # для WebSocket и EventSource (SSE)
```

* `GET /api/health` доступен всегда (для мониторинга).
* Без ключа: `401` + заголовок `WWW-Authenticate: ApiKey`; WebSocket закрывается кодом `1008`.
* Веб-GUI получает одноразовый `ui_token` (встроен в `index.html` при выдаче) — основной ключ браузеру не нужен.

---

## Синтез речи

### `POST /api/tts`

Основной эндпоинт. Отдаёт готовое аудио; при `stream=true` — чанками (`Transfer-Encoding: chunked`).

**Поля запроса** (все, кроме текста, необязательны):

| Поле | Тип | Описание |
|---|---|---|
| `text` **или** `input` | string | текст для озвучивания (непустой; `input` — синоним в OpenAI-стиле) |
| `voice` | string \| object \| string[] | id/имя профиля **или** имя встроенного тембра |
| `profile` | string | id профиля-клона (альтернатива `voice`) |
| `speaker` | string | встроенный тембр CustomVoice: `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee` |
| `instruct` / `instructions` | string | стиль/эмоция («Скажи это радостно») — для CustomVoice и VoiceDesign |
| `ref_audio` | string | эталон для клонирования «на лету»: путь, URL, base64 или `data:audio/wav;base64,...` |
| `ref_text` | string | транскрипт эталона (с ним клон точнее) |
| `x_vector_only_mode` | bool | клонировать только тембр, без транскрипта |
| `language` | string | `Auto` \| `Russian` \| `English` \| `Chinese` \| `Japanese` \| … (принимаются и коды: `ru`, `en-us`) |
| `model` | string | ключ модели из `config.json` (`models[].key`) |
| `task_type` | string | `clone` \| `preset` \| `design` (обычно определяется автоматически) |
| `format` / `response_format` | string | `wav` (по умолчанию) \| `pcm` \| `mp3` \| `ogg` \| `flac` \| `opus` \| `m4a` |
| `sample_rate` | int | 8000–96000; по умолчанию 24000 (нативная частота модели) |
| `speed` | float | 0.5–2.0 |
| `stream` | bool | `true` — отдавать аудио чанками |
| `seed` | int | для воспроизводимости |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_new_tokens`, `do_sample` | — | переопределение параметров генерации на один запрос |

**Пример**

```bash
curl -s http://127.0.0.1:8770/api/tts -H 'Content-Type: application/json' -d '{
  "text": "Привет! Это локальный синтез речи.",
  "voice": "mama",
  "language": "ru",
  "format": "wav",
  "stream": false
}' -o hello.wav -D headers.txt
```

**Ответ:** тело — аудио (`Content-Type: audio/wav` или `audio/L16;rate=24000;channels=1` для `pcm`),
плюс заголовки метрик:

```
X-Job-Id: 71c7f3a5cf41
X-Audio-Duration: 2.286
X-Sample-Rate: 24000
X-TTS-RTF: 0.251
X-TTS-TTFA-ms: 573.0
X-TTS-Elapsed-ms: 574.1
X-TTS-Chunks: 1
X-Request-ID: ded81a
Cache-Control: no-store
```

В режиме `stream=true` ответ — **единый аудиопоток** (чанки идут подряд, без дублей):
для `wav` сначала один RIFF-заголовок с размерами `0xFFFFFFFF` (стандартная конвенция
«бесконечного» стрима — ffmpeg/ffplay, VLC и libsndfile читают его до конца потока),
далее сырые PCM16-чанки; для `pcm` — сразу сырые PCM16-чанки. Склейка тела ответа —
один валидный аудиофайл; при сохранении на диск рекомендуется пропатчить размеры
RIFF/data реальными значениями (веб-GUI делает это автоматически). Заголовки метрик
относятся к началу потока; полные метрики — в `GET /api/jobs/{X-Job-Id}`.

### `POST /api/tts/audio`

То же, что `/api/tts`, но всегда потоком и без JSON-обёрток — удобно для `<audio>`-плеера:

```js
const res = await fetch('/api/tts/audio', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Длинный текст...', voice: 'mama', format: 'pcm' }),
});
// res.body — поток PCM 24 кГц 16 бит моно: можно скармливать AudioWorklet/WebAudio
```

### `POST /api/tts/json`

Аудио в base64 + метаданные (удобно для серверных интеграций и отладки).

```json
{
  "job_id": "2688a70c4935",
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "format": "wav",
  "mime": "audio/wav",
  "sample_rate": 24000,
  "audio_sec": 1.429,
  "metrics": { "elapsed_ms": 358.7, "ttfa_ms": 358.1, "rtf": 0.251, "chunks": 1 },
  "output_file": "/home/user/.local/share/qtts/outputs/job_2688a70c4935_20261105_075202_123.wav"
}
```

---

## OpenAI-совместимый эндпоинт

### `POST /v1/audio/speech`

Принимает поля OpenAI (`input`, `voice`, `response_format`, `speed`, `stream`, `model`)
и наши расширения (`language`, `instruct`/`instructions`, `ref_audio`, `ref_text`,
`x_vector_only_mode`, `task_type`, `sample_rate`).

```bash
curl -s http://127.0.0.1:8770/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "qwen3-tts",
  "input": "Hello from a local server!",
  "voice": "vivian",
  "response_format": "wav"
}' -o speech.wav
```

Клонирование через этот же эндпоинт:

```json
{
  "input": "Синтез голосом из эталона.",
  "task_type": "Base",
  "ref_audio": "data:audio/wav;base64,UklGRiQAAABXQVZF...",
  "ref_text": "Текст эталонной записи.",
  "language": "Russian"
}
```

Совместимость с SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8770/v1", api_key="не-важно")
resp = client.audio.speech.create(model="qwen3-tts", voice="vivian", input="Привет!")
resp.write_to_file("speech.wav")
```

---

## Задачи (async-режим)

### `POST /api/tts/async` → `202`

Ставит задачу в очередь и сразу возвращает id (генерация выполняется в фоне):

```json
{ "job_id": "84edd81815bf", "state": "queued", "status_url": "/api/jobs/84edd81815bf" }
```

### `GET /api/jobs?limit=50&state=done`

```json
{
  "jobs": [ { "id": "84edd81815bf", "state": "done", "progress": 1.0, "audio_sec": 2.43, "rtf": 0.251 } ],
  "stats": {
    "total": 7, "by_state": { "done": 7 }, "audio_sec_total": 13.5,
    "avg_rtf": 0.251, "avg_ttfa_ms": 483.5, "errors": 0
  }
}
```

### `GET /api/jobs/active`

`{ "active": [...], "count": N }` — задачи в состояниях `queued`/`running`/`streaming` (для индикаторов GUI).

### `GET /api/jobs/{id}?events=true`

Полный объект задачи (см. [Объекты данных](#объект-задачи)); `events=true` добавляет ленту этапов,
`with_text` — исходный текст.

### `GET /api/jobs/{id}/audio`

Скачивает сохранённый результат (`200`), если задача `done` и `audio.save_outputs=true`;
иначе `409` (аудио ещё нет) или `404` (файл не сохранён/удалён при ротации).

### `POST /api/jobs/{id}/cancel`

Выставляет флаг отмены: текущий чанк останавливается, задача переходит в `cancelled`.
Для задачи в очереди состояние меняется сразу.

### `DELETE /api/jobs`

Очищает историю задач → `{ "cleared": N }`.

---

## WebSocket `/ws/tts`

Подключение: `ws://127.0.0.1:8770/ws/tts` (при включённом ключе — `?api_key=<ключ>`).

**Клиент → сервер** (JSON):

```json
{ "text": "Раз. Два. Три.", "voice": "mama", "language": "Russian", "format": "pcm", "stream": true }
```

Поддерживаются все поля `POST /api/tts`. Отмена задачи: `{ "cancel": "<job_id>" }`.

**Сервер → клиент:**

| Сообщение | Когда |
|---|---|
| `{"event":"job","job_id":"...","chunks":null}` | задача принята |
| `{"event":"progress","type":"header","sample_rate":24000,"format":"wav"}` | при `format:"wav"` — перед первым бинарным кадром (RIFF-заголовок потока) |
| `{"event":"progress","type":"audio","index":0,"total":3,"audio_sec":1.2,"gen_ms":310,"tok_per_sec":41.2,"elapsed_ms":318,"text":"Раз."}` | готов очередной чанк |
| **бинарный кадр** | байты единого аудиопотока (сразу после соответствующего `progress`): для `pcm` — сырые PCM16-чанки; для `wav` — первый кадр RIFF-заголовок (размеры `0xFFFFFFFF`), далее PCM16-чанки |
| `{"event":"done","type":"final","audio_sec":3.1,"rtf":0.25,"ttfa_ms":318,"chunks":3,"output_file":"..."}` | синтез завершён (аудио НЕ дублируется — оно уже пришло бинарными кадрами) |
| `{"event":"error","error":"...","code":"bad_request\|bad_json\|internal\|..."}` | ошибка (соединение остаётся живым) |
| `{"event":"cancelled","job_id":"..."}` | отмена принята |

Пример на Python — в [README](../README.md#api). Для проигрывания в браузере удобно использовать
`format:"pcm"` и AudioWorklet (24 кГц, 16 бит, моно).

---

## Голоса и профили

### `GET /api/voices`

```json
{
  "presets": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"],
  "profiles": [ { "id": "mama", "name": "Мама", "kind": "clone", "duration_sec": 3.0 } ],
  "default_profile": "mama",
  "count": 10
}
```

### `GET /api/voices/presets` · `GET /api/voices/languages`

Список встроенных тембров (и признак, загружена ли модель) и поддерживаемых языков.

### `GET /api/profiles` · `GET /api/profiles/{id}`

Список и карточка профиля (см. [Объект профиля](#объект-профиля)).

### `POST /api/profiles`

Создание профиля. **Два формата тела** — выбирается по `Content-Type`:

* `multipart/form-data` (так делает GUI): поля `name`, `kind`, `ref_text`, `instruct`, `language`,
  `speaker`, `x_vector_only`, `tags` (строка через запятую или массив), `notes`, `model_key`,
  `profile_id` + файл в поле `file`;
* `application/json`: та же модель, но образец передаётся строкой `audio`
  (base64, `data:audio/wav;base64,...`, http(s)-URL или путь **на сервере**).

Ответ `200`:

```json
{
  "profile": { "id": "mama", "name": "Мама", "kind": "clone", "audio": { "duration_sec": 3.0, "sample_rate": 24000 } },
  "warnings": [], "stats": { "duration_sec": 3.0, "rms_db": -12.4, "peak": 0.31, "clipping": false, "silent": false }
}
```

Ошибки: `415` (непонятный `Content-Type`), `400` (битая форма/JSON), `422` (нет `name`, нет образца
для `clone`, нечитаемое аудио), `409` (id/имя уже занято, у `preset` нет `speaker`), `413` (файл больше
`server.max_upload_mb`).

### `PATCH /api/profiles/{id}` (`PUT` — синоним)

Частичное обновление: `name`, `ref_text`, `instruct`, `language`, `speaker`, `x_vector_only`,
`tags`, `notes`, `kind`, `audio` (замена эталона).

### `DELETE /api/profiles/{id}` → `{ "deleted": "mama" }`

### `POST /api/profiles/{id}/duplicate?name=Копия`

### `GET /api/profiles/{id}/audio`

Эталонное аудио профиля: WAV 24 кГц, моно, 16 бит.

### `POST /api/profiles/{id}/test`

Тестовый синтез голосом профиля. Тело (необязательное): `{"text": "...", "language": "..."}`.
Ответ — WAV + заголовки `X-Job-Id`, `X-Audio-Duration`.

### `POST /api/profiles/analyze`

Разобрать аудио **без сохранения** (проверка эталона перед созданием профиля).
Тело — multipart с полем `file`.

```json
{
  "filename": "ref.wav",
  "stats": { "duration_sec": 3.0, "sample_rate": 24000, "samples": 72000,
             "rms": 0.21, "rms_db": -13.5, "peak": 0.31, "clipping": false, "silent": false },
  "warnings": ["короткий эталон (1.4 с), нужно ≥ 2 с"],
  "target_sr": 24000
}
```

---

## Система и диагностика ROCm

### `GET /api/health` (публичный)

```json
{ "status": "ok", "time": 1793864060.1, "uptime_sec": 234.0, "engine": "qwen_tts",
  "model_loaded": true, "active_model": "model", "device": "cuda:0" }
```

### `GET /api/system/info`

Версии пакетов (`torch`, `qwen_tts`, `transformers`, `fastapi`, …), текущий конфиг (api_key маскируется),
путь к конфигам, каталоги данных, состояние движка, значимые переменные окружения ROCm,
статистика задач, число профилей, список доступных движков.

### `GET /api/system/status`

Состояние движка и очереди: `backend`, `ready`, `loaded_models`, `device`, `dtype`, `attn`, `gfx`,
`vram_*`, `uptime_sec`, счётчики, `profiles`, `jobs`.

### `GET /api/system/diagnostics?refresh=false&bench=true`

Полная проверка ROCm/GPU. В ответе — массив `checks` (`name`, `status` = `ok|warn|error`, `message`, `hint`)
и сводный `health`:

| Проверка | Что смотрит |
|---|---|
| `ОС / драйвер amdgpu` | загружен ли модуль ядра `amdgpu` |
| `ROCm runtime` | наличие `/opt/rocm`, `rocm-smi`, `rocminfo`, версия ROCm |
| `Права на GPU` | `/dev/kfd`, `/dev/dri/renderD*`, группы `render`/`video` |
| `PyTorch` | импортируется ли, версия, `torch.version.hip`, пакеты `qwen-tts`/`transformers` |
| `Видимость GPU` | `torch.cuda.is_available()`, имя карты, **gfx**, VRAM |
| `Поддержка gfx в сборке torch` | есть ли ваша архитектура в `torch.cuda.get_arch_list()` |
| `HSA_OVERRIDE_GFX_VERSION` | не подменена ли архитектура без необходимости |
| `SDPA backends` | какие ядра attention реально работают (flash / mem-efficient / math) |
| `Пакет qwen-tts`, `Версия transformers` | установлены ли и тех ли версий |
| `Бенчмарк GPU` (`bench=true`) | микро-тест матричного умножения на GPU |
| `VRAM под модель 1.7B` | хватит ли памяти для bf16-весов |
| `ffmpeg` | доступен ли для `mp3`/`ogg`/`flac`/`opus` |

### `GET /api/system/diagnostics/text`

Тот же отчёт текстом (удобно копировать в issue).

### `GET /api/system/readiness`

```json
{ "ready": true, "gpu_visible": true, "engine": "qwen_tts", "model_loaded": true,
  "active_model": "model", "device": "cuda:0", "rocm_version": "10.0", "gpu_name": "AMD Radeon RX 9060 XT",
  "gfx": "gfx1200", "problems": [], "health": "ok" }
```

Это то, что показывает GUI в ответ на «проверь, видна ли карта через ROCm».

### `GET /api/system/gpu`

Быстрый снимок: VRAM (allocated/reserved/total), загрузка, температура/частоты (если доступны).

### `POST /api/system/test`

Кнопка «Проверить режим работы»: сначала диагностика, затем живой синтез тестовой фразы.

```json
{ "diagnostics": { "...": "..." }, "diagnostics_ms": 12.4,
  "test": { "ok": true, "elapsed_ms": 573.0, "audio_bytes": 109758, "audio_sec": 2.29, "rtf": 0.251,
            "job": { "id": "...", "state": "done" } },
  "test_ms": 590.1, "ok": true }
```

### `GET /api/system/env` · `POST /api/system/env`

Просмотр значимых переменных окружения и применение пресета/своих значений:

```json
{ "preset": "rdna4_rocm10", "env": { "AMD_LOG_LEVEL": "0" }, "overwrite": false }
```

> Переменные ROCm читаются рантаймом **один раз при инициализации**, поэтому применять их нужно
> до запуска процесса (`device.env` в конфиге, `--env K=V`, `--preset`, переменные в systemd-юните).
> Эндпоинт полезен для проверки и для последующего перезапуска движка.

---

## Модели

| Эндпоинт | Описание |
|---|---|
| `GET /api/models` | список моделей из конфига + признак загрузки + метаданные активной |
| `POST /api/models/load` | `{"model_key":"model","force":false}` — загрузить веса в VRAM |
| `POST /api/models/unload` | выгрузить (освобождает VRAM) |
| `POST /api/models/add` | `{"key":"small","source":"Qwen/Qwen3-TTS-12Hz-0.6B-Base","kind":"auto","label":"0.6B"}` |
| `DELETE /api/models/{key}` | убрать модель из конфига (если это была последняя — подставляется модель по умолчанию) |

`source` — repo id HuggingFace **или** путь к локальному каталогу с весами.

---

## Настройки

### `GET /api/config`

Текущий конфиг целиком (`server.api_key` маскируется как `***`).

### `PATCH /api/config`

Применяется сразу и пишется в тот же `config.json`, откуда конфиг загружен.
Тело — частичный патч по секциям:

```json
{ "generation": { "chunk_target_chars": 180, "temperature": 0.8 },
  "device":     { "dtype": "bfloat16", "attn": "sdpa" },
  "log":        { "level": "DEBUG" } }
```

Ответ:

```json
{ "ok": true, "restart_required": ["model"], "config": { "...": "полный конфиг" } }
```

`restart_required` — что нужно пересоздать: `backend` (смена движка) и/или `model`
(смена модели, `dtype`, `attn`, `device`). Перезагрузка без перезапуска сервера:
`POST /api/models/load {"force": true}`.

### `POST /api/config/save` · `POST /api/config/reload`

Сохранить текущее состояние в файл / перечитать файл с диска.

### `POST /api/backend`

`{"backend": "qwen_tts" | "vllm_omni" | "dummy"}` — переключить движок и перезагрузить модель.

---

## Журнал

### `GET /api/logs?limit=200&level=ERROR`

```json
{ "logs": [ { "ts": 1793864060.129, "time": "07:34:20", "level": "REQ", "name": "http",
              "msg": "→ POST /api/tts  client=127.0.0.1  ua=curl/8.5.0", "rid": "ded81a" } ],
  "counters": { "INFO": 12, "REQ": 34, "GEN": 6, "GPU": 2, "WARNING": 1 } }
```

Уровни: стандартные `DEBUG/INFO/WARNING/ERROR` + специальные `REQ` (запросы), `GEN` (этапы генерации),
`GPU` (устройства и диагностика).

### `GET /api/logs/stream` (SSE)

Живой поток для вкладки «Журнал»:

```
retry: 2000
event: hello
data: {"ts": 1793864060.1}

data: {"ts":..., "time":"07:34:20", "level":"REQ", "name":"http", "msg":"→ POST /api/tts ...", "rid":"ded81a"}

: keep-alive
```

Подписка в браузере: `new EventSource('/api/logs/stream?level=ERROR&api_key=...')`.
Событие `hello` не содержит поля `level` — учитывайте это в обработчике.

### `GET /api/logs/file`

Скачивает файл журнала (`404`, если файловый журнал выключен).

---

## Результаты синтеза

### `GET /api/outputs?limit=100`

```json
{ "dir": "/home/user/.local/share/qtts/outputs",
  "files": [ { "name": "job_71c7f3a5cf41_20261105_075201_123.wav", "size": 109758,
               "mtime": 1793864061.0, "url": "/api/outputs/job_71c7f3a5cf41_20261105_075201_123.wav" } ] }
```

### `GET /api/outputs/{name}`

Отдаёт файл. Выход за пределы каталога (`../`) блокируется → `404`.
Ротация: хранится не больше `audio.keep_outputs` файлов.

---

## Объекты данных

### Объект задачи

```json
{
  "id": "84edd81815bf", "state": "done", "stage": "done", "progress": 1.0,
  "created_at": 1793864060.0, "started_at": 1793864060.1, "finished_at": 1793864061.0,
  "elapsed_ms": 990.2, "client": "127.0.0.1", "endpoint": "/api/tts/async",
  "model": "model", "voice": "clone:mama", "language": "Russian", "format": "wav",
  "chunks_done": 1, "chunks_total": 1, "audio_sec": 2.43, "bytes_out": 116780,
  "sample_rate": 24000, "ttfa_ms": 608.8, "gen_ms": 610.0, "rtf": 0.251, "tok_per_sec": 41.2,
  "error": null, "error_code": null,
  "output_file": "/home/user/.local/share/qtts/outputs/job_84edd81815bf_20261105_075203_456.wav",
  "events": [ { "ts": 1793864060.1, "stage": "running" }, { "ts": 1793864061.0, "stage": "done" } ]
}
```

Состояния: `queued` → `running` → `streaming` → `done` | `error` | `cancelled`.

### Объект профиля

```json
{
  "id": "mama", "name": "Мама", "kind": "clone",
  "speaker": null, "instruct": null, "language": "Russian",
  "ref_text": "Привет, это мой эталонный голос.", "x_vector_only": false,
  "audio_file": "ref.wav", "tags": ["семья"], "notes": "", "model_key": null,
  "created_at": 1793864000.0, "updated_at": 1793864000.0,
  "audio": { "duration_sec": 3.0, "sample_rate": 24000, "samples": 72000, "rms": 0.21,
             "rms_db": -13.5, "peak": 0.31, "clipping": false, "silent": false,
             "size_bytes": 144044, "path": "/home/user/.local/share/qtts/profiles/mama/ref.wav" },
  "warnings": [], "url": "/api/profiles/mama/audio"
}
```

`kind`: `clone` (нужен образец), `preset` (встроенный тембр, нужен `speaker`), `designed` (голос по описанию).

---

## Ошибки

Формат тела ошибки:

```json
{ "error": "очередь переполнена (32/32) — повторите позже", "code": "queue_full" }
```

или (для ошибок валидации запроса):

```json
{ "error": "некорректный запрос", "details": [ { "field": "body.text", "msg": "...", "type": "value_error" } ] }
```

| HTTP | `code` | Смысл |
|---|---|---|
| 400 | — | не удалось разобрать форму/JSON |
| 401 | — | нет или неверен API-ключ |
| 404 | — | задача/профиль/модель/файл не найдены |
| 409 | `profile_error` | имя/id профиля занято; аудио задачи ещё нет |
| 413 | — | файл больше `server.max_upload_mb` |
| 415 | — | неподдерживаемый `Content-Type` |
| 422 | `empty_text`, `no_reference`, `no_speaker`, `no_instruct`, `bad_audio` | ошибка в данных запроса |
| 429 | `queue_full` | очередь переполнена |
| 499 | `cancelled` | отменено клиентом |
| 500 | `internal` | непредвиденная ошибка (детали — в журнале по `X-Request-ID`) |
| 501 | — | формат требует ffmpeg |
| 502 | `vllm_unreachable`, `vllm_error` | проблема с внешним vLLM-Omni |
| 503 | `no_torch`, `no_qwen_tts`, `gpu_unavailable`, `model_load_failed`, `queue_timeout` | движок не готов |

Для отладки: возьмите `X-Request-ID` из ответа и найдите его в `GET /api/logs` или в журнале сервера —
по нему видны все строки этого запроса, включая этапы генерации.
