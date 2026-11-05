"""Работа с аудио: чтение/нормализация эталонов, кодирование результата синтеза.

Форматы ответа:
  wav  — стандартный RIFF PCM16 (пишем сами, без внешних зависимостей);
  pcm  — сырой signed 16-bit little-endian mono (для стриминга);
  mp3 / ogg / flac / opus — через ffmpeg (если установлен).
"""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
import socket
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .logging_setup import get_logger

log = get_logger("audio")

NATIVE_SR = 24000  # выходная частота Qwen3-TTS-Tokenizer-12Hz
FFMPEG_MAP = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
    "ogg": ["-c:a", "libvorbis", "-q:a", "6"],
    "opus": ["-c:a", "libopus", "-b:a", "96k"],
    "flac": ["-c:a", "flac"],
    "wav": ["-c:a", "pcm_s16le"],
    "m4a": ["-c:a", "aac", "-b:a", "192k"],
}
MIME = {
    "wav": "audio/wav",
    "pcm": "audio/L16;rate=24000;channels=1",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
}


class AudioError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# --------------------------------------------------------------------------- #
# Чтение
# --------------------------------------------------------------------------- #


def _decode_with_soundfile(data: bytes) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def load_audio_bytes(data: bytes, target_sr: int = NATIVE_SR) -> Tuple[np.ndarray, int]:
    """Декодировать произвольный аудио-формат из байт (wav/flac/ogg/mp3/webm...)."""
    if not data:
        raise AudioError("пустые аудиоданные")
    # сначала soundfile (wav/flac/ogg), затем ffmpeg (mp3/webm/m4a/opus)
    try:
        wav, sr = _decode_with_soundfile(data)
    except Exception as exc:  # noqa: BLE001
        wav = None
        first_err = exc
        if not ffmpeg_available():
            raise AudioError(
                f"не удалось прочитать аудио ({type(first_err).__name__}). "
                "Для mp3/webm/m4a установите ffmpeg: sudo apt install ffmpeg"
            ) from first_err
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr), "pipe:1"],
            input=data, capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            raise AudioError(
                f"ffmpeg не смог декодировать аудио: {proc.stderr.decode(errors='ignore')[:300]}"
            ) from first_err
        wav, sr = _decode_with_soundfile(proc.stdout)
    return to_mono(wav), sr


def load_audio_file(path: str | Path, target_sr: Optional[int] = NATIVE_SR) -> Tuple[np.ndarray, int]:
    p = Path(path)
    if not p.exists():
        raise AudioError(f"файл не найден: {p}")
    wav, sr = load_audio_bytes(p.read_bytes(), target_sr=target_sr or NATIVE_SR)
    if target_sr and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def maybe_base64(text: str) -> Optional[bytes]:
    """Если строка похожа на base64 (или data-URI) — декодировать, иначе None.

    Длинная base64-строка неизбежно содержит '/' и '+', поэтому отличаем её от пути
    по длине и по составу символов: в base64 не бывает пробелов и '~'.
    """
    t = (text or "").strip()
    if not t:
        return None
    if t.lower().startswith("data:"):
        try:
            t = t.split(",", 1)[1]
        except IndexError:
            return None
    elif t.lower().startswith(("http://", "https://", "file://")):
        return None
    t_clean = re.sub(r"\s+", "", t)
    if len(t_clean) < 64:
        return None
    # короткий путь вроде "/home/user/ref.wav" base64-алфавиту не соответствует ('.'),
    # а вот длинный base64 — соответствует всегда
    if len(t_clean) <= 255 and not re.fullmatch(r"[A-Za-z0-9+/=_\-]+", t_clean):
        return None
    if len(t_clean) > 255 and re.search(r"[\s~]", t):
        return None
    try:
        return base64.b64decode(t_clean, validate=False)
    except Exception:  # noqa: BLE001
        return None


def _fetch_url(url: str, timeout: float = 30.0) -> bytes:
    """Скачать аудио по http(s)/file URL. Все сетевые ошибки → AudioError."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qtts/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise AudioError(f"URL вернул HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        raise AudioError(f"не удалось скачать аудио по URL {url}: {exc}") from exc


def resolve_audio_source(src: Any, target_sr: int = NATIVE_SR) -> Tuple[np.ndarray, int, str]:
    """Принимает: путь / URL / base64 / data-URI / (ndarray, sr) / ndarray / bytes.

    Возвращает (wav, sr, описание_источника). Любая проблема → AudioError
    (а не «сырое» исключение): ошибки здесь возвращаются клиенту как 422.
    """
    if isinstance(src, tuple) and len(src) == 2 and isinstance(src[0], np.ndarray):
        return to_mono(src[0]), int(src[1]), "ndarray"
    if isinstance(src, np.ndarray):
        return to_mono(src), target_sr, "ndarray"
    if isinstance(src, (bytes, bytearray)):
        try:
            wav, sr = load_audio_bytes(bytes(src), target_sr)
        except AudioError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"не удалось разобрать переданное аудио: {exc}") from exc
        if target_sr and sr != target_sr:
            wav = resample(wav, sr, target_sr)
            sr = target_sr
        return wav, sr, "bytes"
    if not isinstance(src, str):
        raise AudioError(f"неподдерживаемый тип источника аудио: {type(src).__name__}")

    s = src.strip()
    if not s:
        raise AudioError("пустой источник аудио")
    if s.lower().startswith(("http://", "https://", "file://")):
        data = _fetch_url(s)
        try:
            wav, sr = load_audio_bytes(data, target_sr)
        except AudioError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"скачанные данные не являются аудио: {exc}") from exc
        return wav, sr, f"url:{s[:80]}"

    raw = maybe_base64(s)
    if raw is not None:
        try:
            wav, sr = load_audio_bytes(raw, target_sr)
        except AudioError as exc:
            # декодировалось, но это не аудио — объясняем подробнее, чем «LibsndfileError»
            raise AudioError(
                f"base64-данные не являются поддерживаемым аудио ({exc}). "
                "Ожидается wav/mp3/ogg/flac; для mp3/webm на сервере нужен ffmpeg"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"base64-данные не являются аудио: {exc}") from exc
        return wav, sr, "base64"

    try:
        p = Path(s).expanduser()
        exists = p.exists()
    except OSError as exc:
        # слишком длинная строка (ENAMETOOLONG), битая кодировка пути и т.п.
        raise AudioError(
            f"не удалось разобрать источник аудио ({exc.__class__.__name__}). "
            "Ожидается путь к файлу, URL или base64/data-URI"
        ) from exc
    if not exists:
        raise AudioError(f"источник аудио не найден: {s[:120]}")
    wav, sr = load_audio_file(p, target_sr)
    return wav, sr, f"file:{p}"


# --------------------------------------------------------------------------- #
# Обработка
# --------------------------------------------------------------------------- #


def to_mono(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1) if wav.shape[1] <= wav.shape[0] else wav.mean(axis=0)
    return np.ascontiguousarray(wav.reshape(-1), dtype=np.float32)


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    wav = to_mono(wav)
    try:
        import librosa

        return np.asarray(librosa.resample(wav, orig_sr=int(sr), target_sr=int(target_sr)), dtype=np.float32)
    except ImportError:
        pass
    try:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(wav).unsqueeze(0)
        out = AF.resample(t, int(sr), int(target_sr))
        return out.squeeze(0).numpy().astype(np.float32)
    except ImportError:
        # грубый, но рабочий fallback — линейная интерполяция
        ratio = target_sr / float(sr)
        n_out = int(round(len(wav) * ratio))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        log.warning("resample fallback (линейная интерполяция): установите librosa или torchaudio")
        return np.interp(x_new, x_old, wav).astype(np.float32)


def peak_normalize(wav: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if m <= 1e-8 or peak <= 0:
        return wav
    return (wav * (peak / m)).astype(np.float32)


def _numpy_trim_silence(wav: np.ndarray, sr: int, top_db: float = 40.0,
                        frame_ms: float = 10.0) -> np.ndarray:
    """Обрезка тишины по краям без librosa: RMS по окнам 10 мс против пикового уровня."""
    wav = to_mono(wav)
    n = len(wav)
    if n < 8 or sr <= 0:
        return wav
    frame = max(64, int(sr * frame_ms / 1000.0))
    if n < frame * 2:
        return wav
    peak = float(np.max(np.abs(wav))) or 1.0
    threshold = peak * (10.0 ** (-abs(float(top_db)) / 20.0))
    rms = np.sqrt(np.mean(np.square(wav[: (n // frame) * frame].reshape(-1, frame)), axis=1))
    active = np.nonzero(rms > threshold)[0]
    if active.size == 0:
        return wav
    start = int(active[0]) * frame
    end = min(n, int(active[-1] + 1) * frame)
    # небольшой защитный отступ, чтобы не «съесть» первый согласный
    pad = int(sr * 0.01)
    return np.ascontiguousarray(wav[max(0, start - pad): min(n, end + pad)], dtype=np.float32)


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 40.0) -> np.ndarray:
    """Убрать тишину по краям записи (librosa, если есть; иначе свой numpy-вариант)."""
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(np.asarray(wav, dtype=np.float32), top_db=top_db)
        return np.ascontiguousarray(trimmed, dtype=np.float32)
    except ImportError:
        return _numpy_trim_silence(wav, sr, top_db=top_db)
    except Exception as exc:  # noqa: BLE001
        log.debug("trim_silence: librosa не справилась (%s) — пробую numpy-вариант", exc)
        return _numpy_trim_silence(wav, sr, top_db=top_db)


def prepare_reference(
    wav: np.ndarray,
    sr: int,
    target_sr: int = NATIVE_SR,
    max_sec: float = 30.0,
    trim: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Нормализация эталона голоса: моно, 24 кГц, обрезка тишины и длины."""
    wav = to_mono(wav)
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    if trim:
        wav = trim_silence(wav, sr)
    if normalize:
        wav = peak_normalize(wav, 0.97)
    limit = int(max_sec * sr)
    if len(wav) > limit:
        wav = wav[:limit]
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def audio_stats(wav: np.ndarray, sr: int) -> Dict[str, Any]:
    wav = to_mono(wav)
    n = len(wav)
    rms = float(np.sqrt(np.mean(np.square(wav)))) if n else 0.0
    peak = float(np.max(np.abs(wav))) if n else 0.0
    return {
        "duration_sec": round(n / float(sr), 3) if sr else 0.0,
        "sample_rate": int(sr),
        "samples": int(n),
        "rms": round(rms, 5),
        "rms_db": round(20 * np.log10(rms + 1e-9), 2),
        "peak": round(peak, 5),
        "clipping": bool(peak > 0.999),
        "silent": bool(rms < 1e-4),
    }


def add_silence(ms: int, sr: int = NATIVE_SR) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)


def concat(chunks: List[np.ndarray], gap_ms: int = 0, sr: int = NATIVE_SR) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if gap_ms <= 0:
        return np.concatenate([to_mono(c) for c in chunks]).astype(np.float32)
    gap = add_silence(gap_ms, sr)
    out: List[np.ndarray] = []
    for i, c in enumerate(chunks):
        if i:
            out.append(gap)
        out.append(to_mono(c))
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# Кодирование
# --------------------------------------------------------------------------- #


def float_to_pcm16(wav: np.ndarray) -> bytes:
    w = np.clip(to_mono(wav), -1.0, 1.0)
    return (w * 32767.0).astype("<i2").tobytes()


def encode_wav(wav: np.ndarray, sr: int, sample_width: int = 2) -> bytes:
    pcm = float_to_pcm16(wav)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(int(sr))
        wf.writeframes(pcm)
    return buf.getvalue()


def encode_ffmpeg(wav: np.ndarray, sr: int, fmt: str) -> bytes:
    if not ffmpeg_available():
        raise AudioError(
            f"формат '{fmt}' требует ffmpeg. Установите его (sudo apt install ffmpeg) "
            "или используйте response_format=wav|pcm"
        )
    codec = FFMPEG_MAP.get(fmt)
    if codec is None:
        raise AudioError(f"неподдерживаемый формат: {fmt}")
    pcm = float_to_pcm16(wav)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "s16le", "-ar", str(int(sr)), "-ac", "1", "-i", "pipe:0", *codec, "-f", fmt if fmt != "m4a" else "mp4", "pipe:1"],
        input=pcm, capture_output=True, timeout=300,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg error: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout


def encode(wav: np.ndarray, sr: int, fmt: str = "wav", normalize: bool = True, peak: float = 0.97) -> bytes:
    fmt = (fmt or "wav").lower().lstrip(".")
    if fmt in ("raw", "pcm", "pcm16", "l16"):
        return float_to_pcm16(wav)
    if normalize:
        wav = peak_normalize(wav, peak)
    if fmt == "wav":
        return encode_wav(wav, sr)
    return encode_ffmpeg(wav, sr, fmt)


def wav_header(data_bytes: int, sr: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Заголовок RIFF для стриминга PCM (длина известна заранее)."""
    import struct

    byte_rate = sr * channels * sample_width
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_bytes, b"WAVE", b"fmt ", 16, 1, channels,
        sr, byte_rate, channels * sample_width, sample_width * 8, b"data", data_bytes,
    )


def wav_stream_header(sr: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Заголовок RIFF для потокового WAV, когда итоговая длина заранее неизвестна.

    Размеры RIFF и data выставляются в 0xFFFFFFFF — общепринятая конвенция
    «бесконечного» стрима: ffmpeg/ffplay, VLC, libsndfile (soundfile) и браузеры
    читают такой файл до конца потока, а клиент при сохранении может пропатчить
    реальные размеры (GUI делает это автоматически).
    """
    import struct

    byte_rate = sr * channels * sample_width
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE", b"fmt ", 16, 1, channels,
        sr, byte_rate, channels * sample_width, sample_width * 8, b"data", 0xFFFFFFFF,
    )


def save_output(data: bytes, directory: Path, fmt: str, prefix: str = "tts") -> Optional[Path]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        ts = time_str()
        p = directory / f"{prefix}_{ts}.{fmt}"
        p.write_bytes(data)
        return p
    except Exception as exc:  # noqa: BLE001
        log.warning("не удалось сохранить вывод: %s", exc)
        return None


def time_str() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def cleanup_outputs(directory: Path, keep: int) -> int:
    """Удалить самые старые файлы, оставив `keep` последних. Возвращает число удалённых."""
    if keep <= 0 or not directory.exists():
        return 0
    files = sorted(directory.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for f in files[keep:]:
        try:
            f.unlink()
            removed += 1
        except Exception:  # noqa: BLE001
            pass
    if removed:
        log.debug("очистка outputs: удалено %d файл(ов)", removed)
    return removed


def db64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def write_wav_file(path: str | Path, wav: np.ndarray, sr: int) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(encode_wav(wav, sr))
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_size(p: str | Path) -> int:
    try:
        return os.path.getsize(p)
    except OSError:
        return 0
