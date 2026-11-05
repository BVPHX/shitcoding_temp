"""Аудио-слой: разбор источников, нормализация эталона, кодирование результата."""

from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from qtts.app import audio as A
from tests.conftest import wav_bytes


def tone(seconds: float = 1.0, sr: int = A.NATIVE_SR, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# resolve_audio_source
# --------------------------------------------------------------------------- #


def test_resolve_bytes():
    wav, sr, origin = A.resolve_audio_source(wav_bytes(2.0))
    assert origin == "bytes"
    assert sr == A.NATIVE_SR
    assert 1.9 <= len(wav) / sr <= 2.1


def test_resolve_ndarray_and_tuple():
    wav, sr, origin = A.resolve_audio_source(tone(0.5))
    assert origin == "ndarray" and sr == A.NATIVE_SR
    wav2, sr2, origin2 = A.resolve_audio_source((tone(0.5, sr=16000), 16000))
    assert origin2 == "ndarray" and sr2 == 16000


def test_resolve_base64_and_data_uri():
    raw = wav_bytes(2.0)
    b64 = base64.b64encode(raw).decode()
    _, _, origin = A.resolve_audio_source(b64)
    assert origin == "base64"
    _, _, origin2 = A.resolve_audio_source("data:audio/wav;base64," + b64)
    assert origin2 == "base64"


def test_resolve_base64_with_slashes_and_plus():
    """base64 неизбежно содержит '/' и '+' — важно не принять его за путь к файлу."""
    wav = tone(3.0)
    raw = A.encode_wav(wav, A.NATIVE_SR)
    b64 = base64.b64encode(raw).decode()
    assert "/" in b64 or "+" in b64, "тест должен работать на «неудобной» base64-строке"
    out, sr, origin = A.resolve_audio_source(b64)
    assert origin == "base64" and sr == A.NATIVE_SR and len(out) > 0


def test_resolve_file_path(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(wav_bytes(2.0))
    wav, sr, origin = A.resolve_audio_source(str(p))
    assert origin == f"file:{p}"
    assert len(wav) > 0


def test_resolve_missing_path_raises_audio_error():
    with pytest.raises(A.AudioError):
        A.resolve_audio_source("/tmp/qtts-does-not-exist-12345.wav")


def test_resolve_garbage_long_string_raises_audio_error_not_oserror():
    with pytest.raises(A.AudioError):
        A.resolve_audio_source("A" * 5000)          # ENAMETOOLONG не должен протекать наружу


def test_resolve_bad_url_raises_audio_error():
    with pytest.raises(A.AudioError):
        A.resolve_audio_source("http://127.0.0.1:1/nope.wav")


def test_resolve_unsupported_type():
    with pytest.raises(A.AudioError):
        A.resolve_audio_source(12345)


# --------------------------------------------------------------------------- #
# Обработка
# --------------------------------------------------------------------------- #


def test_to_mono_downmixes_stereo():
    stereo = np.stack([tone(0.2), tone(0.2, freq=880.0)], axis=1)
    mono = A.to_mono(stereo)
    assert mono.ndim == 1 and len(mono) == stereo.shape[0]


def test_resample_changes_length():
    wav = tone(1.0, sr=48000)
    out = A.resample(wav, 48000, 24000)
    assert abs(len(out) - 24000) <= 200


def test_peak_normalize_clamps_to_target():
    loud = tone(0.2) * 3.0
    out = A.peak_normalize(loud, peak=0.9)
    assert float(np.max(np.abs(out))) <= 0.9 + 1e-5


def test_trim_silence_removes_padding():
    padded = np.concatenate([np.zeros(4800, dtype=np.float32), tone(0.5), np.zeros(4800, dtype=np.float32)])
    out = A.trim_silence(padded, A.NATIVE_SR)
    assert len(out) < len(padded)


def test_prepare_reference_limits_duration_and_normalizes():
    long_wav = tone(40.0, sr=16000)
    out, sr = A.prepare_reference(long_wav, 16000, max_sec=10.0)
    assert sr == A.NATIVE_SR
    assert len(out) / sr <= 10.05
    assert float(np.max(np.abs(out))) <= 0.98


def test_audio_stats_flags_silence_and_clipping():
    silent = np.zeros(2400, dtype=np.float32)
    assert A.audio_stats(silent, A.NATIVE_SR)["silent"] is True
    clipped = np.full(2400, 1.0, dtype=np.float32)
    st = A.audio_stats(clipped, A.NATIVE_SR)
    assert st["clipping"] is True and st["duration_sec"] == pytest.approx(0.1, abs=0.01)


def test_concat_with_gap_inserts_silence():
    a, b = tone(0.1), tone(0.1)
    plain = A.concat([a, b], gap_ms=0)
    gapped = A.concat([a, b], gap_ms=100)
    assert len(gapped) - len(plain) == A.NATIVE_SR // 10
    assert abs(float(np.abs(gapped[len(a): len(a) + 240]).max())) < 1e-6


# --------------------------------------------------------------------------- #
# Кодирование
# --------------------------------------------------------------------------- #


def test_encode_pcm_is_raw_16bit():
    data = A.encode(tone(0.5), A.NATIVE_SR, "pcm")
    assert len(data) == int(0.5 * A.NATIVE_SR) * 2
    assert not data.startswith(b"RIFF")


def test_encode_wav_roundtrip():
    data = A.encode(tone(0.5), A.NATIVE_SR, "wav")
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == A.NATIVE_SR
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert abs(w.getnframes() - A.NATIVE_SR * 0.5) < 10


def test_wav_header_matches_encoder():
    wav = tone(0.25)
    body = A.float_to_pcm16(wav)
    hdr = A.wav_header(len(body), A.NATIVE_SR)
    with wave.open(io.BytesIO(hdr + body), "rb") as w:
        assert w.getnframes() == len(body) // 2


def test_encode_ffmpeg_formats_skipped_or_work():
    """mp3/ogg требуют ffmpeg: тест не должен падать, если его нет."""
    if not A.ffmpeg_available():
        pytest.skip("ffmpeg не установлен")
    data = A.encode(tone(0.3), A.NATIVE_SR, "mp3")
    assert len(data) > 0 and not data.startswith(b"RIFF")


def test_save_output_and_cleanup(tmp_path):
    d = tmp_path / "outputs"
    paths = []
    for i in range(5):
        p = A.save_output(A.encode(tone(0.1), A.NATIVE_SR, "wav"), d, "wav", prefix=f"job_{i}")
        assert p and p.exists()
        paths.append(p)
    removed = A.cleanup_outputs(d, keep=2)
    assert removed == 3
    assert len(list(d.glob("*.wav"))) == 2
