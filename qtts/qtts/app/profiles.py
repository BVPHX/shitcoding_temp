"""Профили голосов (образцы для клонирования).

Профиль = запись в `data/profiles/profiles.json` + каталог с эталонным аудио:

    data/profiles/<slug>/ref.wav      — нормализованный эталон (моно, 24 кГц)
    data/profiles/<slug>/meta.json    — копия метаданных (для переносимости)

Профиль может быть одного из типов:
  * clone        — клонирование голоса (нужна модель *-Base): ref.wav + ref_text;
  * preset       — встроенный тембр CustomVoice (speaker + instruct);
  * designed     — голос, созданный по описанию (VoiceDesign) и сохранённый как эталон.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import audio as audio_mod
from .logging_setup import get_logger

log = get_logger("profiles")


class ProfileError(RuntimeError):
    pass


@dataclass
class Profile:
    id: str
    name: str
    kind: str = "clone"                       # clone | preset | designed
    speaker: Optional[str] = None             # для preset
    instruct: Optional[str] = None            # стиль/эмоция (preset и designed)
    language: Optional[str] = None            # язык эталона (подсказка модели)
    ref_text: Optional[str] = None            # транскрипт эталона (обязателен для ICL-клона)
    x_vector_only: bool = False               # True = только speaker embedding (без транскрипта)
    audio_file: Optional[str] = None          # относительный путь внутри каталога профиля
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    model_key: Optional[str] = None           # к какой модели привязан (None = любая)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Profile":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})


_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "і": "i", "ї": "i", "є": "e", "ґ": "g",
}


def slugify(name: str, fallback: str = "voice") -> str:
    """ASCII-slug для id профиля: удобен в URL и путях (кириллица транслитерируется).

    Порядок важен:
      1. сначала таблица транслитерации по исходной строке — иначе NFKD разложит
         «ы» → «и» + бреве и «й» → «и» + бреве, и они превратятся в «i»;
      2. затем NFKD + удаление диакритик для прочих языков («é» → «e», «ñ» → «n»);
      3. если ASCII не осталось (например, имя из иероглифов) — fallback + хэш.
    """
    import hashlib

    s = unicodedata.normalize("NFC", str(name)).strip().lower()
    s = "".join(_TRANSLIT.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s_-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")[:48]
    if not s:
        s = f"{fallback}-{hashlib.sha1(str(name).encode()).hexdigest()[:6]}"
    return s


class ProfileStore:
    def __init__(self, root: Path, min_sec: float = 2.0, max_sec: float = 30.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "profiles.json"
        self.min_sec = min_sec
        self.max_sec = max_sec
        self._lock = threading.RLock()
        self._data: Dict[str, Profile] = {}
        self.load()

    # ------------------------------- хранение ------------------------------- #

    def load(self) -> None:
        with self._lock:
            self._data = {}
            if not self.index.exists():
                return
            try:
                raw = json.loads(self.index.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log.error("не удалось прочитать %s: %s", self.index, exc)
                return
            items = raw.get("profiles", raw if isinstance(raw, list) else [])
            for item in items:
                try:
                    p = Profile.from_dict(item)
                    self._data[p.id] = p
                except Exception as exc:  # noqa: BLE001
                    log.warning("пропуск повреждённого профиля %r: %s", item, exc)
            log.debug("загружено профилей: %d", len(self._data))

    def _flush(self) -> None:
        tmp = self.index.with_suffix(".tmp")
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "profiles": [p.to_dict() for p in sorted(self._data.values(), key=lambda x: x.created_at)],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index)

    def profile_dir(self, pid: str) -> Path:
        return self.root / pid

    # -------------------------------- выборка -------------------------------- #

    def list(self) -> List[Profile]:
        with self._lock:
            return sorted(self._data.values(), key=lambda p: p.created_at)

    def get(self, pid: str) -> Profile:
        with self._lock:
            if pid not in self._data:
                raise ProfileError(f"профиль '{pid}' не найден")
            return self._data[pid]

    def find_by_name(self, name: str) -> Optional[Profile]:
        key = str(name).strip().lower()
        with self._lock:
            for p in self._data.values():
                if p.id.lower() == key or p.name.lower() == key:
                    return p
        return None

    def exists(self, pid: str) -> bool:
        with self._lock:
            return pid in self._data

    def audio_path(self, p: Profile) -> Optional[Path]:
        if not p.audio_file:
            return None
        path = self.profile_dir(p.id) / p.audio_file
        return path if path.exists() else None

    # ------------------------------- операции ------------------------------- #

    def create(
        self,
        name: str,
        *,
        kind: str = "clone",
        audio_src: Any = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        x_vector_only: bool = False,
        tags: Optional[List[str]] = None,
        notes: str = "",
        model_key: Optional[str] = None,
        target_sr: int = audio_mod.NATIVE_SR,
        trim: bool = True,
        normalize: bool = True,
        profile_id: Optional[str] = None,
    ) -> Tuple[Profile, Dict[str, Any]]:
        """Создать профиль. audio_src: путь / URL / base64 / bytes / (ndarray, sr)."""
        with self._lock:
            base = slugify(name or (speaker or "voice"))
            pid = slugify(profile_id or base) or slugify(f"voice-{int(time.time())}")
            if pid in self._data:
                raise ProfileError(f"профиль с id '{pid}' уже существует")

            kind = (kind or "clone").lower()
            if kind not in ("clone", "preset", "designed"):
                raise ProfileError(f"неизвестный тип профиля: {kind}")
            if kind == "preset" and not speaker:
                raise ProfileError("для профиля типа preset нужен speaker (встроенный тембр)")

            warnings: List[str] = []
            stats: Dict[str, Any] = {}
            audio_rel: Optional[str] = None

            pdir = self.profile_dir(pid)
            pdir.mkdir(parents=True, exist_ok=True)

            if audio_src is not None:
                wav, sr, origin = audio_mod.resolve_audio_source(audio_src, target_sr=target_sr)
                stats = audio_mod.audio_stats(wav, sr)
                if stats["duration_sec"] < self.min_sec:
                    warnings.append(
                        f"эталон короткий ({stats['duration_sec']:.1f} с). "
                        f"Рекомендуется {self.min_sec:.0f}–10 с чистой речи."
                    )
                if stats["duration_sec"] > self.max_sec:
                    warnings.append(f"эталон длинный ({stats['duration_sec']:.1f} с) — обрезан до {self.max_sec:.0f} с")
                if stats.get("silent"):
                    warnings.append("эталон почти беззвучный — проверьте запись")
                if stats.get("clipping"):
                    warnings.append("в эталоне есть клиппинг (перегруз) — качество клона может снизиться")
                if stats.get("rms_db", 0) < -40:
                    warnings.append("эталон очень тихий — увеличьте громкость записи")
                wav, sr = audio_mod.prepare_reference(
                    wav, sr, target_sr=target_sr, max_sec=self.max_sec, trim=trim, normalize=normalize
                )
                stats = audio_mod.audio_stats(wav, sr)
                audio_rel = "ref.wav"
                audio_mod.write_wav_file(pdir / audio_rel, wav, sr)
                stats["source"] = origin
                log.info(
                    "профиль %s: эталон %.2f с @ %d Гц (%s)", pid, stats["duration_sec"], sr, origin
                )

            if kind == "clone" and audio_rel is None:
                shutil.rmtree(pdir, ignore_errors=True)
                raise ProfileError("для профиля clone нужен образец голоса (audio)")
            if kind == "clone" and not x_vector_only and not (ref_text or "").strip():
                warnings.append(
                    "не указан ref_text (транскрипт эталона). Клонирование перейдёт в режим "
                    "x-vector-only (ниже сходство) — лучше заполнить транскрипт."
                )

            p = Profile(
                id=pid,
                name=(name or pid).strip(),
                kind=kind,
                speaker=speaker,
                instruct=(instruct or "").strip() or None,
                language=language,
                ref_text=(ref_text or "").strip() or None,
                x_vector_only=bool(x_vector_only),
                audio_file=audio_rel,
                tags=[t.strip() for t in (tags or []) if t and t.strip()],
                notes=notes or "",
                model_key=model_key,
                extra={"stats": stats, "warnings": warnings},
            )
            self._data[pid] = p
            (pdir / "meta.json").write_text(
                json.dumps(p.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._flush()
            log.info("создан профиль '%s' (%s, %s)", p.name, p.id, p.kind)
            for w in warnings:
                log.warning("профиль %s: %s", p.id, w)
            return p, {"warnings": warnings, "stats": stats}

    def update(self, pid: str, **changes: Any) -> Profile:
        with self._lock:
            p = self.get(pid)
            audio_src = changes.pop("audio", None)
            for k, v in changes.items():
                if k in ("id", "created_at", "extra"):
                    continue
                if hasattr(p, k) and v is not None:
                    setattr(p, k, v)
            if audio_src is not None:
                pdir = self.profile_dir(pid)
                pdir.mkdir(parents=True, exist_ok=True)
                wav, sr, origin = audio_mod.resolve_audio_source(audio_src, target_sr=audio_mod.NATIVE_SR)
                wav, sr = audio_mod.prepare_reference(wav, sr, target_sr=audio_mod.NATIVE_SR, max_sec=self.max_sec)
                audio_mod.write_wav_file(pdir / "ref.wav", wav, sr)
                p.audio_file = "ref.wav"
                p.extra["stats"] = audio_mod.audio_stats(wav, sr)
                p.extra["stats"]["source"] = origin
            p.updated_at = time.time()
            (self.profile_dir(pid) / "meta.json").write_text(
                json.dumps(p.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._flush()
            log.info("обновлён профиль '%s'", pid)
            return p

    def delete(self, pid: str) -> bool:
        with self._lock:
            if pid not in self._data:
                return False
            del self._data[pid]
            shutil.rmtree(self.profile_dir(pid), ignore_errors=True)
            self._flush()
            log.info("удалён профиль '%s'", pid)
            return True

    def duplicate(self, pid: str, new_name: str) -> Profile:
        with self._lock:
            src = self.get(pid)
            new_id = slugify(new_name)
            n = 2
            while new_id in self._data:
                new_id = f"{slugify(new_name)}-{n}"
                n += 1
            new_dir = self.profile_dir(new_id)
            if new_dir.exists():
                shutil.rmtree(new_dir, ignore_errors=True)
            shutil.copytree(self.profile_dir(pid), new_dir)
            data = src.to_dict()
            data.update(id=new_id, name=new_name, created_at=time.time(), updated_at=time.time())
            p = Profile.from_dict(data)
            self._data[new_id] = p
            (new_dir / "meta.json").write_text(json.dumps(p.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._flush()
            return p

    def describe(self, p: Profile) -> Dict[str, Any]:
        d = p.to_dict()
        ap = self.audio_path(p)
        d["audio"] = None
        if ap:
            st = dict(p.extra.get("stats") or {})
            try:
                st["size_bytes"] = ap.stat().st_size
            except OSError:
                pass
            st["path"] = str(ap)
            d["audio"] = st
        d["warnings"] = p.extra.get("warnings") or []
        d.pop("extra", None)
        d["url"] = f"/api/profiles/{p.id}/audio"
        return d

    def load_reference(self, pid: str) -> Tuple[Optional[Path], Optional[str]]:
        p = self.get(pid)
        return self.audio_path(p), p.ref_text
