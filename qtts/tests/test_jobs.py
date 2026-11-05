"""Очередь задач синтеза: состояния, история, подписка на события."""

from __future__ import annotations

import time

from qtts.app.engine.base import SynthesisRequest, VoiceSpec
from qtts.app.jobs import JobStore


def make_store() -> JobStore:
    return JobStore(max_history=3)


def req(text: str = "Текст") -> SynthesisRequest:
    return SynthesisRequest(text=text, voice=VoiceSpec(kind="preset", speaker="vivian"))


def test_create_sets_queued_state():
    st = make_store()
    job = st.create(req(), client="127.0.0.1", endpoint="/api/tts")
    assert job.state == "queued"
    assert job.client == "127.0.0.1" and job.endpoint == "/api/tts"
    assert st.get(job.id) is job


def test_touch_updates_stage_and_timing():
    st = make_store()
    job = st.create(req())
    job.touch("generating", chunks_total=3)
    assert job.stage == "generating" and job.chunks_total == 3
    time.sleep(0.02)
    assert job.elapsed_ms() >= 10


def test_history_is_bounded():
    st = make_store()
    jobs = [st.create(req(f"текст {i}")) for i in range(6)]
    listed = st.list(limit=50)
    assert len(listed) == 3
    assert st.get(jobs[0].id) is None          # самые старые вытеснены
    assert st.get(jobs[-1].id) is not None


def test_list_filter_by_state_and_limit():
    st = JobStore(max_history=10)
    a = st.create(req("a"))
    st.create(req("b"))
    st.update(a, state="done", audio_sec=1.5)
    assert len(st.list(state="done")) == 1
    assert len(st.list(limit=1)) == 1
    assert len(st.list()) == 2


def test_stats_counts_states_and_rtf():
    st = JobStore(max_history=10)
    j1 = st.create(req("a"))
    st.update(j1, state="done", audio_sec=4.0, rtf=0.25, ttfa_ms=300.0)
    j2 = st.create(req("b"))
    st.update(j2, state="error", error="упало")
    st.create(req("c"))
    s = st.stats()
    assert s["total"] == 3
    assert s["by_state"]["done"] == 1 and s["by_state"]["error"] == 1
    assert s["errors"] == 1
    assert s["avg_rtf"] == 0.25
    assert s["avg_ttfa_ms"] == 300.0
    assert s["audio_sec_total"] == 4.0


def test_subscribe_receives_publications():
    st = make_store()
    q = st.subscribe()
    job = st.create(req("подписка"))          # create() уже публикует событие
    first = q.get(timeout=1.0)
    assert first.id == job.id

    st.update(job, state="running")           # update() тоже публикует
    assert q.get(timeout=1.0).state == "running"

    while not q.empty():                      # выгребаем остаток
        q.get_nowait()
    st.unsubscribe(q)
    st.publish(job)
    assert q.empty()                          # после отписки событий не приходит


def test_to_dict_hides_text_by_default():
    st = make_store()
    job = st.create(req("секретный текст"))
    d = job.to_dict()
    assert "секретный текст" not in str(d)
    d2 = job.to_dict(with_text=True)
    assert "секретный текст" in str(d2)


def test_clear():
    st = make_store()
    st.create(req("a"))
    st.create(req("b"))
    assert st.clear() == 2
    assert st.list() == []
