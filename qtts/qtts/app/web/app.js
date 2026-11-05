/* Qwen TTS Studio — веб-GUI (vanilla JS, без внешних зависимостей). */
'use strict';

const BOOT = window.__QTTS_BOOT__ || {};
const state = {
  cfg: null,
  voices: { presets: [], profiles: [] },
  diag: null,
  job: null,
  abort: null,
  ws: null,
  logLines: [],
  gpuHist: [],
  es: null,
  mediaRec: null,
  recChunks: [],
  recBlob: null,
};

/* ---------------------------------------------------------------- utils */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}

function toast(msg, kind = '', ms = 4200) {
  const host = $('#toast-host');
  const t = el('div', { class: `toast ${kind}`, text: String(msg) });
  host.append(t);
  setTimeout(() => t.remove(), ms);
}

function authHeaders(extra = {}) {
  const h = Object.assign({}, extra);
  if (BOOT.uiToken) h['X-API-Key'] = BOOT.uiToken;
  return h;
}

async function api(path, opts = {}) {
  const o = Object.assign({ method: 'GET' }, opts);
  o.headers = authHeaders(o.headers || {});
  if (o.body && !(o.body instanceof FormData) && typeof o.body !== 'string') {
    o.headers['Content-Type'] = 'application/json';
    o.body = JSON.stringify(o.body);
  }
  const r = await fetch(path, o);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      detail = j.detail ? (typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail))
        : (j.error || detail);
    } catch (e) { /* ignore */ }
    const err = new Error(detail);
    err.status = r.status;
    throw err;
  }
  if (r.status === 204) return null;
  const ct = r.headers.get('content-type') || '';
  return ct.includes('application/json') ? r.json() : r.text();
}

const fmtMs = (ms) => (ms === null || ms === undefined ? '—' : `${Math.round(ms)} мс`);
const fmtSec = (s) => (s === null || s === undefined ? '—' : `${Number(s).toFixed(2)} с`);
const fmtBytes = (n) => {
  if (!n && n !== 0) return '—';
  const u = ['Б', 'КиБ', 'МиБ', 'ГиБ'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};

function kv(host, obj) {
  const node = typeof host === 'string' ? $(host) : host;
  if (!node) return;
  node.innerHTML = '';
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined || v === '') continue;
    node.append(el('div', {}, [el('span', { text: k }), el('b', { text: String(v) })]));
  }
}

/* ----------------------------------------------------------------- tabs */
const TAB_LOADERS = {
  synth: loadSynthTab,
  profiles: loadProfilesTab,
  system: loadSystemTab,
  settings: loadSettingsTab,
  logs: loadLogsTab,
};

function switchTab(name) {
  $$('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.panel').forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
  state.activeTab = name;
  const fn = TAB_LOADERS[name];
  if (fn) Promise.resolve().then(fn).catch((e) => toast(`Ошибка вкладки: ${e.message}`, 'err'));
}

/* ------------------------------------------------------------- заголовки */
async function refreshHeader() {
  try {
    const [ready, gpu] = await Promise.all([
      api('/api/system/readiness').catch(() => null),
      api('/api/system/gpu').catch(() => null),
    ]);
    const pillHealth = $('#pill-health');
    const pillGpu = $('#pill-gpu');
    const pillBack = $('#pill-backend');
    if (ready) {
      pillBack.textContent = `backend: ${ready.engine || '?'}`;
      pillBack.className = 'pill ' + (ready.engine === 'qwen_tts' ? 'ok' : 'warn');
      const name = ready.gpu_name ? `${ready.gpu_name}${ready.gfx ? ' ' + ready.gfx : ''}` : 'не определена';
      pillGpu.textContent = `GPU: ${name}`;
      pillGpu.className = 'pill ' + (ready.gpu_visible ? 'ok' : 'err');
      const label = ready.ready ? (ready.model_loaded ? 'готов' : 'модель не загружена')
        : (ready.gpu_visible ? 'не готов' : 'нет GPU');
      pillHealth.textContent = `● ${label}`;
      pillHealth.className = 'pill ' + (ready.ready ? 'ok' : ready.gpu_visible ? 'warn' : 'err');
    }
    if (gpu && gpu.gpu_visible !== undefined) {
      const sysfs = (gpu.sysfs || [])[0];
      if (sysfs) {
        const used = sysfs.vram_used_bytes || 0;
        const total = sysfs.vram_total_bytes || 1;
        $('#pill-gpu').title = `VRAM ${fmtBytes(used)} / ${fmtBytes(total)}, загрузка ${sysfs.gpu_busy_percent}%`;
      }
      state.gpuHist.push({ t: Date.now(), vram: gpu.vram_used || (gpu.sysfs && gpu.sysfs[0] && gpu.sysfs[0].vram_used_bytes) || 0,
        total: gpu.vram_total || (gpu.sysfs && gpu.sysfs[0] && gpu.sysfs[0].vram_total_bytes) || 1,
        busy: gpu.sysfs && gpu.sysfs[0] ? gpu.sysfs[0].gpu_busy_percent : -1 });
      if (state.gpuHist.length > 180) state.gpuHist.shift();
      if (state.activeTab === 'system') drawGpuChart();
    }
  } catch (e) { /* сервер мог перезапускаться */ }
}

/* ------------------------------------------------------- вкладка: синтез */
async function loadSynthTab() {
  await Promise.all([loadVoices(), loadConfigIfMissing()]);
  fillVoiceSelect();
  await refreshJobs();
}

async function loadConfigIfMissing() {
  if (!state.cfg) state.cfg = await api('/api/config').catch(() => null);
}

async function loadVoices() {
  try {
    state.voices = await api('/api/voices');
  } catch (e) {
    state.voices = { presets: [], profiles: [] };
  }
  const langs = await api('/api/voices/languages').catch(() => ({ languages: ['Auto'] }));
  state.languages = langs.languages || ['Auto'];
  fillLanguageSelects();
}

function fillLanguageSelects() {
  for (const id of ['#synth-language', '#pf-language']) {
    const sel = $(id);
    if (!sel) continue;
    const cur = sel.value;
    sel.innerHTML = '';
    for (const l of state.languages || ['Auto']) sel.append(el('option', { value: l, text: l }));
    if (cur) sel.value = cur;
  }
}

function fillVoiceSelect() {
  const sel = $('#synth-voice');
  if (!sel) return;
  const cur = sel.value;
  sel.innerHTML = '';
  const profiles = state.voices.profiles || [];
  if (profiles.length) {
    const g = el('optgroup', { label: 'Профили (клонирование)' });
    for (const p of profiles) g.append(el('option', { value: p.id, text: `${p.name}${p.kind === 'preset' ? ' (пресет)' : ''}` }));
    sel.append(g);
  }
  const presets = state.voices.presets || [];
  if (presets.length) {
    const g = el('optgroup', { label: 'Встроенные тембры' });
    for (const s of presets) g.append(el('option', { value: s, text: s }));
    sel.append(g);
  }
  if (cur) sel.value = cur;
  const models = (state.cfg && state.cfg.models) || [];
  const ms = $('#synth-model');
  if (ms) {
    ms.innerHTML = '';
    for (const m of models) ms.append(el('option', { value: m.key, text: `${m.label || m.key} — ${m.source}` }));
  }
  const sp = $('#pf-speaker');
  if (sp) {
    sp.innerHTML = '';
    for (const s of presets.length ? presets : ['vivian', 'ryan', 'aiden', 'serena', 'dylan', 'eric', 'ono_anna', 'sohee', 'uncle_fu']) {
      sp.append(el('option', { value: s, text: s }));
    }
  }
  if (state.cfg && state.cfg.generation) {
    $('#synth-stream').checked = !!state.cfg.generation.streaming;
  }
}

function currentSynthBody() {
  const body = {
    text: $('#synth-text').value,
    voice: $('#synth-voice').value || null,
    language: $('#synth-language').value || 'Auto',
    model: $('#synth-model').value || null,
    format: $('#synth-format').value,
    stream: $('#synth-stream').checked,
    speed: parseFloat($('#synth-speed').value) || 1,
  };
  const put = (id, key, parse = (v) => v) => {
    const v = $(id).value.trim();
    if (v !== '') body[key] = parse(v);
  };
  put('#synth-temp', 'temperature', parseFloat);
  put('#synth-topp', 'top_p', parseFloat);
  put('#synth-topk', 'top_k', parseInt);
  put('#synth-seed', 'seed', parseInt);
  put('#synth-mnt', 'max_new_tokens', parseInt);
  put('#synth-instruct', 'instruct');
  if ($('#synth-xvec').checked) body.x_vector_only_mode = true;
  return body;
}

function setMetric(id, value) {
  const n = $(id);
  if (n) n.textContent = value === undefined || value === null ? '—' : value;
}

function resetMetrics() {
  for (const id of ['#m-state', '#m-stage', '#m-ttfa', '#m-elapsed', '#m-audio', '#m-rtf', '#m-chunks', '#m-tps']) setMetric(id, '—');
  $('#synth-progress').style.width = '0%';
  $('#synth-error').classList.add('hidden');
}

/* --- обычный HTTP-стриминг: читаем чанки и играем через AudioContext --- */
let audioCtx = null;
let playHead = 0;

function ensureCtx(sr) {
  if (!audioCtx || audioCtx.sampleRate !== sr) {
    if (audioCtx) audioCtx.close();
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sr });
    playHead = 0;
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}

function playPcm(pcm16, sr) {
  const ctx = ensureCtx(sr);
  const f32 = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 32768;
  const buf = ctx.createBuffer(1, f32.length, sr);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const now = ctx.currentTime;
  const start = Math.max(now, playHead);
  src.start(start);
  playHead = start + buf.duration;
}

async function synthHttp(body) {
  resetMetrics();
  setMetric('#m-state', 'запрос…');
  state.abort = new AbortController();
  const headers = authHeaders({ 'Content-Type': 'application/json' });
  let resp;
  try {
    resp = await fetch('/api/tts', {
      method: 'POST', headers, body: JSON.stringify(body), signal: state.abort.signal,
    });
  } catch (e) {
    setMetric('#m-state', 'ошибка');
    showError(`Сеть: ${e.message}`);
    return;
  }
  if (!resp.ok) {
    let detail = `${resp.status}`;
    try { const j = await resp.json(); detail = j.detail ? JSON.stringify(j.detail) : detail; } catch (e) {}
    setMetric('#m-state', 'ошибка');
    showError(detail);
    return;
  }
  const jobId = resp.headers.get('X-Job-Id');
  state.job = jobId;
  const srHeader = parseInt(resp.headers.get('X-Sample-Rate') || '24000', 10);
  const ct = resp.headers.get('content-type') || '';
  const chunks = [];
  let got = 0, started = performance.now(), firstAt = null;

  if (body.stream) {
    setMetric('#m-state', 'стриминг');
    // Сервер отдаёт ЕДИНЫЙ поток: для wav — 44-байтовый RIFF-заголовок (размеры
    // 0xFFFFFFFF) и далее сырые PCM16-чанки; для pcm — сразу PCM16.
    // Сетевые фрагменты произвольной длины: заголовок добираем по байтам,
    // нечётный хвост переносим в следующий фрагмент.
    const isWavStream = !ct.includes('audio/L16') && body.format !== 'pcm';
    const reader = resp.body.getReader();
    let needHeader = isWavStream ? 44 : 0;
    let headerBuf = new Uint8Array(0);
    let streamSr = srHeader;
    let carry = -1;
    let nAudio = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || !value.length) continue;
      if (firstAt === null) { firstAt = performance.now(); setMetric('#m-ttfa', fmtMs(firstAt - started)); }
      got += value.length;
      chunks.push(value);
      setMetric('#m-elapsed', fmtMs(performance.now() - started));
      let view = value;
      if (needHeader > 0) {
        const take = Math.min(needHeader, view.length);
        const nb = new Uint8Array(headerBuf.length + take);
        nb.set(headerBuf);
        nb.set(view.subarray(0, take), headerBuf.length);
        headerBuf = nb;
        needHeader -= take;
        view = view.subarray(take);
        if (needHeader === 0 && headerBuf.length >= 28 &&
            String.fromCharCode(headerBuf[0], headerBuf[1], headerBuf[2], headerBuf[3]) === 'RIFF') {
          streamSr = new DataView(headerBuf.buffer, headerBuf.byteOffset, 28).getUint32(24, true) || streamSr;
        }
      }
      if (view.length) nAudio++;
      let pcm = view;
      if (carry >= 0) {
        const merged = new Uint8Array(pcm.length + 1);
        merged[0] = carry;
        merged.set(pcm, 1);
        pcm = merged;
        carry = -1;
      }
      if (pcm.length % 2) { carry = pcm[pcm.length - 1]; pcm = pcm.subarray(0, pcm.length - 1); }
      if (pcm.length) {
        const dv = new DataView(pcm.buffer, pcm.byteOffset, pcm.length);
        const i16 = new Int16Array(pcm.length / 2);
        for (let i = 0; i < i16.length; i++) i16[i] = dv.getInt16(i * 2, true);
        playPcm(i16, streamSr);
      }
      setMetric('#m-chunks', `${nAudio}`);
      pollJob(jobId);
    }
  } else {
    setMetric('#m-state', 'генерация…');
    const buf = await resp.arrayBuffer();
    chunks.push(new Uint8Array(buf));
    got = buf.byteLength;
    firstAt = performance.now();
    setMetric('#m-ttfa', fmtMs(firstAt - started));
  }

  const all = new Uint8Array(got);
  let pos = 0;
  for (const c of chunks) { all.set(c, pos); pos += c.length; }
  // Для стрим-WAV пропатчим размеры RIFF/data (в потоке они 0xFFFFFFFF) —
  // тогда скачанный файл и <audio>-плеер получают полностью валидный WAV.
  if (body.stream && all.length >= 44 &&
      String.fromCharCode(all[0], all[1], all[2], all[3]) === 'RIFF' &&
      String.fromCharCode(all[8], all[9], all[10], all[11]) === 'WAVE') {
    const dv = new DataView(all.buffer);
    dv.setUint32(4, all.length - 8, true);
    dv.setUint32(40, all.length - 44, true);
  }
  const blob = new Blob([all], { type: ct || 'audio/wav' });
  const url = URL.createObjectURL(blob);
  const audio = $('#synth-audio');
  audio.src = url;
  if (!body.stream) audio.play().catch(() => {});
  const a = $('#synth-download');
  a.href = url;
  a.download = `qwen-tts-${Date.now()}.${body.format === 'pcm' ? 'pcm' : body.format}`;
  setMetric('#m-state', 'готово');
  setMetric('#m-elapsed', fmtMs(performance.now() - started));
  const dur = await resp.headers.get('X-Audio-Duration');
  if (dur) { setMetric('#m-audio', `${Number(dur).toFixed(2)} с`); setMetric('#m-rtf', resp.headers.get('X-TTS-RTF') || '—'); }
  if (jobId) pollJob(jobId, true);
  refreshJobs();
  toast(`Готово: ${fmtBytes(got)}`, 'ok');
}

// Состояния задания приходят с сервера по-английски — в интерфейсе показываем по-русски.
const JOB_STATE_RU = {
  queued: 'в очереди', running: 'генерация…', streaming: 'стриминг',
  done: 'готово', error: 'ошибка', cancelled: 'отменено', timeout: 'тайм-аут',
};
function ruJobState(s) { return JOB_STATE_RU[s] || s || '—'; }

async function pollJob(jobId, final = false) {
  if (!jobId) return;
  try {
    const j = await api(`/api/jobs/${jobId}`);
    setMetric('#m-stage', j.stage || '—');
    if (j.ttfa_ms) setMetric('#m-ttfa', fmtMs(j.ttfa_ms));
    if (j.audio_sec) setMetric('#m-audio', fmtSec(j.audio_sec));
    if (j.rtf) setMetric('#m-rtf', j.rtf);
    if (j.tok_per_sec) setMetric('#m-tps', j.tok_per_sec);
    if (j.chunks_total) {
      setMetric('#m-chunks', `${j.chunks_done}/${j.chunks_total}`);
      $('#synth-progress').style.width = `${Math.round((j.progress || 0) * 100)}%`;
    }
    if (j.error) showError(`${j.error_code || 'error'}: ${j.error}`);
    if (final) setMetric('#m-state', ruJobState(j.state));
  } catch (e) { /* ignore */ }
}

function showError(text) {
  const box = $('#synth-error');
  box.textContent = text;
  box.classList.remove('hidden');
}

/* --- WebSocket-стриминг --- */
function synthWs(body) {
  resetMetrics();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/tts`;
  let ws;
  try { ws = new WebSocket(url); } catch (e) { showError(e.message); return; }
  state.ws = ws;
  ws.binaryType = 'arraybuffer';
  const started = performance.now();
  let got = 0, n = 0;
  const parts = [];
  let sr = 24000;
  ws.onopen = () => {
    setMetric('#m-state', 'WS: отправка');
    ws.send(JSON.stringify(Object.assign({ stream: true, format: body.format === 'pcm' ? 'pcm' : 'pcm' }, body)));
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      if (m.event === 'job') { state.job = m.job_id; setMetric('#m-state', `WS: job ${m.job_id}`); }
      else if (m.event === 'progress') {
        n++;
        if (m.sample_rate) sr = m.sample_rate;
        if (m.audio_sec) setMetric('#m-audio', `${(parseFloat(m.audio_sec) || 0).toFixed(2)} с`);
        if (m.gen_ms) setMetric('#m-elapsed', fmtMs(m.gen_ms));
        if (m.tok_per_sec) setMetric('#m-tps', m.tok_per_sec);
        setMetric('#m-chunks', `${n}${m.total ? '/' + m.total : ''}`);
        setMetric('#m-stage', `чанк ${n}`);
        if (m.total) $('#synth-progress').style.width = `${Math.round(((m.index + 1) / m.total) * 100)}%`;
        if (m.elapsed_ms && n === 1) setMetric('#m-ttfa', fmtMs(m.elapsed_ms));
      } else if (m.event === 'done') {
        setMetric('#m-state', 'готово');
        if (m.rtf) setMetric('#m-rtf', m.rtf);
        if (m.ttfa_ms) setMetric('#m-ttfa', fmtMs(m.ttfa_ms));
        $('#synth-progress').style.width = '100%';
        finish();
      } else if (m.event === 'error') {
        setMetric('#m-state', 'ошибка');
        showError(`${m.code || 'error'}: ${m.error}`);
      }
    } else {
      got += ev.data.byteLength;
      parts.push(new Uint8Array(ev.data));
      const pcm = new Int16Array(ev.data);
      playPcm(pcm, sr);
    }
  };
  ws.onerror = () => { setMetric('#m-state', 'ошибка WS'); };
  ws.onclose = () => { state.ws = null; };
  function finish() {
    const all = new Uint8Array(got);
    let p = 0; for (const c of parts) { all.set(c, p); p += c.length; }
    const blob = new Blob([all], { type: 'audio/L16' });
    const url2 = URL.createObjectURL(blob);
    const a = $('#synth-download');
    a.href = url2; a.download = `qwen-tts-${Date.now()}.pcm`;
    setMetric('#m-elapsed', fmtMs(performance.now() - started));
    refreshJobs();
    try { ws.close(); } catch (e) {}
  }
}

function stopAll() {
  if (state.abort) { try { state.abort.abort(); } catch (e) {} state.abort = null; }
  if (state.ws) { try { state.ws.send(JSON.stringify({ cancel: state.job })); state.ws.close(); } catch (e) {} }
  if (state.job) api(`/api/jobs/${state.job}/cancel`, { method: 'POST' }).catch(() => {});
  setMetric('#m-state', 'остановлено');
  toast('Остановка запрошена');
}

/* --------------------------------------------------------------- задания */
async function refreshJobs() {
  try {
    const data = await api('/api/jobs?limit=15');
    const tb = $('#jobs-table tbody');
    tb.innerHTML = '';
    for (const j of data.jobs || []) {
      const badge = el('span', { class: `badge ${j.state === 'done' ? 'ok' : j.state === 'error' ? 'err' : 'run'}`, text: j.state });
      const tr = el('tr', {}, [
        el('td', { class: 'mono', text: j.id }),
        el('td', {}, [badge]),
        el('td', { text: j.stage || '—' }),
        el('td', { text: j.voice || '—' }),
        el('td', { class: 'mono', text: (j.text || '').length }),
        el('td', { class: 'mono', text: j.audio_sec ? `${j.audio_sec.toFixed(2)} с` : '—' }),
        el('td', { class: 'mono', text: j.ttfa_ms ? `${Math.round(j.ttfa_ms)} мс` : '—' }),
        el('td', { class: 'mono', text: j.rtf || '—' }),
        el('td', { class: 'mono', text: j.elapsed_ms ? `${Math.round(j.elapsed_ms)} мс` : '—' }),
        el('td', {}, [
          j.state === 'done' && j.output_file
            ? el('a', { class: 'btn ghost small', href: `/api/jobs/${j.id}/audio`, text: '⬇' })
            : (j.state === 'queued' || j.state === 'running'
              ? el('button', { class: 'ghost small', text: '■', onclick: () => api(`/api/jobs/${j.id}/cancel`, { method: 'POST' }).then(refreshJobs) })
              : el('span', { class: 'muted', text: j.error ? '⚠' : '' })),
        ]),
      ]);
      if (j.error) tr.title = j.error;
      tb.append(tr);
    }
    if (!tb.children.length) tb.append(el('tr', {}, [el('td', { colspan: '10', class: 'muted', text: 'пока пусто' })]));
  } catch (e) { /* ignore */ }
}

/* ----------------------------------------------------- вкладка: профили */
async function loadProfilesTab() {
  await loadVoices();
  fillVoiceSelect();
  renderProfiles();
}

function renderProfiles() {
  const host = $('#profiles-list');
  host.innerHTML = '';
  const list = state.voices.profiles || [];
  $('#profiles-count').textContent = `(${list.length})`;
  if (!list.length) {
    host.append(el('p', { class: 'muted', text: 'Профилей пока нет. Загрузите образец голоса слева.' }));
    return;
  }
  for (const p of list) {
    const card = el('div', { class: 'profile' });
    card.append(el('h4', {}, [p.name, ' ', el('span', { class: `badge ${p.kind === 'clone' ? 'info' : 'ok'}`, text: p.kind })]));
    card.append(el('div', { class: 'meta', text: `id: ${p.id}` }));
    if (p.audio) {
      card.append(el('div', { class: 'meta', text: `${(p.audio.duration_sec || 0).toFixed(2)} с · ${p.audio.sample_rate} Гц · ${fmtBytes(p.audio.size_bytes)} · RMS ${p.audio.rms_db} дБ` }));
      const au = el('audio', { controls: '', preload: 'none' });
      au.src = `${p.url}?api_key=${encodeURIComponent(BOOT.uiToken || '')}`;
      card.append(au);
    }
    if (p.ref_text) card.append(el('div', { class: 'ref', text: `«${p.ref_text}»` }));
    if (p.speaker) card.append(el('div', { class: 'meta', text: `тембр: ${p.speaker}` }));
    if (p.instruct) card.append(el('div', { class: 'meta', text: `стиль: ${p.instruct}` }));
    if (p.tags && p.tags.length) card.append(el('div', { class: 'meta', text: `теги: ${p.tags.join(', ')}` }));
    if (p.warnings && p.warnings.length) card.append(el('div', { class: 'warns', text: `⚠ ${p.warnings.join('; ')}` }));
    const actions = el('div', { class: 'actions' });
    actions.append(el('button', { class: 'small', text: '▶ Тест', onclick: () => testProfile(p.id) }));
    actions.append(el('button', { class: 'small', text: '✎', title: 'Изменить транскрипт/заметки', onclick: () => editProfile(p) }));
    actions.append(el('button', { class: 'small', text: '⧉', title: 'Дублировать', onclick: () => dupProfile(p) }));
    actions.append(el('button', { class: 'small danger', text: '🗑', onclick: () => delProfile(p) }));
    card.append(actions);
    host.append(card);
  }
}

async function testProfile(pid) {
  toast(`Тестовый синтез профиля ${pid}…`);
  try {
    const r = await fetch(`/api/profiles/${pid}/test`, {
      method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: '{}',
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.status);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const au = new Audio(url);
    au.play().catch(() => {});
    toast('Готово — играю тестовую фразу', 'ok');
  } catch (e) { toast(`Ошибка теста: ${e.message}`, 'err', 7000); }
}

async function editProfile(p) {
  const ref = prompt('Транскрипт образца (ref_text):', p.ref_text || '');
  if (ref === null) return;
  const notes = prompt('Заметки:', p.notes || '');
  if (notes === null) return;
  try {
    await api(`/api/profiles/${p.id}`, { method: 'PATCH', body: { ref_text: ref, notes } });
    toast('Профиль обновлён', 'ok');
    await loadVoices(); renderProfiles();
  } catch (e) { toast(e.message, 'err'); }
}

async function dupProfile(p) {
  const name = prompt('Имя копии:', `${p.name} (копия)`);
  if (!name) return;
  try {
    await api(`/api/profiles/${p.id}/duplicate?name=${encodeURIComponent(name)}`, { method: 'POST' });
    await loadVoices(); renderProfiles(); toast('Скопировано', 'ok');
  } catch (e) { toast(e.message, 'err'); }
}

async function delProfile(p) {
  if (!confirm(`Удалить профиль «${p.name}» вместе с образцом?`)) return;
  try {
    await api(`/api/profiles/${p.id}`, { method: 'DELETE' });
    await loadVoices(); renderProfiles(); toast('Удалено', 'ok');
  } catch (e) { toast(e.message, 'err'); }
}

async function createProfile() {
  const kind = $('#pf-kind').value;
  const fd = new FormData();
  fd.append('name', $('#pf-name').value.trim());
  fd.append('kind', kind);
  fd.append('language', $('#pf-language').value || '');
  const instruct = $('#pf-instruct').value.trim();
  if (instruct) fd.append('instruct', instruct);
  const tags = $('#pf-tags').value.trim();
  if (tags) fd.append('tags', tags);
  const notes = $('#pf-notes').value.trim();
  if (notes) fd.append('notes', notes);
  if (kind === 'clone') {
    const rt = $('#pf-reftext').value.trim();
    if (rt) fd.append('ref_text', rt);
    if ($('#pf-xvec').checked) fd.append('x_vector_only', 'true');
    const f = $('#pf-file').files[0];
    if (state.recBlob && (!f || f.__synthetic)) fd.append('file', state.recBlob, 'recording.webm');
    else if (f) fd.append('file', f);
    else { pfMsg('Приложите файл или запишите образец с микрофона', true); return; }
  } else {
    fd.append('speaker', $('#pf-speaker').value);
  }
  if (!fd.get('name')) { pfMsg('Укажите имя профиля', true); return; }
  try {
    const r = await api('/api/profiles', { method: 'POST', body: fd });
    const warns = (r.warnings || []).join('\n');
    pfMsg(`Профиль «${r.profile.name}» создан (id: ${r.profile.id}).${warns ? '\n⚠ ' + warns : ''}`, false);
    $('#pf-name').value = ''; $('#pf-reftext').value = ''; $('#pf-file').value = '';
    state.recBlob = null; $('#pf-preview').classList.add('hidden');
    await loadVoices(); renderProfiles(); fillVoiceSelect();
    toast('Профиль создан', 'ok');
  } catch (e) { pfMsg(e.message, true); }
}

function pfMsg(text, isErr) {
  const n = $('#pf-msg');
  n.textContent = text;
  n.className = `msg ${isErr ? 'err' : ''}`;
  n.classList.remove('hidden');
}

async function analyzeFile() {
  const f = $('#pf-file').files[0] || state.recBlob;
  if (!f) { pfMsg('Сначала выберите файл или запишите аудио', true); return; }
  const fd = new FormData();
  fd.append('file', f, f.name || 'rec.webm');
  try {
    const r = await api('/api/profiles/analyze', { method: 'POST', body: fd });
    const s = r.stats;
    pfMsg(`Длительность ${s.duration_sec.toFixed(2)} с, ${s.sample_rate} Гц, RMS ${s.rms_db} дБ, пик ${s.peak}` +
      (r.warnings.length ? `\n⚠ ${r.warnings.join('; ')}` : '\n✓ образец годится'), r.warnings.length > 1);
  } catch (e) { pfMsg(e.message, true); }
}

function toggleRec() {
  const btn = $('#btn-rec');
  if (state.mediaRec && state.mediaRec.state === 'recording') {
    state.mediaRec.stop();
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    pfMsg('Браузер не даёт доступ к микрофону (нужен https или localhost)', true);
    return;
  }
  navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
    .then((stream) => {
      state.recChunks = [];
      const rec = new MediaRecorder(stream);
      state.mediaRec = rec;
      rec.ondataavailable = (e) => { if (e.data.size) state.recChunks.push(e.data); };
      rec.onstart = () => { btn.textContent = '⏹ Остановить запись'; $('#rec-status').textContent = 'идёт запись…'; };
      rec.onstop = () => {
        btn.textContent = '🎙 Записать с микрофона';
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(state.recChunks, { type: rec.mimeType || 'audio/webm' });
        state.recBlob = blob;
        blob.__synthetic = true;
        const url = URL.createObjectURL(blob);
        const pv = $('#pf-preview');
        pv.src = url; pv.classList.remove('hidden');
        $('#rec-status').textContent = `записано ${fmtBytes(blob.size)} (${rec.mimeType || 'audio/webm'}) — сервер перекодирует в wav 24 кГц`;
      };
      rec.start();
    })
    .catch((e) => pfMsg(`Микрофон недоступен: ${e.message}`, true));
}

/* ------------------------------------------------------- вкладка: система */
async function loadSystemTab() {
  await loadConfigIfMissing();
  renderModelsTable();
  refreshGpuLive();
  const info = await api('/api/system/info').catch(() => null);
  if (info) {
    kv('#versions', {
      'приложение': info.versions.app,
      'python': info.versions.python || '—',
      'torch': info.versions.torch || 'не установлен',
      'torchaudio': info.versions.torchaudio || '—',
      'transformers': info.versions.transformers || '—',
      'qwen-tts': info.versions.qwen_tts || 'не установлен',
      'flash-attn': info.versions.flash_attn || '—',
      'triton': info.versions.triton || '—',
      'vllm': info.versions.vllm || '—',
      'fastapi': info.versions.fastapi,
      'librosa': info.versions.librosa || '—',
      'ffmpeg': info.versions.ffmpeg || '—',
    });
    kv('#env-vars', Object.assign({ 'переменных ROCm не задано': '' }, info.env || {}));
    $('#btn-log-download');
  }
  if (state.diag) renderDiagnostics(state.diag);
}

async function runDiagnostics(btn) {
  const b = btn || $('#btn-diag');
  b.disabled = true; b.textContent = '⏳ Собираю данные…';
  try {
    const rep = await api('/api/system/diagnostics?refresh=true&bench=true');
    state.diag = rep;
    renderDiagnostics(rep);
    toast('Диагностика завершена', rep.health === 'ok' ? 'ok' : 'warn');
  } catch (e) { toast(`Ошибка диагностики: ${e.message}`, 'err', 8000); }
  b.disabled = false; b.textContent = '🔍 Запустить диагностику';
}

function renderDiagnostics(rep) {
  const sum = $('#diag-summary');
  sum.innerHTML = '';
  const t = rep.torch || {};
  const dev = (t.devices || [])[0] || {};
  const sys = rep.system || {};
  const cards = [
    ['Статус', rep.health === 'ok' ? 'OK' : rep.health === 'warn' ? 'есть замечания' : 'ПРОБЛЕМА', rep.health === 'ok' ? 'ok' : rep.health === 'warn' ? 'warn' : 'err'],
    ['GPU', dev.name || (rep.sysfs_gpus && rep.sysfs_gpus[0] ? 'виден в sysfs, но не в torch' : 'не найден'), dev.name ? 'ok' : 'err'],
    ['Архитектура', dev.gfx || '—', dev.gfx ? 'ok' : 'warn'],
    ['ROCm', (rep.rocm || {}).version || '—', 'info'],
    ['PyTorch', t.torch_version || '—', t.is_rocm_build ? 'ok' : 'warn'],
    ['HIP', t.hip_version || '—', t.hip_version ? 'ok' : 'warn'],
    ['VRAM', dev.vram_total_h || '—', 'info'],
    ['драйвер', sys.kernel_driver_amdgpu ? `amdgpu ${sys.kernel_driver_version || ''}` : 'НЕТ', sys.kernel_driver_amdgpu ? 'ok' : 'err'],
    ['/dev/kfd', (rep.groups || {}).kfd_readable ? 'доступен' : 'нет доступа', (rep.groups || {}).kfd_readable ? 'ok' : 'err'],
    ['CPU', sys.cpu_model || '—', 'info'],
    ['RAM', fmtBytes(sys.ram_total), 'info'],
  ];
  for (const [k, v, cls] of cards) sum.append(el('span', { class: `badge ${cls}`, text: `${k}: ${v}` }));

  const host = $('#diag-checks');
  host.innerHTML = '';
  for (const c of rep.checks || []) {
    const row = el('div', { class: `check-row ${c.status === 'error' ? 'err' : c.status === 'warn' ? 'warn' : ''}` });
    row.append(el('span', { class: `badge ${c.status === 'ok' ? 'ok' : c.status === 'warn' ? 'warn' : c.status === 'error' ? 'err' : 'info'}`, text: c.status }));
    row.append(el('span', { class: 'name', text: c.name }));
    row.append(el('span', { class: 'msg', text: c.message }));
    if (c.hint) row.append(el('span', { class: 'hint', text: `→ ${c.hint}` }));
    host.append(row);
  }

  kv('#torch-info', {
    'GPU виден через ROCm': t.gpu_visible ? 'да' : 'НЕТ',
    'устройств': t.device_count || 0,
    'имя': dev.name || '—',
    'gfx': dev.gfx || '—',
    'CU': dev.compute_units || '—',
    'VRAM всего': dev.vram_total_h || '—',
    'VRAM свободно': dev.vram_free_h || '—',
    'arch_list torch': (t.arch_list || []).join(', ') || '—',
    'SDPA flash': String((t.sdpa || {}).flash),
    'SDPA mem_efficient': String((t.sdpa || {}).mem_efficient),
    'SDPA math': String((t.sdpa || {}).math),
    'AOTriton': (t.pkg_aotriton || (rep.env || {}).TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL || '—'),
  });
  const b = t.benchmark || {};
  kv('#bench-info', {
    'GEMM bf16': b.gemm_tflops ? `${b.gemm_tflops} TFLOPS` : '—',
    'время 4096³': b.gemm_ms ? `${b.gemm_ms} мс` : '—',
    'H2D': b.h2d_gbps ? `${b.h2d_gbps} ГБ/с` : '—',
    'запуск ядра': b.kernel_launch_us ? `${b.kernel_launch_us} мкс` : '—',
    'ошибка': b.error || '',
  });
}

async function refreshGpuLive() {
  try {
    const s = await api('/api/system/gpu');
    const sysfs = (s.sysfs || [])[0] || {};
    kv('#gpu-live', {
      'виден в torch': s.gpu_visible === undefined ? '—' : (s.gpu_visible ? 'да' : 'нет'),
      'имя': s.name || sysfs.card || '—',
      'VRAM занято (torch)': fmtBytes(s.vram_allocated),
      'VRAM зарезервировано': fmtBytes(s.vram_reserved),
      'VRAM свободно': fmtBytes(s.vram_free),
      'VRAM занято (драйвер)': sysfs.vram_used || '—',
      'VRAM всего': sysfs.vram_total || fmtBytes(s.vram_total),
      'загрузка GPU': sysfs.gpu_busy_percent >= 0 ? `${sysfs.gpu_busy_percent}%` : '—',
      'мощность': sysfs.power_now_w ? `${Number(sysfs.power_now_w).toFixed(1)} Вт` : '—',
    });
    drawGpuChart();
  } catch (e) { /* ignore */ }
}

function drawGpuChart() {
  const cv = $('#gpu-canvas');
  if (!cv || !cv.getContext) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const hist = state.gpuHist;
  if (!hist.length) {
    ctx.fillStyle = '#5b6b7d'; ctx.font = '12px monospace';
    ctx.fillText('нет данных (GPU не виден или сервер не отвечает)', 12, H / 2);
    return;
  }
  const total = Math.max(...hist.map((h) => h.total)) || 1;
  ctx.beginPath();
  hist.forEach((h, i) => {
    const x = (i / Math.max(1, hist.length - 1)) * (W - 8) + 4;
    const y = H - 6 - (h.vram / total) * (H - 18);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = '#4aa3ff'; ctx.lineWidth = 1.6; ctx.stroke();
  ctx.lineTo(W - 4, H - 6); ctx.lineTo(4, H - 6); ctx.closePath();
  ctx.fillStyle = '#4aa3ff22'; ctx.fill();
  const busy = hist.map((h) => h.busy).filter((b) => b >= 0);
  if (busy.length) {
    ctx.beginPath();
    let i = 0;
    hist.forEach((h) => {
      if (h.busy < 0) return;
      const x = (i / Math.max(1, hist.length - 1)) * (W - 8) + 4;
      const y = H - 6 - (h.busy / 100) * (H - 18);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      i++;
    });
    ctx.strokeStyle = '#7ee787'; ctx.stroke();
  }
  ctx.fillStyle = '#8b98a9'; ctx.font = '11px monospace';
  ctx.fillText(`VRAM: ${fmtBytes(hist[hist.length - 1].vram)} / ${fmtBytes(total)}`, 8, 14);
}

async function renderModelsTable() {
  try {
    const d = await api('/api/models');
    const tb = $('#models-table tbody');
    tb.innerHTML = '';
    for (const m of d.models || []) {
      const info = m.info || {};
      tb.append(el('tr', {}, [
        el('td', { class: 'mono', text: m.key }),
        el('td', { class: 'mono', text: m.source }),
        el('td', { text: info.kind || m.kind }),
        el('td', {}, [el('span', { class: `badge ${m.loaded ? 'ok' : 'warn'}`, text: m.loaded ? (d.active === m.key ? 'активна' : 'загружена') : 'не загружена' })]),
        el('td', { class: 'mono', text: info.device || '—' }),
        el('td', { class: 'mono', text: info.dtype || '—' }),
        el('td', { class: 'mono', text: info.attn || '—' }),
        el('td', { class: 'mono', text: info.gfx || '—' }),
        el('td', {}, [
          el('button', { class: 'small', text: m.loaded ? '♻ перезагрузить' : '⬇ загрузить', onclick: () => loadModel(m.key, true) }),
          ' ',
          m.loaded ? el('button', { class: 'small danger', text: 'выгрузить', onclick: () => unloadModel(m.key) }) : null,
        ]),
      ]));
    }
    const acts = $('#models-actions');
    acts.innerHTML = '';
    acts.append(el('span', { class: 'badge info', text: `backend: ${d.backend}` }));
    for (const b of ['qwen_tts', 'vllm_omni', 'dummy']) {
      acts.append(el('button', {
        class: 'ghost small', text: `переключить на ${b}`, disabled: d.backend === b,
        onclick: () => switchBackend(b),
      }));
    }
  } catch (e) { toast(e.message, 'err'); }
}

async function loadModel(key, force) {
  toast(`Загружаю модель ${key}…`);
  try {
    const r = await api('/api/models/load', { method: 'POST', body: { model_key: key, force: !!force } });
    toast(`Модель ${key} загружена (${Math.round(r.model.load_ms)} мс, ${r.model.attn}/${r.model.dtype})`, 'ok', 7000);
    renderModelsTable(); refreshHeader();
  } catch (e) { toast(`Ошибка загрузки: ${e.message}`, 'err', 12000); }
}

async function unloadModel(key) {
  try {
    await api('/api/models/unload', { method: 'POST', body: { model_key: key } });
    toast(`Модель ${key} выгружена`, 'ok');
    renderModelsTable(); refreshHeader();
  } catch (e) { toast(e.message, 'err'); }
}

async function switchBackend(name) {
  try {
    await api('/api/backend', { method: 'POST', body: { backend: name } });
    toast(`Движок переключён на ${name}`, 'ok');
    renderModelsTable(); refreshHeader();
  } catch (e) { toast(`Не удалось переключить: ${e.message}`, 'err', 10000); }
}

async function selfTest() {
  const btn = $('#btn-selftest');
  btn.disabled = true; btn.textContent = '⏳ проверяю…';
  try {
    const r = await api('/api/system/test', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'Проверка режима работы. Видеокарта определена, синтез запущен.' }),
    });
    if (r.diagnostics) { state.diag = r.diagnostics; renderDiagnostics(r.diagnostics); }
    if (r.ok) {
      toast(`Проверка пройдена: ${r.test.audio_sec} с аудио за ${Math.round(r.test.elapsed_ms)} мс, RTF ${r.test.rtf}`, 'ok', 8000);
    } else {
      toast(`Проверка провалена: ${JSON.stringify(r.test_error)}`, 'err', 12000);
    }
  } catch (e) { toast(`Ошибка проверки: ${e.message}`, 'err', 10000); }
  btn.disabled = false; btn.textContent = '▶ Тестовый синтез';
}

/* ----------------------------------------------------- вкладка: настройки */
async function loadSettingsTab() {
  state.cfg = await api('/api/config');
  const c = state.cfg, d = c.device, g = c.generation, a = c.audio, s = c.server, l = c.log;
  $('#set-backend').value = d.backend;
  $('#set-device').value = d.device;
  $('#set-dtype').value = d.dtype;
  $('#set-attn').value = d.attn;
  $('#set-model-source').value = (c.models && c.models[0] ? c.models[0].source : '');
  $('#set-vllm-url').value = d.vllm_base_url || '';
  renderEnvEditor(d.env || {});
  $('#set-streaming').checked = !!g.streaming;
  $('#set-chunking').value = g.chunking;
  $('#set-chunk-target').value = g.chunk_target_chars;
  $('#set-gap').value = g.inter_chunk_silence_ms;
  $('#set-mnt').value = g.max_new_tokens;
  $('#set-tpc').value = g.tokens_per_char;
  $('#set-temp').value = g.temperature;
  $('#set-topk').value = g.top_k;
  $('#set-topp').value = g.top_p;
  $('#set-reppen').value = g.repetition_penalty;
  $('#set-nonstream').checked = !!g.non_streaming_mode;
  $('#set-warmup').checked = !!g.warmup;
  $('#set-compile').checked = !!g.compile;
  $('#set-promptcache').checked = !!(g.clone && g.clone.prompt_cache);
  $('#set-audio-fmt').value = a.format;
  $('#set-sr').value = a.sample_rate;
  $('#set-normalize').checked = !!a.normalize;
  $('#set-save').checked = !!a.save_outputs;
  $('#set-keep').value = a.keep_outputs;
  $('#set-host').value = s.host;
  $('#set-port').value = s.port;
  $('#set-apikey').value = s.api_key === '***' ? '' : (s.api_key || '');
  $('#set-preload').checked = !!s.preload_model;
  $('#set-loglevel').value = l.level;
  $('#set-logreq').checked = !!l.log_requests;
  $('#set-loggen').checked = !!l.log_generation;
}

function renderEnvEditor(env) {
  const host = $('#set-env-list');
  host.innerHTML = '';
  const entries = Object.entries(env || {});
  if (!entries.length) entries.push(['', '']);
  for (const [k, v] of entries) host.append(envRow(k, v));
}

function envRow(k, v) {
  const row = el('div', { class: 'env-row' });
  const ki = el('input', { type: 'text', value: k, placeholder: 'HSA_OVERRIDE_GFX_VERSION' });
  const vi = el('input', { type: 'text', value: v, placeholder: 'значение' });
  const del = el('button', { class: 'ghost small', text: '✕', onclick: () => row.remove() });
  row.append(ki, vi, del);
  row.__k = ki; row.__v = vi;
  return row;
}

function collectEnv() {
  const out = {};
  for (const row of $$('#set-env-list .env-row')) {
    const k = row.__k.value.trim(), v = row.__v.value.trim();
    if (k) out[k] = v;
  }
  return out;
}

async function saveSettings() {
  const patch = {
    device: {
      backend: $('#set-backend').value,
      device: $('#set-device').value,
      dtype: $('#set-dtype').value,
      attn: $('#set-attn').value,
      vllm_base_url: $('#set-vllm-url').value.trim(),
      env: collectEnv(),
    },
    generation: {
      streaming: $('#set-streaming').checked,
      chunking: $('#set-chunking').value,
      chunk_target_chars: parseInt($('#set-chunk-target').value, 10) || 220,
      inter_chunk_silence_ms: parseInt($('#set-gap').value, 10) || 0,
      max_new_tokens: parseInt($('#set-mnt').value, 10) || 4096,
      tokens_per_char: parseFloat($('#set-tpc').value) || 4,
      temperature: parseFloat($('#set-temp').value) || 0.9,
      top_k: parseInt($('#set-topk').value, 10) || 0,
      top_p: parseFloat($('#set-topp').value) || 1,
      repetition_penalty: parseFloat($('#set-reppen').value) || 1.05,
      non_streaming_mode: $('#set-nonstream').checked,
      warmup: $('#set-warmup').checked,
      compile: $('#set-compile').checked,
      clone: { prompt_cache: $('#set-promptcache').checked },
    },
    audio: {
      format: $('#set-audio-fmt').value,
      sample_rate: parseInt($('#set-sr').value, 10) || 0,
      normalize: $('#set-normalize').checked,
      save_outputs: $('#set-save').checked,
      keep_outputs: parseInt($('#set-keep').value, 10) || 0,
    },
    server: {
      host: $('#set-host').value.trim() || '127.0.0.1',
      port: parseInt($('#set-port').value, 10) || 8770,
      preload_model: $('#set-preload').checked,
    },
    log: {
      level: $('#set-loglevel').value,
      log_requests: $('#set-logreq').checked,
      log_generation: $('#set-loggen').checked,
    },
  };
  const key = $('#set-apikey').value.trim();
  if (key) patch.server.api_key = key;
  const src = $('#set-model-source').value.trim();
  if (src && state.cfg && state.cfg.models) {
    patch.models = state.cfg.models.map((m, i) => (i === 0 ? Object.assign({}, m, { source: src }) : m));
  }
  try {
    const r = await api('/api/config', { method: 'PATCH', body: patch });
    state.cfg = r.config;
    const need = r.restart_required || [];
    cfgMsg(need.length
      ? `Сохранено. Требуется перезагрузка: ${need.join(', ')} — нажмите «Перезагрузить движок» или перезапустите сервер.`
      : 'Сохранено и применено.', need.length > 0);
    if (need.includes('log') || patch.log) connectLogStream();
  } catch (e) { cfgMsg(`Ошибка: ${e.message}`, true); }
}

function cfgMsg(text, isWarn) {
  const n = $('#cfg-msg');
  n.textContent = text;
  n.className = `msg ${isWarn ? 'err' : ''}`;
  n.classList.remove('hidden');
}

async function reloadEngine() {
  cfgMsg('Перезагружаю движок и модель…', false);
  try {
    await api('/api/models/load', { method: 'POST', body: { force: true } });
    cfgMsg('Движок и модель перезагружены.', false);
    toast('Модель перезагружена', 'ok');
    refreshHeader();
  } catch (e) { cfgMsg(`Ошибка: ${e.message}`, true); }
}

async function applyEnvPreset() {
  try {
    const r = await api('/api/system/env', { method: 'POST', body: { preset: 'rdna4_rocm10', overwrite: false } });
    renderEnvEditor(Object.assign({}, collectEnv(), Object.fromEntries(r.applied.map((x) => [x.name, x.value]))));
    toast(r.note, r.restart_recommended ? 'warn' : 'ok', 9000);
  } catch (e) { toast(e.message, 'err'); }
}

async function checkMode() {
  switchTab('system');
  await selfTest();
}

/* -------------------------------------------------------- вкладка: журнал */
function loadLogsTab() {
  connectLogStream();
}

function pushLog(rec) {
  const host = $('#log-view');
  if (!host) return;
  const line = el('div', { class: 'log-line' });
  line.append(
    el('span', { class: 't', text: rec.time || '' }),
    el('span', { class: `l l-${rec.level}`, text: rec.level }),
    el('span', { class: 'n', text: rec.name || '' }),
    el('span', { class: 'm', text: rec.msg || '' }),
  );
  host.append(line);
  state.logLines.push(line);
  while (state.logLines.length > 2500) { const old = state.logLines.shift(); old.remove(); }
  if ($('#log-autoscroll').checked) host.scrollTop = host.scrollHeight;
}

function connectLogStream() {
  if (state.es) { try { state.es.close(); } catch (e) {} }
  const lvl = $('#log-filter').value;
  const url = `/api/logs/stream?api_key=${encodeURIComponent(BOOT.uiToken || '')}${lvl ? `&level=${lvl}` : ''}`;
  const es = new EventSource(url);
  state.es = es;
  es.onmessage = (ev) => {
    try { pushLog(JSON.parse(ev.data)); } catch (e) { /* ignore */ }
  };
  es.onerror = () => { /* EventSource сам переподключится */ };
  api(`/api/logs?limit=250${lvl ? `&level=${lvl}` : ''}`).then((d) => {
    $('#log-view').innerHTML = ''; state.logLines = [];
    for (const rec of d.logs || []) pushLog(rec);
  }).catch(() => {});
}

/* ------------------------------------------------------------- копирование */
function copyCurl() {
  const body = currentSynthBody();
  const host = location.origin;
  const key = BOOT.apiKeySet ? ' -H "X-API-Key: $QTTSS_KEY"' : '';
  const cmd = `curl -sS -X POST ${host}/api/tts \\\n  -H "Content-Type: application/json"${key} \\\n  -d '${JSON.stringify(body).replace(/'/g, "'\\''")}' \\\n  --output out.${body.format === 'pcm' ? 'pcm' : body.format}`;
  navigator.clipboard.writeText(cmd).then(() => toast('curl скопирован в буфер', 'ok'), () => prompt('Скопируйте вручную:', cmd));
}

async function saveAsProfile() {
  const au = $('#synth-audio');
  if (!au.src || au.src.startsWith('data:')) { toast('Сначала синтез', 'err'); return; }
  const blob = await (await fetch(au.src)).blob();
  const name = prompt('Имя нового профиля (голос из синтезированного аудио):', 'designed-voice');
  if (!name) return;
  const fd = new FormData();
  fd.append('name', name);
  fd.append('kind', 'clone');
  fd.append('file', blob, 'ref.wav');
  fd.append('ref_text', $('#synth-text').value.slice(0, 400));
  try {
    await api('/api/profiles', { method: 'POST', body: fd });
    toast('Профиль создан из синтезированного аудио', 'ok');
    await loadVoices(); fillVoiceSelect(); renderProfiles();
  } catch (e) { toast(e.message, 'err'); }
}

/* ------------------------------------------------------------------ init */
function bindEvents() {
  $$('.tab').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));

  $('#synth-text').addEventListener('input', () => {
    $('#synth-chars').textContent = `${$('#synth-text').value.length} симв.`;
  });
  $('#btn-synth').addEventListener('click', () => synthHttp(currentSynthBody()));
  $('#btn-synth-ws').addEventListener('click', () => synthWs(currentSynthBody()));
  $('#btn-stop').addEventListener('click', stopAll);
  $('#btn-copy-curl').addEventListener('click', copyCurl);
  $('#btn-to-profile').addEventListener('click', saveAsProfile);

  $('#btn-create-profile').addEventListener('click', createProfile);
  $('#btn-analyze').addEventListener('click', analyzeFile);
  $('#btn-rec').addEventListener('click', toggleRec);
  $('#pf-kind').addEventListener('change', () => {
    const clone = $('#pf-kind').value === 'clone';
    $('#pf-clone-fields').classList.toggle('hidden', !clone);
    $('#pf-preset-fields').classList.toggle('hidden', clone);
  });
  $('#pf-file').addEventListener('change', () => {
    const f = $('#pf-file').files[0];
    const pv = $('#pf-preview');
    if (f) { pv.src = URL.createObjectURL(f); pv.classList.remove('hidden'); state.recBlob = null; }
  });

  $('#btn-diag').addEventListener('click', () => runDiagnostics());
  $('#btn-selftest').addEventListener('click', selfTest);
  $('#btn-diag-copy').addEventListener('click', async () => {
    try {
      const txt = await api('/api/system/diagnostics/text');
      await navigator.clipboard.writeText(txt);
      toast('Отчёт скопирован', 'ok');
    } catch (e) { toast(e.message, 'err'); }
  });

  $('#btn-save-cfg').addEventListener('click', saveSettings);
  $('#btn-reload-engine').addEventListener('click', reloadEngine);
  $('#btn-check-mode').addEventListener('click', checkMode);
  $('#btn-env-preset').addEventListener('click', applyEnvPreset);
  $('#btn-env-add').addEventListener('click', () => $('#set-env-list').append(envRow('', '')));
  $('#log-filter').addEventListener('change', connectLogStream);
  $('#btn-log-clear').addEventListener('click', () => { $('#log-view').innerHTML = ''; state.logLines = []; });

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); synthHttp(currentSynthBody()); }
  });
}

async function boot() {
  $('#hdr-version').textContent = `v${BOOT.version || '?'} · ${BOOT.backend || ''}`;
  bindEvents();
  switchTab('synth');
  refreshHeader();
  setInterval(refreshHeader, 4000);
  setInterval(() => { if (state.activeTab === 'system') refreshGpuLive(); }, 2000);
  setInterval(() => { if (state.activeTab === 'synth') refreshJobs(); }, 5000);
  if (BOOT.apiKeySet && !BOOT.uiToken) toast('Сервер требует API-ключ: откройте GUI с локальной машины или укажите server.api_key', 'warn', 9000);
}

document.addEventListener('DOMContentLoaded', boot);
