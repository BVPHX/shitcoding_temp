/**
 * Smoke-тест веб-GUI в jsdom: поднимает index.html + app.js, проксирует fetch/WS
 * на живой сервер (backend=dummy) и «кликает» по всем вкладкам.
 *
 *   node tests/gui_smoke.mjs [baseUrl]
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { JSDOM, VirtualConsole } from 'jsdom';
import WebSocket from 'ws';

const BASE = process.argv[2] || 'http://127.0.0.1:8799';
const here = dirname(fileURLToPath(import.meta.url));
const webDir = join(here, '..', 'qtts', 'app', 'web');

let html = readFileSync(join(webDir, 'index.html'), 'utf8');
html = html.replace('/*__BOOT__*/', 'window.__QTTS_BOOT__ = {"version":"0.1.0-test","uiToken":"","apiKeySet":false,"backend":"dummy"};');
html = html.replace('<link rel="stylesheet" href="/ui/app.css">', '');

const problems = [];
const vc = new VirtualConsole();
vc.on('jsdomError', (e) => problems.push(`jsdomError: ${e.message}\n${(e.stack || '').split('\n').slice(0, 4).join('\n')}`));
vc.on('error', (...a) => problems.push(`console.error: ${a.join(' ')}`));
vc.on('warn', (...a) => problems.push(`console.warn: ${a.join(' ')}`));
vc.forwardTo(console, { omitJSDOMErrors: true });

const dom = new JSDOM(html, {
  url: BASE + '/',
  runScripts: 'outside-only',
  pretendToBeVisual: true,
  virtualConsole: vc,
});
const { window } = dom;

/* ---------------- полифилы того, чего нет в jsdom ---------------- */
window.fetch = async (url, opts = {}) => {
  const abs = new URL(url, BASE).toString();
  const r = await fetch(abs, opts);
  return r;
};
window.Headers = Headers;
window.Request = Request;
window.Response = Response;
window.FormData = FormData;
window.Blob = Blob;
window.File = File;
window.URL.createObjectURL = () => 'blob:stub';
window.URL.revokeObjectURL = () => {};
window.WebSocket = WebSocket;
window.EventSource = class {
  constructor(url) {
    this.url = new URL(url, BASE).toString();
    this.readyState = 0;
    setTimeout(() => this.close(), 4000);
  }
  close() { this.readyState = 2; }
  addEventListener() {}
};
window.AudioContext = class {
  constructor() { this.currentTime = 0; this.sampleRate = 24000; this.state = 'running'; this.destination = {}; }
  resume() {}
  close() {}
  createBuffer(ch, len, sr) { return { duration: len / sr, copyToChannel() {} }; }
  createBufferSource() { return { buffer: null, connect() {}, start() {} }; }
};
window.HTMLMediaElement.prototype.play = function () { return Promise.resolve(); };
// jsdom не умеет canvas — подставляем заглушку (в браузере работает настоящий 2D-контекст)
window.HTMLCanvasElement.prototype.getContext = function () {
  const noop = () => {};
  return {
    clearRect: noop, beginPath: noop, moveTo: noop, lineTo: noop, stroke: noop, fill: noop,
    closePath: noop, fillText: noop, set fillStyle(v) {}, set strokeStyle(v) {}, set lineWidth(v) {},
    set font(v) {}, canvas: { width: 600, height: 90 },
  };
};

/* ---------------- запуск app.js ---------------- */
const appJs = readFileSync(join(webDir, 'app.js'), 'utf8');
window.eval(appJs);
window.document.dispatchEvent(new window.Event('DOMContentLoaded', { bubbles: true }));

const $ = (s) => window.document.querySelector(s);
const click = async (sel) => {
  const n = $(sel);
  if (!n) { problems.push(`нет элемента ${sel}`); return; }
  n.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 1200));
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const results = [];
async function step(name, fn) {
  const before = problems.length;
  try {
    await fn();
    results.push(`${problems.length > before ? '✖' : '✔'} ${name}`);
  } catch (e) {
    problems.push(`${name}: ${e.stack || e}`);
    results.push(`✖ ${name} (exception)`);
  }
  if (problems.length > before) results[results.length - 1] += `  [+${problems.length - before} проблем]`;
}

await sleep(2500); // boot: header/status/voices

await step('вкладка «Синтез» загружена', async () => {
  const opts = $('#synth-voice').options.length;
  if (!opts) throw new Error('список голосов пуст');
  $('#synth-text').value = 'Первое предложение для проверки интерфейса. Второе предложение чуть длиннее. Третье!';
  $('#synth-text').dispatchEvent(new window.Event('input', { bubbles: true }));
});

await step('синтез через HTTP-стрим', async () => {
  $('#synth-format').value = 'pcm';
  $('#synth-stream').checked = true;
  await click('#btn-synth');
  await sleep(6000);
  const state = $('#m-state').textContent;
  if (!/готово|ошибка/.test(state)) throw new Error(`состояние не финальное: ${state}`);
  if (state.includes('ошибка')) throw new Error('синтез вернул ошибку: ' + $('#synth-error').textContent);
});

await step('вкладка «Профили»', async () => {
  await click('.tab[data-tab="profiles"]');
  await sleep(1500);
  if (!$('#profiles-list')) throw new Error('нет списка профилей');
});

await step('создание preset-профиля из GUI', async () => {
  $('#pf-kind').value = 'preset';
  $('#pf-kind').dispatchEvent(new window.Event('change', { bubbles: true }));
  $('#pf-name').value = 'GUI Test Voice ' + Date.now();
  $('#pf-speaker').value = 'vivian';
  $('#pf-instruct').value = 'Спокойно и тепло';
  await click('#btn-create-profile');
  await sleep(1500);
  const msg = $('#pf-msg').textContent;
  if (!/создан/.test(msg)) throw new Error('профиль не создан: ' + msg);
});

await step('вкладка «Система» + диагностика', async () => {
  await click('.tab[data-tab="system"]');
  await sleep(1000);
  await click('#btn-diag');
  await sleep(6000);
  const rows = window.document.querySelectorAll('#diag-checks .check-row').length;
  if (!rows) throw new Error('диагностика не отрисовала checks');
  const sum = $('#diag-summary').textContent.slice(0, 90);
  console.log('   диагностика:', sum, '| checks:', rows);
});

await step('вкладка «Настройки» (загрузка конфига)', async () => {
  await click('.tab[data-tab="settings"]');
  await sleep(1500);
  if (!$('#set-backend').value) throw new Error('backend не заполнен');
  $('#set-chunk-target').value = '160';
  await click('#btn-save-cfg');
  await sleep(1500);
  const msg = $('#cfg-msg').textContent;
  if (!/Сохранено/.test(msg)) throw new Error('конфиг не сохранён: ' + msg);
});

await step('вкладка «Журнал»', async () => {
  await click('.tab[data-tab="logs"]');
  await sleep(2500);
  const lines = window.document.querySelectorAll('#log-view .log-line').length;
  console.log('   строк журнала в DOM:', lines);
  if (lines < 1) throw new Error('журнал пуст (SSE/история не подтянулись)');
});

await step('WS-синтез', async () => {
  await click('.tab[data-tab="synth"]');
  await sleep(800);
  $('#synth-text').value = 'Проверка веб-сокета из интерфейса. Второй чанк тоже нужен.';
  await click('#btn-synth-ws');
  await sleep(7000);
  const st = $('#m-state').textContent;
  console.log('   WS состояние:', st, '| чанки:', $('#m-chunks').textContent);
});

console.log('\n=== РЕЗУЛЬТАТ ===');
for (const r of results) console.log(' ', r);
console.log('\nпроблем:', problems.length);
for (const p of problems.slice(0, 25)) console.log(' -', p);
process.exit(problems.length ? 1 : 0);
