import { Engine, DRUM_ROWS, wavBlob, detectPitch } from './engine.js';
import { autoLoop, autoDrums, vibe } from './auto.js';

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));

const SOURCES = [['mix', 'Full mix'], ['nodrums', 'No drums'], ['vocals', 'Vocals'], ['other', 'Melody'], ['bass', 'Bass'], ['drums', 'Drums']];
const PAD_KEYS = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k'];
const PIANO = ['a', 'w', 's', 'e', 'd', 'f', 't', 'g', 'y', 'h', 'u', 'j', 'k', 'o', 'l'];
const BLACK = new Set([1, 3, 6, 8, 10, 13]);
const NOTES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
const LENS = [[0.25, '¼'], [0.5, '½'], [1, '1'], [2, '2'], [4, '4']];
const KITS = [['song', 'Song'], ['dust', 'Dust'], ['808', '808'], ['soft', 'Soft']];
const DRUM_LABELS = ['K', 'S', 'H', 'O'];
const G = { gut: 26, rh: 17, gap: 10, dh: 18 };

const state = {
  song: null, source: null, file: null, buffer: null,
  bpm: 120, bars: 4, pads: [], sel: 0, notes: [],
  drums: { kit: 'dust', rows: {}, swing: 0, vol: 0.8, userKit: false },
  choke: true, gate: false, lenAll: true,
  keys: false, octave: 0, rec: false, seed: 1, drumVar: 0,
};
let engine = null, peaks = null, overviewCache = null, padEls = [], keyEls = [];
const held = new Map();
const undoStack = [];

// ---------------------------------------------------------------- helpers

const sharp = s => (s || '').replace('#', '♯');
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
let C = {};
function readTheme() { for (const k of ['bg', 'panel', 'pad', 'ink', 'mute', 'line', 'accent']) C[k] = css('--' + k); }

function fit(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  if (height) canvas.style.height = height + 'px';
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 2600);
}

function beatAt(t) {
  const b = state.song.beats;
  let lo = 0, hi = b.length - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (b[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo + (t - b[lo]) / (b[lo + 1] - b[lo]);
}

function timeAt(pos) {
  const b = state.song.beats, k = clamp(Math.floor(pos), 0, b.length - 2);
  return b[k] + (pos - k) * (b[k + 1] - b[k]);
}

const padEnd = p => Math.min(state.buffer.duration, timeAt(beatAt(p.start) + p.len));
const total = () => state.bars * state.song.beatsPerBar * state.song.stepsPerBeat;

function nearestSlice(t) {
  const s = state.song.slices;
  let lo = 0, hi = s.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (s[mid].t < t) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(s[lo - 1].t - t) < Math.abs(s[lo].t - t)) lo--;
  return lo;
}

function noteName(midi) {
  const n = Math.round(midi - state.song.tuning / 100);
  return NOTES[((n % 12) + 12) % 12] + (Math.floor(n / 12) - 1);
}

// ---------------------------------------------------------------- project state

function project() {
  const { bpm, bars, pads, sel, drums, choke, gate, lenAll } = state;
  return { bpm, bars, pads, sel, drums, choke, gate, lenAll, notes: state.notes.map(({ pad, step, len, semi }) => ({ pad, step, len, semi: semi || 0 })) };
}

function snap() {
  undoStack.push(JSON.stringify(project()));
  if (undoStack.length > 100) undoStack.shift();
}

function undo() {
  const s = undoStack.pop();
  if (!s) return;
  Object.assign(state, JSON.parse(s));
  refresh();
}

const storeKey = () => `autosampler:${state.song.id}:${state.source}`;
let saveTimer;
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(storeKey(), JSON.stringify(project())); } catch { /* storage full */ }
  }, 300);
}

function padFromSlice(sl, len) {
  return { start: sl.t, len, tune: 0, gain: 1, note: sl.note || null, midi: sl.midi ?? null, beat: sl.beat, chroma: sl.chroma, score: sl.score };
}

function defaultLen() {
  const beat = 60 / state.song.bpm, target = state.source === 'drums' ? 0.45 : 0.95;
  return LENS.map(l => l[0]).reduce((a, b) => (Math.abs(b * beat - target) < Math.abs(a * beat - target) ? b : a));
}

function initProject() {
  const song = state.song, len = defaultLen(), feel = vibe(song);
  Object.assign(state, {
    bpm: song.bpm, bars: 4, sel: 0, notes: [], choke: true, gate: false, lenAll: true,
    pads: Array.from({ length: 16 }, (_, i) => (song.pads[i] != null ? padFromSlice(song.slices[song.pads[i]], len) : null)),
    drums: { kit: feel.kit, rows: {}, swing: feel.swing, vol: 0.8, userKit: false },
  });
  state.drumVar = 0;
  fitRows();
  arrange(true, true);
}

function fitRows() {
  const n = total();
  for (const row of DRUM_ROWS) {
    const old = state.drums.rows[row] || [];
    state.drums.rows[row] = Array.from({ length: n }, (_, i) => (old.length ? old[i % old.length] || 0 : 0));
  }
}

function setBars(n) {
  if (n === state.bars) return;
  snap();
  const prev = total();
  state.bars = n;
  const now = total();
  fitRows();
  if (now > prev) {
    const base = state.notes.slice();
    for (let o = prev; o < now; o += prev) for (const x of base) if (x.step + o < now) state.notes.push({ ...x, step: x.step + o });
  } else {
    state.notes = state.notes.filter(x => x.step < now);
  }
  for (const x of state.notes) x.len = Math.min(x.len, now - x.step);
  refresh();
}

function arrange(loop, drums) {
  if (loop) {
    state.seed = (Math.random() * 1e9) | 0;
    state.notes = autoLoop(state, state.seed);
  }
  if (drums) {
    const d = autoDrums(state, (Math.random() * 1e9) | 0, state.drumVar++);
    state.drums.rows = d.rows;
    if (!state.drums.userKit) state.drums.kit = d.kit;
  }
}

// ---------------------------------------------------------------- waveform

function buildPeaks(buf) {
  const B = 256, n = Math.ceil(buf.length / B), mn = new Float32Array(n), mx = new Float32Array(n);
  const a = buf.getChannelData(0), b = buf.numberOfChannels > 1 ? buf.getChannelData(1) : a;
  let peak = 0;
  for (let i = 0; i < n; i++) {
    let lo = 1, hi = -1;
    const e = Math.min(buf.length, (i + 1) * B);
    for (let j = i * B; j < e; j++) {
      const v = (a[j] + b[j]) * 0.5;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    mn[i] = lo; mx[i] = hi;
    peak = Math.max(peak, hi, -lo);
  }
  return { B, mn, mx, a, b, sr: buf.sampleRate, peak: peak || 1 };
}

function range(t0, t1) {
  const p = peaks, s0 = Math.max(0, Math.floor(t0 * p.sr)), s1 = Math.min(p.a.length, Math.ceil(t1 * p.sr));
  if (s1 <= s0) return [0, 0];
  let lo = 1, hi = -1;
  if (s1 - s0 < p.B * 2) {
    for (let j = s0; j < s1; j++) {
      const v = (p.a[j] + p.b[j]) * 0.5;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  } else {
    for (let i = Math.floor(s0 / p.B), e = Math.ceil(s1 / p.B); i < e; i++) {
      if (p.mn[i] < lo) lo = p.mn[i];
      if (p.mx[i] > hi) hi = p.mx[i];
    }
  }
  return [lo, hi];
}

function drawWave(g, x, y, w, h, t0, t1, norm) {
  const dt = (t1 - t0) / w, cols = [];
  let peak = 0;
  for (let px = 0; px < w; px++) {
    const r = range(t0 + px * dt, t0 + (px + 1) * dt);
    cols.push(r);
    peak = Math.max(peak, r[1], -r[0]);
  }
  const k = (h / 2) / (norm ? Math.max(peak, 0.02) : peaks.peak);
  g.beginPath();
  cols.forEach(([lo, hi], px) => g.rect(x + px, y + h / 2 - hi * k, 1, Math.max(1, (hi - lo) * k)));
  g.fill();
}

// ---------------------------------------------------------------- overview

function drawOverview() {
  const c = $('overview');
  const [g, w, h] = fit(c);
  const dur = state.buffer.duration;
  if (!overviewCache || overviewCache.width !== c.width || overviewCache.theme !== C.ink) {
    overviewCache = document.createElement('canvas');
    overviewCache.width = c.width; overviewCache.height = c.height; overviewCache.theme = C.ink;
    const og = overviewCache.getContext('2d');
    og.scale(c.width / w, c.height / h);
    og.fillStyle = C.mute;
    og.globalAlpha = 0.75;
    drawWave(og, 0, 5, Math.floor(w), h - 10, 0, dur, false);
  }
  g.save();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(overviewCache, 0, 0);
  g.restore();
  state.pads.forEach((p, i) => {
    if (!p) return;
    const x = (p.start / dur) * w, sel = i === state.sel;
    if (sel) {
      g.fillStyle = C.accent;
      g.globalAlpha = 0.25;
      g.fillRect(x, 0, Math.max(2, ((padEnd(p) - p.start) / dur) * w), h);
      g.globalAlpha = 1;
    }
    g.fillStyle = sel ? C.accent : C.ink;
    g.fillRect(Math.round(x), sel ? 0 : h - 12, sel ? 1.5 : 1, sel ? h : 12);
  });
}

function overviewClick(e) {
  const c = $('overview'), r = c.getBoundingClientRect(), x = e.clientX - r.left;
  const dur = state.buffer.duration;
  let best = -1, bd = 6;
  state.pads.forEach((p, i) => {
    if (!p) return;
    const d = Math.abs((p.start / dur) * r.width - x);
    if (d < bd) { bd = d; best = i; }
  });
  if (best >= 0 && best !== state.sel) { select(best); tap(best); return; }
  const t = (x / r.width) * dur;
  snap();
  moveStart(state.sel, t, !e.altKey, 0.6);
  tap(state.sel);
}

// ---------------------------------------------------------------- pads

function moveStart(i, t, snapToHit, tolerance = 0.03) {
  const p = state.pads[i];
  if (!p) return;
  t = clamp(t, 0, state.buffer.duration - 0.1);
  let sl = null;
  if (snapToHit && state.song.slices.length) {
    const cand = state.song.slices[nearestSlice(t)];
    if (Math.abs(cand.t - t) <= tolerance) { sl = cand; t = cand.t; }
  }
  p.start = t;
  p.beat = sl ? sl.beat : beatAt(t + 0.003);
  if (sl) {
    Object.assign(p, { note: sl.note || null, midi: sl.midi ?? null, chroma: sl.chroma, score: sl.score });
  } else {
    clearTimeout(moveStart.timer);
    moveStart.timer = setTimeout(() => {
      const m = state.source === 'drums' ? null : detectPitch(state.buffer, p.start);
      p.midi = m;
      p.note = m == null ? null : noteName(m);
      renderPads();
      save();
    }, 250);
  }
  padChanged(i);
}

function padChanged(i) {
  drawPad(i);
  if (i === state.sel) { drawZoom(); renderControls(); }
  if (state.keys) renderPads();
  drawOverview();
  save();
}

function drawPad(i) {
  const el = padEls[i], p = state.pads[i];
  if (!el || state.keys) return;
  el.querySelector('.n').textContent = p ? sharp(p.note) || '' : '';
  const [g, w, h] = fit(el.querySelector('canvas'));
  if (!p) return;
  g.fillStyle = C.ink;
  drawWave(g, 8, 0, Math.floor(w - 16), h, p.start, padEnd(p), true);
}

function keyBase() {
  const p = state.pads[state.sel];
  const root = p && p.midi != null ? Math.round(p.midi - state.song.tuning / 100) : 60;
  return Math.floor(root / 12) * 12 + 12 * state.octave;
}

function keySemi(k) {
  const p = state.pads[state.sel];
  const exact = p && p.midi != null ? p.midi - state.song.tuning / 100 : 60;
  return keyBase() + k - exact;
}

function renderPads() {
  const box = $('pads');
  box.classList.toggle('keys', state.keys);
  box.textContent = '';
  padEls = []; keyEls = [];
  if (state.keys) {
    const base = keyBase(), p = state.pads[state.sel];
    const root = p && p.midi != null ? Math.round(p.midi - state.song.tuning / 100) : 60;
    PIANO.forEach((key, k) => {
      const b = document.createElement('button');
      const m = base + k;
      b.className = 'pad' + (BLACK.has(k) ? ' black' : '') + (m === root ? ' root' : '');
      b.innerHTML = `<span class="k">${key.toUpperCase()}</span><span class="n">${NOTES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}</span>`;
      b.addEventListener('pointerdown', e => { e.preventDefault(); b.setPointerCapture(e.pointerId); press('p' + k, state.sel, keySemi(k), b); });
      b.addEventListener('pointerup', () => lift('p' + k));
      b.addEventListener('pointercancel', () => lift('p' + k));
      box.append(b);
      keyEls.push(b);
    });
    return;
  }
  state.pads.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'pad' + (i === state.sel ? ' sel' : '') + (p ? '' : ' empty');
    b.innerHTML = `<span class="k">${PAD_KEYS[i].toUpperCase()}</span><span class="n"></span><canvas></canvas>`;
    b.addEventListener('pointerdown', e => { e.preventDefault(); b.setPointerCapture(e.pointerId); press('m' + i, i); select(i); });
    b.addEventListener('pointerup', () => lift('m' + i));
    b.addEventListener('pointercancel', () => lift('m' + i));
    box.append(b);
    padEls.push(b);
  });
  state.pads.forEach((_, i) => drawPad(i));
}

function select(i) {
  if (i === state.sel || !state.pads[i]) return;
  state.sel = i;
  padEls.forEach((el, k) => el.classList.toggle('sel', k === i));
  if (state.keys) renderPads();
  drawZoom(); renderControls(); drawOverview(); drawGrid();
}

function flash(el, sticky) {
  if (!el) return;
  el.classList.add('hit');
  if (!sticky) setTimeout(() => el.classList.remove('hit'), 110);
}

function oneShot(p) { return padEnd(p) - p.start; }

function tap(i) {
  const p = state.pads[i];
  if (!p || !engine) return;
  engine.resume();
  engine.hit(i, { dur: oneShot(p) });
  flash(padEls[i]);
}

function press(id, i, semi = 0, el = padEls[i]) {
  const p = state.pads[i];
  if (!p || held.has(id)) return;
  engine.resume();
  const v = engine.hit(i, { dur: oneShot(p), semi, hold: state.gate });
  flash(el, state.gate);
  let note = null;
  if (state.rec && engine.playing) note = record(i, semi);
  held.set(id, { v, el, note, t: engine.ctx.currentTime });
}

function lift(id) {
  const h = held.get(id);
  if (!h) return;
  held.delete(id);
  if (!state.gate) return;
  engine.release(h.v, 0, 0.03);
  h.el && h.el.classList.remove('hit');
  if (h.note) {
    const n = total(), len = Math.round((engine.ctx.currentTime - h.t) / engine.stepDur);
    h.note.len = clamp(len, 1, n - h.note.step);
    drawGrid();
    save();
  }
}

function record(i, semi) {
  const pos = engine.position();
  if (pos < 0) return null;
  const n = total(), spb = state.song.stepsPerBeat;
  const q = Math.round(pos) % n;
  snap();
  state.notes = state.notes.filter(x => !(x.pad === i && x.step === q));
  const note = { pad: i, step: q, len: state.gate ? 1 : clamp(Math.round(state.pads[i].len * spb), 1, n - q), semi };
  if (Math.round(pos) > pos) note.skipUntil = engine.ctx.currentTime + engine.stepDur * 0.9;
  state.notes.push(note);
  drawGrid();
  save();
  return note;
}

// ---------------------------------------------------------------- editor

let view = null;

function zoomView() {
  const p = state.pads[state.sel];
  const end = padEnd(p), span = end - p.start, m = Math.max(0.3, span * 0.4);
  return { t0: p.start - m, t1: end + m };
}

function drawZoom(keepView) {
  const p = state.pads[state.sel];
  const [g, w, h] = fit($('zoom'));
  if (!p) return;
  if (!keepView || !view) view = zoomView();
  const { t0, t1 } = view, X = t => ((t - t0) / (t1 - t0)) * w;
  const end = padEnd(p), xs = X(p.start), xe = X(end);
  const song = state.song;

  g.fillStyle = C.bg;
  g.globalAlpha = 0.9;
  g.beginPath();
  g.roundRect(xs, 0, Math.max(1, xe - xs), h, 4);
  g.fill();
  g.globalAlpha = 1;

  // beat grid
  const b0 = Math.floor(beatAt(Math.max(0, t0))), b1 = Math.ceil(beatAt(t1));
  for (let b = b0; b <= b1; b++) {
    const x = X(timeAt(b));
    if (x < 0 || x > w) continue;
    const down = (((b - song.phase) % song.beatsPerBar) + song.beatsPerBar) % song.beatsPerBar === 0;
    g.fillStyle = C.mute;
    g.globalAlpha = down ? 0.7 : 0.35;
    g.fillRect(Math.round(x), 0, 1, down ? 9 : 5);
  }
  g.globalAlpha = 1;

  // waveform: dim outside the slice, ink inside
  g.fillStyle = C.mute;
  g.globalAlpha = 0.55;
  drawWave(g, 0, 10, Math.floor(w), h - 20, t0, t1, true);
  g.globalAlpha = 1;
  g.save();
  g.beginPath();
  g.rect(xs, 0, xe - xs, h);
  g.clip();
  g.fillStyle = C.ink;
  drawWave(g, 0, 10, Math.floor(w), h - 20, t0, t1, true);
  g.restore();

  // detected hits
  g.fillStyle = C.mute;
  const s = song.slices;
  for (let i = nearestSlice(t0); i < s.length && s[i].t <= t1; i++) {
    g.globalAlpha = 0.3 + 0.6 * s[i].strength;
    g.fillRect(Math.round(X(s[i].t)), h - 5, 1, 5);
  }
  g.globalAlpha = 1;

  g.fillStyle = C.accent;
  g.fillRect(Math.round(xs) - 1, 0, 2, h);
  g.beginPath(); g.moveTo(xs, 0); g.lineTo(xs + 7, 0); g.lineTo(xs, 8); g.fill();
  g.fillStyle = C.ink;
  g.fillRect(Math.round(xe) - 1, 0, 1.5, h);
  g.beginPath(); g.moveTo(xe, h); g.lineTo(xe - 7, h); g.lineTo(xe, h - 8); g.fill();
}

function setupZoom() {
  const c = $('zoom');
  let drag = null;
  const T = e => {
    const r = c.getBoundingClientRect();
    return { x: e.clientX - r.left, w: r.width, t: view.t0 + ((e.clientX - r.left) / r.width) * (view.t1 - view.t0) };
  };
  c.addEventListener('pointerdown', e => {
    const p = state.pads[state.sel];
    if (!p || !view) return;
    engine.resume();
    const { x, w, t } = T(e);
    const X = tt => ((tt - view.t0) / (view.t1 - view.t0)) * w;
    const ds = Math.abs(X(p.start) - x), de = Math.abs(X(padEnd(p)) - x);
    c.setPointerCapture(e.pointerId);
    snap();
    if (de < 9 && de <= ds) drag = { kind: 'end' };
    else if (ds < 9) drag = { kind: 'start' };
    else if (t > p.start && t < padEnd(p)) { drag = { kind: 'none' }; undoStack.pop(); tap(state.sel); }
    else { drag = { kind: 'start' }; apply(e); }
  });
  const apply = e => {
    const p = state.pads[state.sel], { t, w } = T(e);
    const perPx = (view.t1 - view.t0) / w;
    if (drag.kind === 'start') {
      moveStartKeep(t, !e.altKey, perPx * 9);
    } else if (drag.kind === 'end') {
      let len = beatAt(t) - beatAt(p.start);
      // snap to quarter beats for long samples, slide freely for short ones
      if (!e.altKey && len >= 0.375) len = Math.round(len * 4) / 4;
      p.len = clamp(len, 0.03, 16);
      drawPad(state.sel); drawZoom(true); drawOverview(); renderControls();
    }
  };
  const moveStartKeep = (t, snapToHit, tol) => {
    const keep = view;
    moveStart(state.sel, t, snapToHit, tol);
    view = keep;
    drawZoom(true);
  };
  c.addEventListener('pointermove', e => { if (drag && drag.kind !== 'none') apply(e); });
  const finish = () => {
    if (!drag) return;
    const moved = drag.kind !== 'none';
    drag = null;
    if (moved) { drawZoom(); save(); tap(state.sel); }
  };
  c.addEventListener('pointerup', finish);
  c.addEventListener('pointercancel', finish);
}

function stepHit(dir, fine) {
  const p = state.pads[state.sel];
  if (!p) return;
  snap();
  if (fine) { moveStart(state.sel, p.start + dir * 0.002, false); return; }
  const s = state.song.slices;
  let i = nearestSlice(p.start);
  if (Math.abs(s[i].t - p.start) < 0.004) i += dir;
  else if ((s[i].t > p.start) !== (dir > 0)) i += dir;
  i = clamp(i, 0, s.length - 1);
  moveStart(state.sel, s[i].t, true);
  tap(state.sel);
}

function repick() {
  const p = state.pads[state.sel], song = state.song, spacing = 60 / song.bpm;
  if (!p) return;
  const taken = state.pads.filter(Boolean).map(x => x.start);
  const ranked = song.slices.filter(s => taken.every(t => Math.abs(t - s.t) > spacing)).sort((a, b) => b.score - a.score).slice(0, 40);
  if (!ranked.length) return;
  snap();
  const sl = ranked[Math.floor(Math.random() * Math.min(ranked.length, 12))];
  moveStart(state.sel, sl.t, true);
  tap(state.sel);
}

function dragNumber(el, { get, set, min, max, step = 1, px = 4, reset }) {
  el.addEventListener('pointerdown', e => {
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    const y0 = e.clientY, v0 = get();
    let moved = false;
    const move = ev => {
      const v = clamp(Math.round((v0 + Math.round((y0 - ev.clientY) / px) * step) / step) * step, min, max);
      if (v === get()) return;
      if (!moved) { snap(); moved = true; }
      set(v);
    };
    const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); el.removeEventListener('pointercancel', up); save(); };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  });
  if (reset) el.addEventListener('dblclick', () => { snap(); set(reset()); save(); });
}

function seg(el, options, current, onPick, disabled = () => false) {
  el.textContent = '';
  for (const [value, label] of options) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.className = value === current ? 'on' : '';
    b.disabled = disabled(value);
    b.addEventListener('click', () => { b.blur(); onPick(value); });
    el.append(b);
  }
}

function renderControls() {
  const p = state.pads[state.sel];
  seg($('len'), LENS, p ? p.len : null, v => {
    snap();
    (state.lenAll ? state.pads : [p]).forEach(x => { if (x) x.len = v; });
    refresh();
  });
  $('lenAll').classList.toggle('on', state.lenAll);
  $('tune').textContent = p ? (p.tune > 0 ? '+' : '') + p.tune : '';
  $('gain').textContent = p ? Math.round(p.gain * 100) : '';
  $('rev').classList.toggle('on', !!(p && p.rev));
  seg($('chop'), [[0, 'Off'], [2, '2'], [3, '3'], [4, '4'], [8, '8']], p ? p.chop || 0 : 0, v => { snap(); p.chop = v; renderControls(); save(); tap(state.sel); });
  $('gate').classList.toggle('on', state.gate);
  $('choke').classList.toggle('on', !state.choke);
  $('keys').classList.toggle('on', state.keys);
}

// ---------------------------------------------------------------- header

function fmtBpm(v) { return (Number.isInteger(v) ? String(v) : v.toFixed(1)); }

function renderHeader() {
  const song = state.song;
  $('title').textContent = song.title;
  document.title = song.title;
  $('bpm').textContent = fmtBpm(state.bpm);
  $('orig').textContent = (song.stable ? '' : '~') + fmtBpm(song.bpm) + ' →';
  $('orig').hidden = state.bpm === song.bpm && song.stable;
  const minor = song.key.endsWith('m');
  $('key').textContent = sharp(minor ? song.key.slice(0, -1) : song.key) + (minor ? ' minor' : ' major');
  $('meter').textContent = song.meter;
  seg($('sources'), SOURCES, state.source, v => { if (v !== state.source) openSong(state.file, v, song.title); });
  seg($('bars'), [1, 2, 4, 8].map(n => [n, String(n)]), state.bars, setBars);
  $('rec').classList.toggle('on', state.rec);
  $('play').classList.toggle('on', !!(engine && engine.playing));
  $('playIcon').setAttribute('d', engine && engine.playing ? 'M4 3.5h8v9H4z' : 'M4.5 2.8v10.4L13 8z');
}

function renderDrumBar() {
  const has = (state.song.drums.kit || []).length > 0;
  seg($('kit'), KITS.filter(k => k[0] !== 'song' || has), state.drums.kit, v => {
    snap();
    state.drums.kit = v;
    state.drums.userKit = true;
    renderDrumBar();
    save();
    if (!engine.playing) engine.drum('kick', engine.ctx.currentTime, 1);
  });
  $('swing').textContent = Math.round(state.drums.swing * 100);
  $('drumVol').textContent = Math.round(state.drums.vol * 100);
  engine.bus.drums.gain.value = state.drums.vol;
}

// ---------------------------------------------------------------- grid

function layout() {
  const c = $('grid'), w = c.clientWidth, n = total();
  // fit the whole grid on screen when the window allows it
  const room = window.innerHeight - (c.getBoundingClientRect().top + window.scrollY) - 20;
  G.rh = clamp(Math.floor((room - G.gap - 4 * G.dh) / 16), 12, 18);
  const drumTop = 16 * G.rh + G.gap;
  return { w, n, sw: (w - G.gut) / n, drumTop, h: drumTop + 4 * G.dh };
}

function drawGrid() {
  const L = layout();
  const [g, w] = fit($('grid'), L.h);
  const song = state.song, spb = song.stepsPerBeat, perBar = spb * song.beatsPerBar;
  const padH = 16 * G.rh;

  for (let bar = 0; bar < state.bars; bar++) {
    g.fillStyle = C.panel;
    g.globalAlpha = bar % 2 ? 0.45 : 1;
    const x = G.gut + bar * perBar * L.sw;
    g.beginPath(); g.roundRect(x + 1, 0, perBar * L.sw - 2, padH, 6); g.fill();
    g.beginPath(); g.roundRect(x + 1, L.drumTop, perBar * L.sw - 2, 4 * G.dh, 6); g.fill();
  }
  g.globalAlpha = 1;

  // selected row
  g.fillStyle = C.ink;
  g.globalAlpha = 0.05;
  g.fillRect(G.gut, state.sel * G.rh, w - G.gut, G.rh);
  g.globalAlpha = 1;

  // beat lines
  g.fillStyle = C.line;
  for (let s = spb; s < L.n; s += spb) {
    if (s % perBar === 0) continue;
    const x = Math.round(G.gut + s * L.sw);
    g.fillRect(x, 3, 1, padH - 6);
    g.fillRect(x, L.drumTop + 3, 1, 4 * G.dh - 6);
  }

  g.font = `500 ${G.rh < 14 ? 9 : 10}px ${css('--mono')}`;
  g.textBaseline = 'middle';
  g.textAlign = 'center';
  for (let i = 0; i < 16; i++) {
    g.fillStyle = i === state.sel ? C.ink : C.mute;
    g.globalAlpha = state.pads[i] ? 1 : 0.3;
    g.fillText(PAD_KEYS[i].toUpperCase(), G.gut / 2 - 2, i * G.rh + G.rh / 2 + 0.5);
  }
  g.globalAlpha = 1;
  g.fillStyle = C.mute;
  DRUM_LABELS.forEach((l, i) => g.fillText(l, G.gut / 2 - 2, L.drumTop + i * G.dh + G.dh / 2 + 0.5));

  for (const n of state.notes) {
    if (n.step >= L.n) continue;
    g.fillStyle = n.pad === state.sel ? C.accent : C.ink;
    g.beginPath();
    g.roundRect(G.gut + n.step * L.sw + 1, n.pad * G.rh + 2, Math.max(3, n.len * L.sw - 2), G.rh - 4, 3);
    g.fill();
  }

  const cell = Math.min(L.sw - 2, G.dh - 6), r = Math.min(3, cell / 3);
  DRUM_ROWS.forEach((row, i) => {
    const y = L.drumTop + i * G.dh + (G.dh - cell) / 2;
    for (let s = 0; s < L.n; s++) {
      const v = state.drums.rows[row][s], x = G.gut + s * L.sw + (L.sw - cell) / 2;
      if (v) { g.fillStyle = C.ink; g.globalAlpha = v >= 1 ? 1 : 0.4; }
      else { g.fillStyle = C.mute; g.globalAlpha = s % spb === 0 ? 0.22 : 0.1; }
      g.beginPath(); g.roundRect(x, y, cell, cell, r); g.fill();
    }
  });
  g.globalAlpha = 1;
}

function setupGrid() {
  const c = $('grid');
  let drag = null;
  const at = e => {
    const r = c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, L = layout();
    const step = Math.floor((x - G.gut) / L.sw);
    if (y < 16 * G.rh) return { kind: 'pad', row: clamp(Math.floor(y / G.rh), 0, 15), step, x, L };
    if (y >= L.drumTop) return { kind: 'drum', row: clamp(Math.floor((y - L.drumTop) / G.dh), 0, 3), step, x, L };
    return null;
  };
  const noteAt = (row, step) => state.notes.find(n => n.pad === row && step >= n.step && step < n.step + n.len);
  const room = n => {
    let lim = total() - n.step;
    for (const o of state.notes) if (o !== n && o.pad === n.pad && o.step > n.step) lim = Math.min(lim, o.step - n.step);
    return lim;
  };

  c.addEventListener('pointerdown', e => {
    const h = at(e);
    if (!h) return;
    engine.resume();
    if (h.x < G.gut) {
      if (h.kind === 'pad') { select(h.row); tap(h.row); }
      else engine.drum(DRUM_ROWS[h.row], engine.ctx.currentTime, 1);
      return;
    }
    if (h.step < 0 || h.step >= h.L.n) return;
    c.setPointerCapture(e.pointerId);
    snap();
    if (h.kind === 'drum') {
      const name = DRUM_ROWS[h.row], row = state.drums.rows[name];
      const val = row[h.step] ? 0 : e.shiftKey ? 0.5 : 1;
      row[h.step] = val;
      if (val && !engine.playing) engine.drum(name, engine.ctx.currentTime, val);
      drag = { kind: 'paint', row: name, val };
    } else {
      if (!state.pads[h.row]) { undoStack.pop(); return; }
      const n = noteAt(h.row, h.step);
      if (n) {
        const edge = G.gut + (n.step + n.len) * h.L.sw - h.x;
        drag = edge < 6 ? { kind: 'size', n } : { kind: 'move', n, from: h.step, orig: n.step, moved: false };
      } else {
        const note = { pad: h.row, step: h.step, len: 1, semi: 0 };
        state.notes.push(note);
        note.len = clamp(Math.round(state.pads[h.row].len * state.song.stepsPerBeat), 1, room(note));
        drag = { kind: 'size', n: note, fresh: true };
        select(h.row);
        if (!engine.playing) tap(h.row);
      }
    }
    drawGrid();
  });
  c.addEventListener('pointermove', e => {
    const h = at(e);
    if (!drag) {
      if (h && h.kind === 'pad' && h.x >= G.gut) {
        const n = noteAt(h.row, h.step);
        c.style.cursor = n && G.gut + (n.step + n.len) * h.L.sw - h.x < 6 ? 'ew-resize' : 'pointer';
      } else c.style.cursor = 'pointer';
      return;
    }
    const L = layout(), r = c.getBoundingClientRect();
    const step = clamp(Math.floor((e.clientX - r.left - G.gut) / L.sw), 0, L.n - 1);
    if (drag.kind === 'paint') state.drums.rows[drag.row][step] = drag.val;
    else if (drag.kind === 'size') {
      if (drag.fresh && step === drag.n.step && !drag.sized) return;
      drag.sized = true;
      drag.n.len = clamp(step - drag.n.step + 1, 1, room(drag.n));
    } else if (drag.kind === 'move') {
      const to = clamp(drag.orig + step - drag.from, 0, L.n - drag.n.len);
      if (to !== drag.n.step) { drag.n.step = to; drag.moved = true; }
    }
    drawGrid();
  });
  const finish = () => {
    if (!drag) return;
    if (drag.kind === 'move' && !drag.moved) state.notes = state.notes.filter(n => n !== drag.n);
    drag = null;
    drawGrid();
    save();
  };
  c.addEventListener('pointerup', finish);
  c.addEventListener('pointercancel', finish);
}

// ---------------------------------------------------------------- transport

let lastStep = -1;
function frame() {
  const head = $('head');
  if (!engine || !engine.playing) { head.hidden = true; lastStep = -1; return; }
  const pos = engine.position(), L = layout();
  if (pos >= 0) {
    head.hidden = false;
    head.style.transform = `translateX(${G.gut + pos * L.sw}px)`;
    const step = Math.floor(pos);
    if (step !== lastStep) {
      lastStep = step;
      if (!state.keys) for (const n of state.notes) if (n.step === step) flash(padEls[n.pad]);
    }
  }
  requestAnimationFrame(frame);
}

function togglePlay() {
  if (!engine) return;
  if (engine.playing) engine.stop();
  else { engine.play(); requestAnimationFrame(frame); }
  renderHeader();
}

function toggleRec() {
  state.rec = !state.rec;
  if (state.rec && !engine.playing) togglePlay();
  renderHeader();
}

function toggleKeys() {
  state.keys = !state.keys;
  state.octave = 0;
  for (const id of [...held.keys()]) lift(id);
  renderPads();
  renderControls();
}

async function exportLoop() {
  if (!state.notes.length && !DRUM_ROWS.some(r => state.drums.rows[r].some(Boolean))) return;
  const data = await engine.renderLoop();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(wavBlob(data));
  a.download = `${state.song.title} - ${state.bars} bar loop ${fmtBpm(state.bpm)}bpm.wav`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function refresh() {
  fitRows();
  state.notes = state.notes.filter(n => state.pads[n.pad] && n.step < total());
  if (!state.pads[state.sel]) state.sel = Math.max(0, state.pads.findIndex(Boolean));
  renderHeader();
  renderPads();
  renderControls();
  renderDrumBar();
  drawOverview();
  drawZoom();
  drawGrid();
  save();
}

// ---------------------------------------------------------------- library & loading

function progress(stage, frac) {
  $('progressStage').textContent = stage;
  $('progressBar').style.width = Math.round(frac * 100) + '%';
}

async function showLibrary() {
  if (engine && engine.playing) togglePlay();
  $('app').hidden = true;
  $('progress').hidden = true;
  $('library').hidden = false;
  document.title = 'Autosampler';
  localStorage.removeItem('autosampler:last');
  const list = await (await fetch('/api/library')).json();
  const ul = $('songs');
  ul.textContent = '';
  for (const s of list) {
    const li = document.createElement('li'), b = document.createElement('button');
    b.className = s.ready ? 'ready' : '';
    b.innerHTML = '<span></span><i></i>';
    b.firstChild.textContent = s.title;
    b.addEventListener('click', () => ask(s.file, s.title));
    li.append(b);
    ul.append(li);
  }
}

function ask(file, title) {
  const sheet = $('sheet');
  let choice = 'nodrums';
  $('sheetTitle').textContent = title;
  const draw = () => seg($('sheetSources'), SOURCES, choice, v => { choice = v; draw(); });
  draw();
  $('sheetCancel').onclick = () => sheet.close('cancel');
  sheet.onclose = () => { if (sheet.returnValue === 'go') openSong(file, choice, title); };
  sheet.returnValue = '';
  sheet.showModal();
  $('sheetGo').focus();
}

async function upload(f) {
  const body = new FormData();
  body.append('file', f);
  const r = await fetch('/api/upload', { method: 'POST', body });
  if (!r.ok) { toast('That file is not audio'); return; }
  const s = await r.json();
  await showLibrary();
  ask(s.file, s.title);
}

async function openSong(file, source, title) {
  if (engine && engine.playing) togglePlay();
  $('library').hidden = true;
  $('app').hidden = true;
  $('progress').hidden = false;
  $('progressTitle').textContent = title;
  progress('', 0.02);
  try {
    const start = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file, source }) });
    if (!start.ok) throw new Error('Song not found');
    const { id } = await start.json();
    let job;
    for (;;) {
      job = await (await fetch(`/api/jobs/${id}`)).json();
      if (job.done) break;
      progress(job.stage, job.progress * 0.9);
      await sleep(350);
    }
    if (job.error) throw new Error(job.error);
    const song = job.result;
    progress('Loading', 0.94);
    engine = engine || new Engine(state);
    await engine.ready;
    const [buffer] = await Promise.all([
      fetch(`/api/audio/${song.id}/${source}.flac`).then(r => r.arrayBuffer()).then(b => engine.ctx.decodeAudioData(b)),
      engine.loadSongKit(song.id, song.drums.kit || []),
    ]);
    Object.assign(state, { song, source, file, buffer, keys: false, rec: false });
    peaks = buildPeaks(buffer);
    overviewCache = null;
    undoStack.length = 0;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(storeKey())); } catch { /* fresh */ }
    $('app').hidden = false;
    if (saved && Array.isArray(saved.pads)) Object.assign(state, saved);
    else initProject();
    $('progress').hidden = true;
    refresh();
    localStorage.setItem('autosampler:last', JSON.stringify({ file, source, title }));
  } catch (err) {
    await showLibrary();
    toast(err.message || 'Analysis failed');
  }
}

// ---------------------------------------------------------------- wiring

function wire() {
  readTheme();
  setupZoom();
  setupGrid();

  $('title').addEventListener('click', showLibrary);
  $('play').addEventListener('click', e => { e.currentTarget.blur(); togglePlay(); });
  $('rec').addEventListener('click', e => { e.currentTarget.blur(); toggleRec(); });
  $('undo').addEventListener('click', e => { e.currentTarget.blur(); undo(); });
  $('export').addEventListener('click', e => { e.currentTarget.blur(); exportLoop(); });
  $('auto').addEventListener('click', e => {
    e.currentTarget.blur();
    snap();
    arrange(true, true);
    refresh();
    if (!engine.playing) togglePlay();
  });
  $('autoLoop').addEventListener('click', () => { snap(); arrange(true, false); refresh(); });
  $('autoDrums').addEventListener('click', () => { snap(); arrange(false, true); refresh(); });
  $('clearLoop').addEventListener('click', () => { snap(); state.notes = []; refresh(); });
  $('clearDrums').addEventListener('click', () => { snap(); state.drums.rows = {}; refresh(); });
  $('prev').addEventListener('click', () => stepHit(-1));
  $('next').addEventListener('click', () => stepHit(1));
  $('repick').addEventListener('click', repick);
  $('rev').addEventListener('click', () => { const p = state.pads[state.sel]; if (!p) return; snap(); p.rev = !p.rev; renderControls(); save(); tap(state.sel); });
  $('lenAll').addEventListener('click', () => { state.lenAll = !state.lenAll; renderControls(); save(); });
  $('gate').addEventListener('click', () => { state.gate = !state.gate; renderControls(); save(); });
  $('choke').addEventListener('click', () => { state.choke = !state.choke; renderControls(); save(); });
  $('keys').addEventListener('click', e => { e.currentTarget.blur(); toggleKeys(); });
  $('overview').addEventListener('pointerdown', e => { engine.resume(); overviewClick(e); });

  const sel = () => state.pads[state.sel];
  dragNumber($('bpm'), { get: () => state.bpm, set: v => { state.bpm = v; renderHeader(); }, min: 40, max: 240, step: 1, reset: () => state.song.bpm });
  dragNumber($('tune'), { get: () => sel().tune, set: v => { sel().tune = v; renderControls(); tap(state.sel); }, min: -24, max: 24, reset: () => 0 });
  dragNumber($('gain'), { get: () => Math.round(sel().gain * 100), set: v => { sel().gain = v / 100; renderControls(); }, min: 0, max: 150, step: 5, reset: () => 100 });
  dragNumber($('swing'), { get: () => Math.round(state.drums.swing * 100), set: v => { state.drums.swing = v / 100; renderDrumBar(); }, min: 0, max: 60, step: 2, reset: () => 0 });
  dragNumber($('drumVol'), { get: () => Math.round(state.drums.vol * 100), set: v => { state.drums.vol = v / 100; renderDrumBar(); }, min: 0, max: 120, step: 5, reset: () => 80 });

  $('add').addEventListener('click', () => $('file').click());
  $('file').addEventListener('change', e => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ''; });
  window.addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('drag'); });
  window.addEventListener('dragleave', () => document.body.classList.remove('drag'));
  window.addEventListener('drop', e => {
    e.preventDefault();
    document.body.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });

  window.addEventListener('keydown', e => {
    if ($('app').hidden || $('sheet').open) return;
    const code = e.code, k = code.startsWith('Key') ? code.slice(3).toLowerCase() : code;
    if (e.metaKey || e.ctrlKey) {
      if (k === 'z') { e.preventDefault(); undo(); }
      return;
    }
    if (e.altKey) return;
    if (k === 'Space') { e.preventDefault(); if (!e.repeat) togglePlay(); return; }
    if (k === 'Tab') { e.preventDefault(); if (!e.repeat) toggleKeys(); return; }
    if (k === 'Enter') { e.preventDefault(); if (!e.repeat) toggleRec(); return; }
    if (k === 'ArrowLeft' || k === 'ArrowRight') { e.preventDefault(); stepHit(k === 'ArrowLeft' ? -1 : 1, e.shiftKey); return; }
    if (e.repeat) return;
    if (state.keys) {
      const i = PIANO.indexOf(k);
      if (i >= 0) press('k' + k, state.sel, keySemi(i), keyEls[i]);
      else if (k === 'z' || k === 'x') { state.octave = clamp(state.octave + (k === 'z' ? -1 : 1), -3, 3); renderPads(); }
      return;
    }
    const i = PAD_KEYS.indexOf(k);
    if (i >= 0) { press('k' + k, i); select(i); }
  });
  window.addEventListener('keyup', e => {
    const k = e.code.startsWith('Key') ? e.code.slice(3).toLowerCase() : e.code;
    lift('k' + k);
  });
  window.addEventListener('blur', () => { for (const id of [...held.keys()]) lift(id); held.clear(); });

  let raf;
  new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      if ($('app').hidden || !state.buffer) return;
      state.pads.forEach((_, i) => drawPad(i));
      drawOverview(); drawZoom(); drawGrid();
    });
  }).observe($('app'));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readTheme(); if (!$('app').hidden) refresh(); });
}

async function boot() {
  wire();
  let last = null;
  try { last = JSON.parse(localStorage.getItem('autosampler:last')); } catch { /* none */ }
  if (last) {
    const list = await (await fetch('/api/library')).json();
    if (list.some(s => s.file === last.file && s.ready)) { $('library').hidden = true; openSong(last.file, last.source, last.title); return; }
  }
  showLibrary();
}

window.autosampler = { state, get engine() { return engine; } };
boot();
