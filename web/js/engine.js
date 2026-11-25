// Audio engine: zero-latency pad voices, look-ahead sequencer, drum kits, offline export.

const SR = 44100;
export const DRUM_ROWS = ['kick', 'snare', 'hat', 'open'];

function softClip(ctx) {
  const ws = ctx.createWaveShaper();
  const n = 2048, c = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1, a = Math.abs(x);
    c[i] = a < 0.6 ? x : Math.sign(x) * (0.6 + 0.4 * Math.tanh((a - 0.6) / 0.4));
  }
  ws.curve = c;
  return ws;
}

function drive(ctx, amount) {
  const ws = ctx.createWaveShaper();
  const n = 1024, c = new Float32Array(n), k = Math.tanh(amount);
  for (let i = 0; i < n; i++) c[i] = Math.tanh(((i / (n - 1)) * 2 - 1) * amount) / k;
  ws.curve = c;
  return ws;
}

function noise(ctx, dur) {
  const b = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
  const d = b.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  const s = ctx.createBufferSource();
  s.buffer = b;
  return s;
}

function env(param, points) {
  param.setValueAtTime(points[0][1], points[0][0]);
  for (let i = 1; i < points.length; i++) {
    const [t, v, exp] = points[i];
    if (exp) param.exponentialRampToValueAtTime(Math.max(v, 1e-4), t);
    else param.linearRampToValueAtTime(v, t);
  }
}

function filter(ctx, type, freq, q = 0.7) {
  const f = ctx.createBiquadFilter();
  f.type = type; f.frequency.value = freq; f.Q.value = q;
  return f;
}

async function render(dur, build) {
  const ctx = new OfflineAudioContext(1, Math.ceil(SR * dur), SR);
  build(ctx, ctx.destination);
  const buf = await ctx.startRendering();
  const d = buf.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
  const g = 0.9 / (peak || 1), fade = Math.min(d.length, 400);
  for (let i = 0; i < d.length; i++) d[i] *= g;
  for (let i = 0; i < fade; i++) d[d.length - 1 - i] *= i / fade;
  return buf;
}

const kick = ({ f0, f1, sweep, decay, click = 0.5, amount = 1.6, lp = 4000 }) => (ctx, out) => {
  const o = ctx.createOscillator(), g = ctx.createGain();
  env(o.frequency, [[0, f0], [sweep, f1, true]]);
  env(g.gain, [[0, 0], [0.0015, 1], [decay * 0.25, 0.7, true], [decay, 0.0005, true]]);
  const sh = drive(ctx, amount), f = filter(ctx, 'lowpass', lp);
  o.connect(g).connect(sh).connect(f).connect(out);
  o.start(0);
  if (click > 0) {
    const n = noise(ctx, 0.02), ng = ctx.createGain(), bp = filter(ctx, 'bandpass', 2600, 0.8);
    env(ng.gain, [[0, click], [0.012, 0.0005, true]]);
    n.connect(bp).connect(ng).connect(f);
    n.start(0);
  }
};

const snare = ({ tones = [185, 330], toneDecay = 0.1, toneLevel = 0.6, hp = 900, bp = 3200, noiseDecay = 0.19, attack = 0.001, amount = 1.4, lp = 11000 }) => (ctx, out) => {
  const sh = drive(ctx, amount), f = filter(ctx, 'lowpass', lp);
  sh.connect(f).connect(out);
  tones.forEach((hz, i) => {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'triangle';
    env(o.frequency, [[0, hz * 1.4], [0.02, hz, true]]);
    env(g.gain, [[0, toneLevel / (i + 1)], [toneDecay, 0.0005, true]]);
    o.connect(g).connect(sh);
    o.start(0);
  });
  const n = noise(ctx, noiseDecay + 0.05), g = ctx.createGain();
  env(g.gain, [[0, 0], [attack, 0.9], [noiseDecay * 0.35, 0.3, true], [noiseDecay, 0.0005, true]]);
  n.connect(filter(ctx, 'highpass', hp)).connect(filter(ctx, 'peaking', bp, 0.8)).connect(g).connect(sh);
  n.start(0);
};

const hat = ({ decay = 0.05, hp = 7500, metallic = false, attack = 0.001, bp = 10000 }) => (ctx, out) => {
  const g = ctx.createGain();
  env(g.gain, [[0, 0], [attack, 1], [attack + decay * 0.3, 0.3, true], [attack + decay, 0.0005, true]]);
  const h = filter(ctx, 'highpass', hp), b = filter(ctx, 'bandpass', bp, 0.6);
  b.connect(h).connect(g).connect(out);
  if (metallic) {
    for (const hz of [205.3, 304.4, 369.6, 522.7, 540, 800]) {
      const o = ctx.createOscillator();
      o.type = 'square'; o.frequency.value = hz * 4.1;
      const og = ctx.createGain(); og.gain.value = 0.18;
      o.connect(og).connect(b);
      o.start(0);
    }
  }
  const n = noise(ctx, attack + decay + 0.05), ng = ctx.createGain();
  ng.gain.value = metallic ? 0.35 : 1;
  n.connect(ng).connect(b);
  n.start(0);
};

const KIT_DEFS = {
  dust: {
    kick: [0.42, kick({ f0: 125, f1: 47, sweep: 0.07, decay: 0.34, click: 0.45, amount: 2.2, lp: 3200 })],
    snare: [0.3, snare({ tones: [178, 286], toneDecay: 0.11, hp: 650, bp: 2400, noiseDecay: 0.2, amount: 2.0, lp: 8500 })],
    hat: [0.09, hat({ decay: 0.045, hp: 6500, bp: 9000 })],
    open: [0.4, hat({ decay: 0.3, hp: 6000, bp: 8500 })],
  },
  '808': {
    kick: [0.9, kick({ f0: 150, f1: 43, sweep: 0.09, decay: 0.8, click: 0.25, amount: 1.3, lp: 2500 })],
    snare: [0.28, snare({ tones: [174, 331], toneDecay: 0.09, toneLevel: 0.8, hp: 1400, bp: 4200, noiseDecay: 0.17, amount: 1.1 })],
    hat: [0.08, hat({ decay: 0.04, hp: 8000, metallic: true })],
    open: [0.5, hat({ decay: 0.4, hp: 7500, metallic: true })],
  },
  soft: {
    kick: [0.4, kick({ f0: 92, f1: 50, sweep: 0.05, decay: 0.3, click: 0.08, amount: 1.1, lp: 900 })],
    snare: [0.36, snare({ tones: [210], toneDecay: 0.05, toneLevel: 0.25, hp: 1100, bp: 2800, noiseDecay: 0.26, attack: 0.006, amount: 0.8, lp: 7000 })],
    hat: [0.07, hat({ decay: 0.03, hp: 9000, attack: 0.002, bp: 11000 })],
    open: [0.2, hat({ decay: 0.12, hp: 5000, attack: 0.02, bp: 6500 })],
  },
};

const GAINS = { kick: 1.0, snare: 0.8, hat: 0.34, open: 0.3 };

export class Engine {
  constructor(state) {
    this.state = state;
    this.ctx = new AudioContext({ latencyHint: 'interactive', sampleRate: SR });
    this.bus = this.master(this.ctx, this.ctx.destination);
    this.voices = new Set();
    this.kits = {};
    this.openHat = null;
    this.playing = false;
    this.queue = [];
    this.onStep = null;
    this.ready = this.buildKits();
    const src = 'let t;onmessage=e=>{clearInterval(t);if(e.data)t=setInterval(()=>postMessage(0),e.data)}';
    this.clock = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    this.clock.onmessage = () => this.tick();
  }

  master(ctx, dest) {
    const sample = ctx.createGain(), drums = ctx.createGain(), sum = ctx.createGain(), clip = softClip(ctx);
    sum.gain.value = 0.85;
    sample.connect(sum); drums.connect(sum);
    sum.connect(clip).connect(dest);
    // short synthetic room on the drum bus
    const len = Math.floor(ctx.sampleRate * 0.28), ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = ir.getChannelData(ch);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2);
    }
    const room = ctx.createConvolver(), wet = ctx.createGain(), send = ctx.createGain();
    room.buffer = ir; wet.gain.value = 0.1;
    send.connect(drums);
    send.connect(room).connect(wet).connect(drums);
    return { sample, drums, drumIn: send };
  }

  async buildKits() {
    for (const [name, def] of Object.entries(KIT_DEFS)) {
      const kit = {};
      await Promise.all(Object.entries(def).map(async ([k, [dur, build]]) => { kit[k] = await render(dur, build); }));
      this.kits[name] = kit;
    }
  }

  async loadSongKit(id, names) {
    const kit = {};
    await Promise.all(names.map(async n => {
      try {
        const r = await fetch(`/api/audio/${id}/kit_${n}.wav`);
        kit[n] = await this.ctx.decodeAudioData(await r.arrayBuffer());
      } catch { /* falls back to dust */ }
    }));
    this.kits.song = kit;
  }

  resume() { if (this.ctx.state !== 'running') this.ctx.resume(); }

  // ---- voices

  voice(ctx, dest, buffer, { when = 0, offset, dur = null, rate = 1, gain = 1, chop = 0 }) {
    const t = Math.max(when, ctx.currentTime);
    const src = ctx.createBufferSource(), g = ctx.createGain();
    src.buffer = buffer;
    src.playbackRate.value = rate;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.002);
    let tail = g;
    if (chop > 0) {
      // rhythmic gate: half of every division is silenced, with short ramps
      const c = ctx.createGain(), span = dur == null ? 8 : dur / rate;
      for (let a = t; a < t + span; a += chop) {
        c.gain.setValueAtTime(1, a);
        c.gain.setValueAtTime(1, a + chop * 0.5);
        c.gain.linearRampToValueAtTime(0, a + chop * 0.5 + 0.004);
        c.gain.setValueAtTime(0, a + chop - 0.003);
        c.gain.linearRampToValueAtTime(1, a + chop);
      }
      g.connect(c);
      tail = c;
    }
    tail.connect(dest);
    src.connect(g);
    const v = { src, g, released: false, start: t };
    if (dur == null) {
      src.start(t, offset);
    } else {
      const d = Math.max(0.02, Math.min(dur, buffer.duration - offset));
      const end = t + d / rate;
      src.start(t, offset, d + 0.05);
      g.gain.setValueAtTime(gain, Math.max(t + 0.002, end - 0.014));
      g.gain.linearRampToValueAtTime(0, end);
      src.stop(end + 0.01);
    }
    return v;
  }

  release(v, when = 0, tc = 0.012) {
    if (v.released) return;
    v.released = true;
    const t = Math.max(when, this.ctx.currentTime, v.start + 0.003);
    if (v.g.gain.cancelAndHoldAtTime) v.g.gain.cancelAndHoldAtTime(t);
    else v.g.gain.cancelScheduledValues(t);
    v.g.gain.setTargetAtTime(0, t, tc);
    try { v.src.stop(t + tc * 8); } catch { /* already stopped */ }
  }

  rate(pad, semi = 0) {
    const s = this.state;
    return (s.bpm / s.song.bpm) * Math.pow(2, ((pad.tune || 0) + semi) / 12);
  }

  reversed(start, dur) {
    const src = this.state.buffer;
    if (this.revFor !== src) { this.revFor = src; this.revCache = new Map(); }
    const key = start.toFixed(4) + ':' + dur.toFixed(3);
    let b = this.revCache.get(key);
    if (!b) {
      const s0 = Math.floor(start * src.sampleRate), n = Math.max(64, Math.min(Math.floor(dur * src.sampleRate), src.length - s0));
      b = this.ctx.createBuffer(src.numberOfChannels, n, src.sampleRate);
      for (let ch = 0; ch < src.numberOfChannels; ch++) {
        const a = src.getChannelData(ch), o = b.getChannelData(ch);
        for (let i = 0; i < n; i++) o[i] = a[s0 + n - 1 - i];
      }
      if (this.revCache.size > 200) this.revCache.clear();
      this.revCache.set(key, b);
    }
    return b;
  }

  // everything a voice needs for this pad, including reverse and chop effects
  spec(pad, dur, semi, hold) {
    const s = this.state, rate = this.rate(pad, semi);
    const o = { offset: pad.start, dur: hold ? null : dur, rate, gain: pad.gain ?? 1, chop: pad.chop ? 60 / s.bpm / pad.chop : 0, buffer: s.buffer };
    if (pad.rev) { o.buffer = this.reversed(pad.start, dur); o.offset = 0; o.dur = dur; }
    return o;
  }

  // live or sequenced pad hit. dur in source seconds; hold = play until released
  hit(padIndex, { when = 0, dur, semi = 0, hold = false } = {}) {
    const s = this.state, pad = s.pads[padIndex];
    if (!pad || !s.buffer) return null;
    if (s.choke) for (const o of this.voices) this.release(o, when, 0.006);
    const o = this.spec(pad, dur, semi, hold);
    const v = this.voice(this.ctx, this.bus.sample, o.buffer, { when, ...o });
    this.voices.add(v);
    v.src.onended = () => { this.voices.delete(v); v.g.disconnect(); };
    return v;
  }

  drum(row, when, vel, ctx = this.ctx, bus = this.bus) {
    const s = this.state;
    const kit = this.kits[s.drums.kit] || {};
    const buf = kit[row] || this.kits.dust[row];
    if (!buf) return;
    const src = ctx.createBufferSource(), g = ctx.createGain();
    src.buffer = buf;
    const level = (s.drums.kit === 'song' && kit[row] ? { kick: 0.9, snare: 0.85, hat: 0.6 }[row] : GAINS[row]) * vel * vel;
    g.gain.setValueAtTime(level, when);
    src.connect(g).connect(bus.drumIn);
    src.start(when);
    if (ctx === this.ctx) {
      if ((row === 'hat' || row === 'open') && this.openHat) {
        this.openHat.gain.setTargetAtTime(0, when, 0.008);
        this.openHat = null;
      }
      if (row === 'open') this.openHat = g;
    }
    src.onended = () => g.disconnect();
  }

  // ---- sequencer

  get stepDur() { return 60 / this.state.bpm / this.state.song.stepsPerBeat; }
  get total() { const s = this.state; return s.bars * s.song.beatsPerBar * s.song.stepsPerBeat; }

  swingOffset(step) {
    const spb = this.state.song.stepsPerBeat;
    if (spb % 2 || step % 2 === 0) return 0;
    return this.state.drums.swing * this.stepDur;
  }

  noteDur(pad, note) {
    // source seconds so that the audible length equals the note length at the loop tempo
    return note.len * this.stepDur * this.rate(pad, note.semi || 0);
  }

  schedule(step, t) {
    const s = this.state;
    for (const n of s.notes) {
      if (n.step !== step || (n.skipUntil && t < n.skipUntil)) continue;
      const pad = s.pads[n.pad];
      if (pad) this.hit(n.pad, { when: t, dur: this.noteDur(pad, n), semi: n.semi || 0 });
    }
    const d = s.drums, off = this.swingOffset(step);
    for (const row of DRUM_ROWS) {
      const v = d.rows[row][step];
      if (v) this.drum(row, t + off, v);
    }
  }

  tick() {
    if (!this.playing) return;
    const now = this.ctx.currentTime;
    while (this.nextTime < now + 0.11) {
      if (this.step >= this.total) this.step = 0;
      this.schedule(this.step, this.nextTime);
      this.queue.push({ step: this.step, time: this.nextTime });
      this.nextTime += this.stepDur;
      this.step = (this.step + 1) % this.total;
    }
    while (this.queue.length > 1 && this.queue[1].time <= now) this.queue.shift();
  }

  play() {
    this.resume();
    this.playing = true;
    this.step = 0;
    this.nextTime = this.ctx.currentTime + 0.06;
    this.queue = [];
    this.bus.drums.gain.value = this.state.drums.vol;
    this.clock.postMessage(25);
    this.tick();
  }

  stop() {
    this.playing = false;
    this.clock.postMessage(0);
    for (const v of this.voices) this.release(v, 0, 0.02);
    this.queue = [];
  }

  // fractional loop position in steps, or -1
  position() {
    if (!this.playing || !this.queue.length) return -1;
    const now = this.ctx.currentTime, q = this.queue[0];
    if (now < q.time) return q.step === 0 ? -1 : q.step;
    return (q.step + Math.min(1, (now - q.time) / this.stepDur)) % this.total;
  }

  // ---- export

  async renderLoop() {
    const s = this.state, total = this.total, sd = this.stepDur, loop = total * sd;
    const ctx = new OfflineAudioContext(2, Math.ceil(SR * loop * 2), SR);
    const bus = this.master(ctx, ctx.destination);
    bus.drums.gain.value = s.drums.vol;
    const events = [];
    for (let pass = 0; pass < 2; pass++) {
      for (const n of s.notes) if (n.step < total && s.pads[n.pad]) events.push({ t: pass * loop + n.step * sd, n });
      for (let st = 0; st < total; st++) {
        for (const row of DRUM_ROWS) {
          const v = s.drums.rows[row][st];
          if (v) this.drum(row, pass * loop + st * sd + this.swingOffset(st), v, ctx, bus);
        }
      }
    }
    events.sort((a, b) => a.t - b.t);
    events.forEach((e, i) => {
      const pad = s.pads[e.n.pad], rate = this.rate(pad, e.n.semi || 0);
      let dur = this.noteDur(pad, e.n);
      const next = events[i + 1];
      if (s.choke && next) dur = Math.min(dur, Math.max(0.02, (next.t - e.t) * rate));
      const o = this.spec(pad, dur, e.n.semi || 0, false);
      this.voice(ctx, bus.sample, o.buffer, { when: e.t, ...o });
    });
    const out = await ctx.startRendering();
    const n = Math.round(SR * loop);
    return [0, 1].map(ch => out.getChannelData(ch).slice(n, n * 2));
  }
}

export function wavBlob(channels, sr = SR) {
  const n = channels[0].length, buf = new ArrayBuffer(44 + n * 4), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 4, true); str(8, 'WAVEfmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, n * 4, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 2; ch++) {
      const x = Math.max(-1, Math.min(1, channels[ch][i]));
      v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7fff, true);
      o += 2;
    }
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// YIN pitch estimate for hand-placed sample starts. Returns fractional MIDI or null.
export function detectPitch(buffer, start) {
  const sr = buffer.sampleRate, a = buffer.getChannelData(0), b = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : a;
  const W = 2048, maxTau = Math.floor(sr / 55), minTau = Math.floor(sr / 1400);
  const found = [];
  for (let f = 0; f < 6; f++) {
    const s0 = Math.floor((start + 0.03 + f * 0.045) * sr);
    if (s0 + W + maxTau >= a.length) break;
    const d = new Float32Array(maxTau + 1);
    for (let tau = 1; tau <= maxTau; tau++) {
      let sum = 0;
      for (let i = 0; i < W; i += 2) {
        const x = a[s0 + i] + b[s0 + i] - a[s0 + i + tau] - b[s0 + i + tau];
        sum += x * x;
      }
      d[tau] = sum;
    }
    let run = 0, tau = -1;
    for (let t = 1; t <= maxTau; t++) {
      run += d[t];
      d[t] = run ? d[t] * t / run : 1;
    }
    for (let t = minTau; t < maxTau; t++) {
      if (d[t] < 0.15) {
        while (t + 1 < maxTau && d[t + 1] < d[t]) t++;
        tau = t;
        break;
      }
    }
    if (tau < 0) continue;
    const y0 = d[tau - 1], y1 = d[tau], y2 = d[tau + 1], den = y0 - 2 * y1 + y2;
    const exact = tau + (den ? 0.5 * (y0 - y2) / den : 0);
    found.push(69 + 12 * Math.log2(sr / exact / 440));
  }
  if (found.length < 3) return null;
  found.sort((x, y) => x - y);
  const med = found[found.length >> 1];
  return found.filter(m => Math.abs(m - med) < 0.7).length >= 3 ? med : null;
}
