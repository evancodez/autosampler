// Automatic arrangement: sample loop and drum pattern generators.

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = (r, list) => list[Math.floor(r() * list.length)];

function weighted(r, items, weight) {
  const w = items.map(weight), sum = w.reduce((a, b) => a + b, 0);
  let x = r() * sum;
  for (let i = 0; i < items.length; i++) { x -= w[i]; if (x <= 0) return items[i]; }
  return items[items.length - 1];
}

function sim(a, b) {
  if (!a || !b) return 0.5;
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < 12; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na * nb) || 1);
}

// bar rhythms as segment lengths in beats
const SHAPES = {
  2: [[1, 1], [1, 1], [1, 1], [2], [1, 0.5, 0.5], [1, 2 / 3, 1 / 3], [2 / 3, 2 / 3, 2 / 3]],
  3: [[3], [2, 1], [2, 1], [1, 1, 1], [1, 2], [2, 0.5, 0.5]],
  4: [[2, 2], [2, 2], [2, 1, 1], [2, 1, 1], [1, 1, 2], [1, 1, 1, 1], [1.5, 1.5, 1], [2, 1, 0.5, 0.5], [3, 1], [1.5, 1, 1.5]],
  6: [[3, 3], [2, 2, 2], [3, 2, 1], [4, 2], [3, 1, 1, 1]],
};

function shapesFor(beats, spb) {
  const list = SHAPES[beats] || [Array(beats).fill(1)];
  return list.filter(shape => {
    let pos = 0;
    return shape.every(len => {
      const whole = v => Math.abs(v - Math.round(v)) < 1e-6;
      const ok = whole(pos * spb) && whole(len * spb);
      pos += len;
      return ok;
    });
  });
}

/** Build a loop from the pads: a motif bar, answered and varied across the loop. */
export function autoLoop(state, seed) {
  const r = rng(seed);
  const { song, pads, bars } = state;
  const spb = song.stepsPerBeat, B = song.beatsPerBar;
  const info = pads.map((p, i) => {
    if (!p) return null;
    const frac = p.beat - Math.round(p.beat);
    const beatIdx = Math.round(p.beat) - song.phase;
    return { i, onBeat: Math.abs(frac) < 0.09, down: Math.abs(frac) < 0.09 && ((beatIdx % B) + B) % B === 0, chroma: p.chroma, score: p.score ?? 0.5 };
  }).filter(Boolean);
  if (!info.length) return [];
  let on = info.filter(p => p.onBeat);
  if (on.length < 3) on = info;

  const home = weighted(r, on, p => Math.pow(p.score, 3) * (p.down ? 1.6 : 1));
  const ranked = on.filter(p => p !== home).sort((a, b) => sim(b.chroma, home.chroma) * (0.6 + b.score) - sim(a.chroma, home.chroma) * (0.6 + a.score));
  const pool = [home, ...ranked.slice(0, 2 + Math.floor(r() * 3))];
  const shapes = shapesFor(B, spb);

  const makeBar = (first) => {
    const shape = pick(r, shapes);
    const segs = [];
    let prev = null;
    shape.forEach((len, k) => {
      let p;
      if (k === 0 && first) p = first;
      else if (len === 0.5 && prev && shape[k - 1] === 0.5) p = prev;
      else {
        const options = pool.filter(x => x !== prev);
        p = options.length ? weighted(r, options, x => 0.4 + x.score) : pool[0];
      }
      segs.push({ pad: p.i, len });
      prev = p;
    });
    return segs;
  };

  const vary = (bar, depth) => {
    const out = bar.map(s => ({ ...s }));
    const n = Math.min(out.length, depth);
    for (let k = out.length - n; k < out.length; k++) {
      const options = pool.filter(x => x.i !== out[k].pad);
      if (options.length) out[k].pad = pick(r, options).i;
    }
    if (depth > 1 && r() < 0.6) {
      const last = out[out.length - 1];
      if (last.len >= 1 && Number.isInteger(last.len * spb / 2)) {
        out.pop();
        out.push({ pad: last.pad, len: last.len / 2 }, { pad: last.pad, len: last.len / 2 });
      }
    }
    return out;
  };

  const A = makeBar(home);
  const Bb = r() < 0.5 ? vary(A, 1) : makeBar(home);
  const C = vary(A, 2);
  const D = vary(Bb, 2);
  const plan = { 1: [A], 2: [A, Bb], 4: [A, Bb, A, C], 8: [A, Bb, A, C, A, Bb, A, D] }[bars] || Array.from({ length: bars }, (_, i) => [A, Bb, A, C][i % 4]);

  const notes = [];
  plan.forEach((bar, bi) => {
    let pos = 0;
    for (const seg of bar) {
      notes.push({ pad: seg.pad, step: Math.round((bi * B + pos) * spb), len: Math.max(1, Math.round(seg.len * spb)) });
      pos += seg.len;
    }
  });
  return notes;
}

// ---------------------------------------------------------------- drums

function emptyRows(total) {
  return { kick: Array(total).fill(0), snare: Array(total).fill(0), hat: Array(total).fill(0), open: Array(total).fill(0) };
}

function barTemplate(song, r, style) {
  const spb = song.stepsPerBeat, B = song.beatsPerBar, n = spb * B;
  const t = { kick: Array(n).fill(0), snare: Array(n).fill(0), hat: Array(n).fill(0), open: Array(n).fill(0) };
  const beat = b => Math.round(b * spb);
  const compound = spb % 3 === 0;

  if (compound) {
    // 6/8, 12/8, shuffles: kick on one, backbeat on the answering beat, hats on the triplet pulse
    const sub = spb / 3;
    t.kick[0] = 1;
    if (B === 2) t.snare[beat(1)] = 1;
    else if (B === 3) { t.snare[beat(1)] = 0.5; t.snare[beat(2)] = 1; }
    else for (let b = 1; b < B; b += 2) t.snare[beat(b)] = 1;
    if (B >= 4) t.kick[beat(2)] = style === 'sparse' ? 0 : 1;
    if (style !== 'sparse' && r() < 0.6) t.kick[beat(B - 1) + 2 * sub] = 0.5;
    if (B === 2 && style === 'busy') t.kick[beat(1) - sub] = 0.5;
    for (let b = 0; b < B; b++) {
      t.hat[beat(b)] = 1;
      if (style !== 'sparse') { t.hat[beat(b) + sub] = style === 'busy' ? 0.5 : 0; t.hat[beat(b) + 2 * sub] = 0.5; }
    }
    return t;
  }

  if (B === 3) {
    t.kick[0] = 1;
    t.snare[beat(2)] = 1;
    if (style !== 'sparse') t.snare[beat(1)] = 0.5;
    for (let b = 0; b < B; b++) { t.hat[beat(b)] = 1; if (spb >= 2 && style !== 'sparse') t.hat[beat(b + 0.5)] = 0.5; }
    return t;
  }

  if (B === 4 && spb === 4) {
    const kicks = {
      sparse: [[0, 10], [0, 8], [0]],
      steady: [[0, 10], [0, 7, 10], [0, 3, 10], [0, 8, 10], [0, 6, 10]],
      busy: [[0, 3, 7, 10], [0, 2, 10, 11], [0, 6, 8, 11]],
      four: [[0, 4, 8, 12]],
    }[style];
    for (const k of pick(r, kicks)) t.kick[k] = 1;
    if (style === 'sparse') t.snare[8] = 1;
    else { t.snare[4] = 1; t.snare[12] = 1; }
    if (style === 'busy' && r() < 0.6) t.snare[pick(r, [7, 9, 15])] = 0.5;
    if (style === 'four') {
      for (const h of [2, 6, 10, 14]) t.hat[h] = 1;
      if (r() < 0.5) { t.hat[14] = 0; t.open[14] = 1; }
    } else {
      for (let h = 0; h < 16; h += 2) t.hat[h] = h % 4 === 0 ? 1 : 0.5;
      if (style === 'busy') for (const h of [3, 11]) t.hat[h] = 0.5;
    }
    return t;
  }

  // any other meter: kick on one, snare on alternate beats, hats on the half-beat
  t.kick[0] = 1;
  for (let b = 1; b < B; b += 2) t.snare[beat(b)] = 1;
  if (B >= 4) t.kick[beat(Math.floor(B / 2))] = 1;
  for (let b = 0; b < B; b++) { t.hat[beat(b)] = 1; if (spb >= 2) t.hat[beat(b + 0.5)] = 0.5; }
  return t;
}

function fromGroove(song, level) {
  const g = song.drums && song.drums.groove;
  if (!g) return null;
  const n = song.stepsPerBeat * song.beatsPerBar;
  const strong = Math.max(...g.kick) > 0.45 && Math.max(...g.snare) > 0.35;
  if (!strong || g.kick.length !== n) return null;
  const t = { kick: [], snare: [], hat: [], open: Array(n).fill(0) };
  for (const row of ['kick', 'snare', 'hat']) {
    t[row] = g[row].map(p => (p >= level ? 1 : p >= level * 0.62 ? 0.5 : 0));
  }
  if (t.hat.reduce((a, b) => a + b, 0) < 2) t.hat = null;
  return t;
}

/** Pick a kit and feel from the song's own drum analysis. */
export function vibe(song) {
  const d = song.drums || {};
  const g = d.groove;
  const density = g ? (g.kick.concat(g.snare, g.hat).reduce((a, b) => a + b, 0)) / song.beatsPerBar : 0;
  const slow = song.bpm < 92;
  const style = !g || d.level < 0.12 ? (slow ? 'sparse' : 'steady')
    : density > 3.2 ? 'busy' : density < 1.3 ? 'sparse' : 'steady';
  const hasSong = (d.kit || []).includes('kick') && (d.kit || []).includes('snare');
  const kit = hasSong ? 'song' : d.level < 0.12 ? 'soft' : song.bpm > 118 ? '808' : 'dust';
  return { style, kit, swing: d.swing > 0.06 ? Math.min(0.4, d.swing) : 0 };
}

/** Drum pattern: the song's own groove on the first pass, then variations in the same feel. */
export function autoDrums(state, seed, variation) {
  const r = rng(seed);
  const { song, bars, notes } = state;
  const spb = song.stepsPerBeat, n = spb * song.beatsPerBar, total = n * bars;
  const feel = vibe(song);
  const styles = ['sparse', 'steady', 'busy'];
  let style = feel.style;
  if (variation > 0) {
    const shift = (variation % 3) - 1;
    style = styles[Math.max(0, Math.min(2, styles.indexOf(style) + shift))];
    if (song.bpm >= 112 && spb === 4 && song.beatsPerBar === 4 && variation % 4 === 3) style = 'four';
  }
  const own = variation % 2 === 0 ? fromGroove(song, variation === 0 ? 0.5 : 0.38) : null;
  const base = barTemplate(song, r, style);
  const A = own ? { kick: own.kick, snare: own.snare, hat: own.hat || base.hat, open: own.open } : base;

  const rows = emptyRows(total);
  for (let bar = 0; bar < bars; bar++) {
    const last = bars > 1 && bar === bars - 1, turn = bars >= 4 && bar % 2 === 1;
    for (const row of Object.keys(rows)) for (let s = 0; s < n; s++) rows[row][bar * n + s] = A[row][s];
    const o = bar * n;
    if (turn || last) {
      // answer bar: a pickup kick and an open hat or ghost snare leading back to one
      const tail = n - (spb % 3 === 0 ? spb / 3 : Math.max(1, spb / 2));
      if (!rows.kick[o + tail] && !rows.snare[o + tail] && r() < 0.7) rows.kick[o + tail] = 0.5;
      if (last && style !== 'sparse') {
        const g = n - 1;
        if (!rows.snare[o + g] && !rows.kick[o + g]) rows.snare[o + g] = 0.5;
        if (r() < 0.5 && rows.hat[o + tail]) { rows.hat[o + tail] = 0; rows.open[o + tail] = 1; }
      }
    }
  }
  // lock the kick to where the sample loop restarts its phrases
  const starts = new Set(notes.filter(x => x.len >= spb).map(x => x.step));
  for (const s of starts) {
    if (s % spb === 0 && !rows.snare[s] && !rows.kick[s] && r() < 0.3) rows.kick[s] = 0.5;
  }
  return { rows, kit: feel.kit, swing: feel.swing };
}
