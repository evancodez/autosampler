"""Song analysis: stems, beat grid, tempo, key, sections, slices, pitch, drum groove."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

warnings.filterwarnings("ignore")

SR = 44100
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
STEMS = ["drums", "bass", "other", "vocals"]
SOURCES = ["mix", "nodrums", "vocals", "other", "bass", "drums"]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAX_CACHED_SONGS = 10
VERSION = 8

_demucs = None
_beat = None


# ---------------------------------------------------------------- io

def song_id(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]


def clean_title(name: str) -> str:
    t = Path(name).stem
    t = re.sub(r"\s*\[[\w-]{6,}\]\s*$", "", t)
    t = re.sub(r"\s*\((Official|Lyric|Audio|HD)[^)]*\)", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def decode(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def write_flac(path: Path, wave: np.ndarray):
    sf.write(str(path), np.clip(wave.T, -1, 1), SR, format="FLAC", subtype="PCM_16")


def read_audio(path: Path) -> np.ndarray:
    y, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return y.T


def prune_cache(keep: str):
    if not CACHE.exists():
        return
    dirs = sorted((d for d in CACHE.iterdir() if d.is_dir() and d.name != keep),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[MAX_CACHED_SONGS - 1:]:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- stems

def separate(wave: np.ndarray, progress) -> dict[str, np.ndarray]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    global _demucs
    if _demucs is None:
        _demucs = get_model("htdemucs")
        _demucs.eval()
    ref = wave.mean(0)
    mu, sd = float(ref.mean()), float(ref.std()) + 1e-8
    x = torch.from_numpy((wave - mu) / sd)[None]
    total = x.shape[-1]

    def on(d):
        if d.get("state") == "end":
            progress(min(1.0, (d.get("segment_offset", 0) + SR * 7.8) / total))

    devices = (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]
    out = None
    for dev in devices:
        try:
            with torch.no_grad():
                out = apply_model(_demucs, x, device=dev, split=True, overlap=0.25, shifts=1, callback=on)
            break
        except Exception:
            if dev == "cpu":
                raise
    out = (out[0] * sd + mu).cpu().numpy()
    return {name: out[i] for i, name in enumerate(_demucs.sources)}


# ---------------------------------------------------------------- beats / tempo

def activations(mono: np.ndarray):
    """Frame-wise beat / downbeat probabilities at 50 fps."""
    import torch
    from beat_this.inference import Audio2Frames

    global _beat
    if _beat is None:
        _beat = Audio2Frames(checkpoint_path="final0", device="cpu")
    with torch.no_grad():
        bl, dl = _beat(mono.astype(np.float32), SR)
    return torch.sigmoid(bl).numpy().astype(float), torch.sigmoid(dl).numpy().astype(float)


FPS = 50


def beat_period(act: np.ndarray) -> float:
    """Beat period in seconds: interval-histogram mode, refined on a comb over the autocorrelation."""
    x = act - act.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0] + 1e-12
    lo, hi = int(0.2 * FPS), min(int(2.0 * FPS), n - 2)
    seg = acf[lo:hi + 1]
    pk = np.nonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
    if len(pk) == 0:
        raise ValueError("no beats")
    strong = pk[seg[pk] >= 0.75 * seg[pk].max()]
    tau0 = float(lo + strong[0]) / FPS
    k_max = int(max(1, min(16, (n / FPS / 3) // tau0)))
    ks = np.arange(1, k_max + 1)
    lags = np.arange(len(acf))
    best, tau = -1e9, tau0
    for cand in np.arange(tau0 * 0.93, tau0 * 1.07, 0.0002):
        v = np.interp(ks * cand * FPS, lags, acf).sum()
        if v > best:
            best, tau = v, cand
    return float(tau)


def track_beats(act: np.ndarray) -> np.ndarray:
    """Tempo-locked dynamic programming over the activation, with sub-frame peak refinement."""
    import librosa

    tau = beat_period(act)
    hop = 441
    _, frames = librosa.beat.beat_track(onset_envelope=act, sr=FPS * hop, hop_length=hop,
                                        bpm=60.0 / tau, tightness=120, trim=False, units="frames")
    out = []
    w = int(np.clip(tau / 3 * FPS, 2, 5))
    for f in frames:
        f = int(f)
        lo, hi = max(0, f - w), min(len(act), f + w + 1)
        wt = act[lo:hi] ** 4 + 1e-9
        out.append(float((np.arange(lo, hi) * wt).sum() / wt.sum()) / FPS)
    return np.unique(np.round(out, 4))


def fallback_beats(mono: np.ndarray):
    import librosa

    _, beats = librosa.beat.beat_track(y=mono, sr=SR, hop_length=256, units="time")
    return np.asarray(beats, float)


def regularize_beats(beats: np.ndarray, dur: float) -> np.ndarray:
    """Fill dropped beats, smooth tracker jitter, extend to cover the whole file."""
    b = np.unique(np.round(beats, 4))
    med = float(np.median(np.diff(b)))
    out = [b[0]]
    for t in b[1:]:
        base = out[-1]
        gap = t - base
        if gap < 0.5 * med:
            continue
        n = int(round(gap / med))
        if n >= 2:
            out.extend(base + gap * j / n for j in range(1, n))
        out.append(t)
    b = np.array(out)
    # symmetric moving average keeps linear trends, removes 20 ms frame quantisation
    sm = b.copy()
    for i in range(len(b)):
        w = min(4, i, len(b) - 1 - i)
        sm[i] = b[i - w:i + w + 1].mean()
    b = sm
    head = float(np.median(np.diff(b[:9])))
    tail = float(np.median(np.diff(b[-9:])))
    pre = []
    t = b[0] - head
    while t > -head:
        pre.append(t)
        t -= head
    post = []
    t = b[-1] + tail
    while t < dur + tail:
        post.append(t)
        t += tail
    return np.concatenate([pre[::-1], b, post])


def estimate_tempo(beats: np.ndarray):
    """Long-span slope statistics: precise for click-tracked songs, honest for drifting ones."""
    n = len(beats)
    span = int(np.clip(n // 4, 4, 32))
    slopes = (beats[span:] - beats[:-span]) / span
    period = float(np.median(slopes))
    drift = float((np.percentile(slopes, 90) - np.percentile(slopes, 10)) / period)
    stable = drift < 0.012
    if stable:
        k = np.arange(n)
        local = np.gradient(beats)
        ok = np.abs(local / period - 1) < 0.04
        if ok.sum() > 16:
            p, a = np.polyfit(k[ok], beats[ok], 1)
            res = beats[ok] - (a + p * k[ok])
            keep = np.abs(res) < 2.5 * (np.std(res) + 1e-6)
            if keep.sum() > 16:
                p, a = np.polyfit(k[ok][keep], beats[ok][keep], 1)
            period = float(p)
    bpm = 60.0 / period
    if stable and abs(bpm - round(bpm)) < 0.08:
        bpm = float(round(bpm))
    elif stable and abs(bpm * 2 - round(bpm * 2)) < 0.08:
        bpm = round(bpm * 2) / 2
    else:
        bpm = round(bpm, 1)
    return bpm, stable, drift


def align_grid(beats: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Shift the grid so beats sit on the real transient starts (median offset of strong hits)."""
    starts, strength, _ = find_onsets(y)
    t = starts[strength > 0.45] / SR + 0.003
    if len(t) < 12:
        return beats
    k = np.clip(np.searchsorted(beats, t), 1, len(beats) - 1)
    near = np.where(np.abs(beats[k] - t) < np.abs(beats[k - 1] - t), beats[k], beats[k - 1])
    off = t - near
    off = off[np.abs(off) < 0.045]
    if len(off) < 12:
        return beats
    return beats + float(np.median(off))


def meter_from_activation(beats: np.ndarray, dact: np.ndarray):
    """Beats per bar and downbeat phase from downbeat probability sampled on the beat grid."""
    d = []
    for t in beats:
        c = int(round(t * FPS))
        d.append(dact[max(0, c - 3):c + 4].max() if 0 <= c < len(dact) else 0.0)
    d = np.array(d)
    if d.max() < 0.3 or len(d) < 24:
        return 4, 0
    score = {}
    for per in (2, 3, 4, 6):
        vals = [d[ph::per].mean() for ph in range(per)]
        score[per] = (max(vals), int(np.argmax(vals)))
    top = max(v[0] for v in score.values())
    per = next(p for p in (2, 3, 4, 6) if score[p][0] >= 0.8 * top)
    phase = score[per][1]
    if per == 2:
        vals = [d[ph::4].mean() for ph in (phase, phase + 2)]
        per, phase = 4, phase + 2 * int(np.argmax(vals))
    return per, phase


def subdivision(beats: np.ndarray, oenv: np.ndarray, hop: int, bpm: float) -> int:
    def at(frac):
        vals = []
        for a, b in zip(beats[:-1], beats[1:]):
            c = int(round((a + (b - a) * frac) * SR / hop))
            if 2 <= c < len(oenv) - 2:
                vals.append(oenv[c - 2:c + 3].max())
        return float(np.mean(vals)) if vals else 0.0

    base = float(np.median(oenv)) + 1e-9
    half = at(0.5) / base
    trip = max(at(1 / 3), at(2 / 3)) / base
    trip_both = (at(1 / 3) + at(2 / 3)) / 2 / base
    quarter = (at(0.25) + at(0.75)) / 2 / base
    if trip > 1.3 * half and trip_both > 1.1 * quarter:
        return 3
    return 2 if bpm > 165 else 4


def beat_pos(beats: np.ndarray, t):
    k = np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2)
    return k + (t - beats[k]) / (beats[k + 1] - beats[k])


def time_at(beats: np.ndarray, pos):
    k = np.clip(np.floor(pos).astype(int), 0, len(beats) - 2)
    return beats[k] + (pos - k) * (beats[k + 1] - beats[k])


# ---------------------------------------------------------------- harmony

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def harmony(nodrums_mono: np.ndarray):
    import librosa

    y = librosa.resample(nodrums_mono, orig_sr=SR, target_sr=22050)
    tuning = float(librosa.estimate_tuning(y=y, sr=22050))
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=2048, tuning=tuning)
    mfcc = librosa.feature.mfcc(y=y, sr=22050, hop_length=2048, n_mfcc=13)
    w = np.sqrt((chroma ** 2).sum(0))
    prof = (chroma * w).sum(1)
    best, key = -2, "C"
    for i in range(12):
        for name, tpl in (("", MAJOR), ("m", MINOR)):
            r = np.corrcoef(np.roll(tpl, i), prof)[0, 1]
            if r > best:
                best, key = r, NOTE_NAMES[i] + name
    return tuning * 100, key, chroma, mfcc, 2048 / 22050


def sections(chroma, mfcc, hop_s, beats, per, phase, dur):
    from sklearn.cluster import AgglomerativeClustering

    bar_idx = np.arange(phase, len(beats), per)
    bars = beats[bar_idx]
    bars = bars[(bars >= 0) & (bars < dur)]
    if len(bars) < 12:
        return [{"start": 0.0, "end": dur, "label": 0}], np.zeros(1, int), np.array([0.0])
    feats = []
    for a, b in zip(bars, np.append(bars[1:], dur)):
        i, j = int(a / hop_s), max(int(a / hop_s) + 1, int(b / hop_s))
        c = chroma[:, i:j].mean(1)
        m = mfcc[:, i:j].mean(1)
        feats.append(np.concatenate([c / (np.linalg.norm(c) + 1e-9) * 2.0, m]))
    X = np.array(feats)
    X[:, 12:] = (X[:, 12:] - X[:, 12:].mean(0)) / (X[:, 12:].std(0) + 1e-9) * 0.35
    k = int(np.clip(len(bars) // 10, 2, 6))
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    sm = labels.copy()
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i]:
            sm[i] = labels[i - 1]
    remap, out = {}, []
    for i, lab in enumerate(sm):
        lab = remap.setdefault(int(lab), len(remap))
        sm[i] = lab
        end = float(bars[i + 1]) if i + 1 < len(bars) else dur
        if out and out[-1]["label"] == lab:
            out[-1]["end"] = end
        else:
            out.append({"start": float(bars[i]) if i else 0.0, "end": end, "label": lab})
    return out, sm, bars


# ---------------------------------------------------------------- onsets

def onset_env(y: np.ndarray, hop=256):
    import librosa

    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop, n_fft=1024, n_mels=64, aggregate=np.median)
    return o / (np.percentile(o, 99.5) + 1e-9)


def amp_env(y: np.ndarray):
    e = maximum_filter1d(np.abs(y), int(SR * 0.001))
    return uniform_filter1d(e, int(SR * 0.003))


def refine_start(env: np.ndarray, peak: int) -> int:
    """Walk back from the attack's amplitude peak to where the rise begins."""
    lo = max(0, peak - int(0.02 * SR))
    hi = min(len(env), peak + int(0.045 * SR))
    if hi - lo < 16:
        return peak
    mx = lo + int(np.argmax(env[lo:hi]))
    back = max(0, mx - int(0.09 * SR))
    seg = env[back:mx + 1]
    floor = float(seg.min())
    thr = floor + 0.12 * (float(env[mx]) - floor)
    below = np.nonzero(seg < thr)[0]
    t0 = back + (int(below[-1]) if len(below) else 0)
    return max(0, t0 - int(0.003 * SR))


def find_onsets(y: np.ndarray, hop=256):
    import librosa

    oenv = onset_env(y, hop)
    peaks = librosa.util.peak_pick(oenv, pre_max=5, post_max=5, pre_avg=30, post_avg=30, delta=0.06, wait=10)
    env = amp_env(y)
    starts, strengths = [], []
    for p in peaks:
        s = refine_start(env, int(p * hop))
        if starts and s - starts[-1] < int(0.05 * SR):
            if oenv[p] > strengths[-1]:
                strengths[-1] = float(oenv[p])
            continue
        starts.append(s)
        strengths.append(float(oenv[p]))
    return np.array(starts, int), np.clip(np.array(strengths), 0, 1), oenv


# ---------------------------------------------------------------- pitch

def detect_pitch(y: np.ndarray, start: int, fmin: float, fmax: float):
    import librosa

    seg = y[start:start + int(0.65 * SR)]
    if len(seg) < int(0.2 * SR) or np.sqrt(np.mean(seg ** 2)) < 1e-4:
        return None, 0.0
    seg = librosa.resample(seg, orig_sr=SR, target_sr=22050)
    f0, flag, prob = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=22050, frame_length=2048, hop_length=256, resolution=0.2)
    t = np.arange(len(f0)) * 256 / 22050
    ok = flag & (t > 0.03) & np.isfinite(f0)
    if ok.sum() < 6:
        return None, 0.0
    midi = 69 + 12 * np.log2(f0[ok] / 440.0)
    pr = prob[ok]
    head = midi[:18]
    centre = float(np.median(head))
    near = np.abs(midi - centre) < 0.6
    if near.sum() < 4:
        return None, 0.0
    value = float(np.average(midi[near], weights=pr[near] + 1e-3))
    conf = float(ok.mean() * pr[near].mean() * min(1.0, near.sum() / 14))
    return value, conf


def note_label(midi: float, tuning_cents: float) -> str:
    n = int(round(midi - tuning_cents / 100))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


PITCH_RANGE = {"vocals": (82, 1050), "bass": (32, 330), "other": (65, 1600)}


# ---------------------------------------------------------------- slices

def analyze_slices(song: dict, source: str, stems: dict[str, np.ndarray], chroma, hop_s) -> dict:
    beats = np.array(song["beats"])
    spb, per, phase = song["stepsPerBeat"], song["beatsPerBar"], song["phase"]
    tuning = song["tuning"]
    y = stems[source]
    starts, strength, oenv = find_onsets(y)
    hop = 256
    rms = np.sqrt(uniform_filter1d(y ** 2, int(0.02 * SR))[::hop] + 1e-12)
    ref = float(np.percentile(rms, 95)) + 1e-9
    period = 60.0 / song["bpm"]
    sec_bars = np.array(song["_barTimes"])
    sec_labels = np.array(song["_barLabels"])

    cands = []
    for s, st in zip(starts, strength):
        t = s / SR
        if t > song["duration"] - 1.5 * period:
            continue
        f = s // hop
        pre = rms[max(0, f - int(0.09 * SR / hop)):max(1, f - 1)].mean()
        post = rms[f + 1:f + int(0.12 * SR / hop)].mean()
        body = rms[f:f + int(period * SR / hop)]
        loud_db = 20 * np.log10(body.mean() / ref + 1e-9)
        if loud_db < -22:
            continue
        contrast = float(np.clip(20 * np.log10(post / (pre + 1e-6)) / 18, 0, 1))
        loud = float(np.clip((loud_db + 20) / 20, 0, 1))
        sustain = float((body > body.max() * 0.25).mean())
        bp = float(beat_pos(beats, t + 0.003))
        step = bp * spb
        err = abs(step - round(step)) / spb * period
        grid = float(np.exp(-(err / 0.03) ** 2))
        k = int(round(step))
        on_beat = k % spb == 0
        on_down = on_beat and ((k // spb) - phase) % per == 0
        accent = grid * (1.0 if on_down else 0.6 if on_beat else 0.2)
        c0, c1 = int(t / hop_s), int((t + period) / hop_s) + 1
        ch = chroma[:, c0:c1].mean(1) if c1 > c0 and c0 < chroma.shape[1] else np.zeros(12)
        bi = int(np.clip(np.searchsorted(sec_bars, t, side="right") - 1, 0, len(sec_labels) - 1))
        pre_score = 0.3 * st + 0.27 * contrast + 0.18 * loud + 0.15 * accent + 0.1 * sustain
        cands.append({"s": int(s), "t": t, "beat": bp, "strength": float(st), "contrast": contrast,
                      "score": pre_score, "chroma": ch, "sec": int(sec_labels[bi]), "midi": None, "conf": 0.0})

    pitched = source != "drums"
    if pitched:
        mono = {k: stems[k] for k in ("vocals", "other", "bass")}
        order = sorted(range(len(cands)), key=lambda i: -cands[i]["score"])[:72]
        for i in order:
            c = cands[i]
            s = c["s"]
            if source in PITCH_RANGE:
                tries = [source]
            else:
                w = slice(s, s + int(0.4 * SR))
                lv = {k: float(np.sqrt(np.mean(mono[k][w] ** 2))) for k in mono}
                tries = ["vocals", "other"] if lv["vocals"] > 0.45 * lv["other"] else ["other", "vocals"]
                if source == "mix" or source == "nodrums":
                    tries.append("bass")
            for name in tries:
                m, conf = detect_pitch(mono[name], s, *PITCH_RANGE[name])
                if m is not None and conf > 0.25:
                    c["midi"], c["conf"] = m, conf
                    break
            if c["midi"] is None:
                ch = c["chroma"]
                if ch.max() > 0 and ch.max() > 1.5 * np.median(ch):
                    c["pc"] = int(np.argmax(ch))
            c["score"] += 0.15 * c["conf"]

    out = []
    for c in cands:
        ch = c["chroma"]
        ch = ch / (ch.max() + 1e-9)
        item = {"t": round(c["t"], 5), "beat": round(c["beat"], 4), "score": round(c["score"], 4),
                "strength": round(c["strength"], 3), "sec": c["sec"],
                "chroma": [int(round(v * 9)) for v in ch]}
        if c["midi"] is not None:
            item["midi"] = round(c["midi"], 3)
            item["note"] = note_label(c["midi"], tuning)
            item["conf"] = round(c["conf"], 3)
        elif "pc" in c:
            item["note"] = NOTE_NAMES[c["pc"]]
        out.append(item)
    return {"source": source, "slices": out, "pads": pick_pads(out, 16)}


def pick_pads(slices: list[dict], n: int) -> list[int]:
    if not slices:
        return []
    chroma = np.array([s["chroma"] for s in slices], float)
    chroma /= np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9
    beat = np.array([s["beat"] for s in slices])
    score = np.array([s["score"] for s in slices])
    notes = [s.get("note") for s in slices]
    secs = [s["sec"] for s in slices]
    chosen: list[int] = []
    while len(chosen) < min(n, len(slices)):
        best, best_v = -1, -1e9
        for i in range(len(slices)):
            if i in chosen:
                continue
            v = score[i]
            if chosen:
                d = np.abs(beat[chosen] - beat[i])
                if d.min() < 1.0:
                    continue
                v -= 0.22 * float(np.exp(-d.min() / 6))
                v -= 0.12 * float((chroma[chosen] @ chroma[i]).max() ** 4)
                v -= 0.05 * sum(1 for j in chosen if notes[j] and notes[j] == notes[i])
                v -= 0.03 * sum(1 for j in chosen if secs[j] == secs[i])
            if v > best_v:
                best, best_v = i, v
        if best < 0:
            break
        chosen.append(best)
    return sorted(chosen, key=lambda i: slices[i]["t"])


# ---------------------------------------------------------------- drums

def analyze_drums(drums: np.ndarray, mix_rms: float, song: dict, out_dir: Path) -> dict:
    import librosa

    beats = np.array(song["beats"])
    spb, per, phase = song["stepsPerBeat"], song["beatsPerBar"], song["phase"]
    steps = spb * per
    level = float(np.sqrt(np.mean(drums ** 2)) / (mix_rms + 1e-9))
    empty = {"level": round(level, 3), "groove": None, "kit": [], "swing": 0.0}
    if level < 0.05:
        return empty
    starts, strength, _ = find_onsets(drums)
    if len(starts) < 24:
        return empty
    S = np.abs(librosa.stft(drums, n_fft=2048, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    bands = {"low": S[freqs < 140].sum(0), "mid": S[(freqs > 180) & (freqs < 3500)].sum(0), "high": S[freqs > 6000].sum(0)}
    hits = []
    for i, (s, st) in enumerate(zip(starts, strength)):
        f = s // 256 + 1
        e = {k: float(v[f:f + 8].sum()) for k, v in bands.items()}
        tot = sum(e.values()) + 1e-12
        lo, mid, hi = e["low"] / tot, e["mid"] / tot, e["high"] / tot
        if lo > 0.45:
            cls, purity = "kick", lo
        elif hi > 0.5 and mid < 0.4:
            cls, purity = "hat", hi
        else:
            cls, purity = "snare", mid + 0.5 * hi
        nxt = (starts[i + 1] - s) / SR if i + 1 < len(starts) else 1.0
        hits.append({"s": int(s), "cls": cls, "purity": purity, "strength": float(st), "gap": nxt, "energy": tot})

    counts = {k: np.zeros(steps) for k in ("kick", "snare", "hat")}
    bars_active = set()
    late = []
    for h in hits:
        pos = float(beat_pos(beats, h["s"] / SR + 0.003)) * spb
        k = int(round(pos))
        if abs(pos - k) > 0.38:
            continue
        bar = (k - phase * spb) // steps
        st = (k - phase * spb) % steps
        counts[h["cls"]][st] += 1
        bars_active.add(bar)
        if spb == 4 and st % 2 == 1:
            late.append(pos - k)
    nb = max(1, len(bars_active))
    groove = {k: [round(float(min(1.0, v / nb)), 3) for v in c] for k, c in counts.items()}
    swing = float(np.clip(np.median(late) * 2, 0, 0.5)) if len(late) > 12 else 0.0

    kit = []
    limits = {"kick": (0.16, 0.45), "snare": (0.16, 0.4), "hat": (0.06, 0.2)}
    for cls, (min_gap, max_len) in limits.items():
        group = [h for h in hits if h["cls"] == cls]
        if len(group) < 4:
            continue
        med_e = np.median([h["energy"] for h in group])
        pool = [h for h in group if h["gap"] >= min_gap and h["energy"] >= med_e] or \
               [h for h in group if h["gap"] >= min_gap]
        if not pool:
            continue
        best = max(pool, key=lambda h: h["purity"] * min(1.0, h["gap"] / max_len) * (0.5 + h["strength"]))
        n = int(min(max_len, max(min_gap, best["gap"] - 0.006)) * SR)
        seg = song["_drumsStereo"][:, best["s"]:best["s"] + n].copy()
        if seg.shape[1] < int(0.04 * SR):
            continue
        fade = max(16, int(seg.shape[1] * 0.35))
        seg[:, -fade:] *= np.linspace(1, 0, fade) ** 2
        seg[:, :64] *= np.linspace(0, 1, 64)
        seg *= 0.89 / (np.abs(seg).max() + 1e-9)
        sf.write(str(out_dir / f"kit_{cls}.wav"), seg.T, SR, subtype="PCM_16")
        kit.append(cls)
    return {"level": round(level, 3), "groove": groove, "kit": kit, "swing": round(swing, 3)}


# ---------------------------------------------------------------- pipeline

def analyze(path: Path, source: str, progress=lambda stage, frac: None) -> dict:
    sid = song_id(path)
    out = CACHE / sid
    out.mkdir(parents=True, exist_ok=True)
    out.touch()
    meta_path = out / "song.json"
    song = None
    if meta_path.exists():
        song = json.loads(meta_path.read_text())
        if song.get("version") != VERSION or not all((out / f"{s}.flac").exists() for s in SOURCES):
            song = None

    stems_mono: dict[str, np.ndarray] = {}
    if song is None:
        prune_cache(sid)
        for old in out.glob("slices_*.json"):
            old.unlink()
        if all((out / f"{s}.flac").exists() for s in SOURCES):
            progress("Reading", 0.3)
            stems = {s: read_audio(out / f"{s}.flac") for s in SOURCES}
            mix = stems["mix"]
            dur = mix.shape[1] / SR
        else:
            progress("Reading", 0.02)
            mix = decode(path)
            dur = mix.shape[1] / SR
            progress("Separating", 0.05)
            stems = separate(mix, lambda f: progress("Separating", 0.05 + 0.6 * f))
            stems["nodrums"] = stems["bass"] + stems["other"] + stems["vocals"]
            stems["mix"] = mix
            for name, w in stems.items():
                write_flac(out / f"{name}.flac", w)
        stems_mono = {k: v.mean(0) for k, v in stems.items()}

        progress("Tempo", 0.68)
        has_drums = np.sqrt(np.mean(stems_mono["drums"] ** 2)) > 0.05 * np.sqrt(np.mean(stems_mono["mix"] ** 2))
        try:
            act, dact = activations(stems_mono["mix"])
            raw_beats = track_beats(act)
            live = np.nonzero(act > 0.5)[0]
            t_lo, t_hi = live[0] / FPS, live[-1] / FPS
        except Exception:
            raw_beats, dact = fallback_beats(stems_mono["mix"]), None
            t_lo, t_hi = raw_beats[0], raw_beats[-1]
        beats = regularize_beats(raw_beats, dur)
        beats = align_grid(beats, stems_mono["drums"] if has_drums else stems_mono["mix"])
        per, phase = meter_from_activation(beats, dact) if dact is not None else (4, 0)
        compound = False
        inner = beats[(beats >= t_lo - 0.05) & (beats <= t_hi + 0.05)]
        bpm, stable, drift = estimate_tempo(inner)
        if per == 6 and bpm >= 150:
            # eighth-note pulse of a 6/8 bar: use the dotted-quarter as the beat
            beats = beats[phase % 3::3]
            first_down = (phase - phase % 3) // 3
            per, phase, compound = 2, first_down % 2, True
            inner = beats[(beats >= t_lo - 0.05) & (beats <= t_hi + 0.05)]
            bpm, stable, drift = estimate_tempo(inner)
            spb = 6 if bpm <= 96 else 3
        else:
            spb = subdivision(inner, onset_env(stems_mono["mix"]), 256, bpm)
            compound = spb == 3
        if per == 2:
            meter = "6/8"
        elif compound:
            meter = f"{per * 3}/8"
        else:
            meter = f"{per}/4"

        progress("Harmony", 0.76)
        tuning, key, chroma, mfcc, hop_s = harmony(stems_mono["nodrums"])
        secs, bar_labels, bar_times = sections(chroma, mfcc, hop_s, beats, per, phase, dur)
        np.save(out / "chroma.npy", chroma.astype(np.float32))

        song = {"version": VERSION, "id": sid, "title": clean_title(path.name), "file": path.name,
                "duration": round(dur, 3), "bpm": bpm, "stable": bool(stable), "drift": round(drift, 4),
                "beats": [round(float(b), 4) for b in beats], "beatsPerBar": per, "phase": phase,
                "stepsPerBeat": spb, "meter": meter, "key": key, "tuning": round(tuning, 1), "sections": secs,
                "_barTimes": [round(float(b), 3) for b in bar_times], "_barLabels": [int(x) for x in bar_labels],
                "_hop": hop_s}
        progress("Drums", 0.82)
        song["_drumsStereo"] = stems["drums"]
        drums = analyze_drums(stems_mono["drums"], float(np.sqrt(np.mean(stems_mono["mix"] ** 2))), song, out)
        del song["_drumsStereo"]
        song["drums"] = drums
        meta_path.write_text(json.dumps(song))

    slice_path = out / f"slices_{source}.json"
    if slice_path.exists():
        sl = json.loads(slice_path.read_text())
    else:
        progress("Slicing", 0.88)
        need = {source, "vocals", "other", "bass"}
        for name in need:
            if name not in stems_mono:
                stems_mono[name] = read_audio(out / f"{name}.flac").mean(0)
        chroma = np.load(out / "chroma.npy")
        sl = analyze_slices(song, source, stems_mono, chroma, song["_hop"])
        slice_path.write_text(json.dumps(sl))
    progress("Done", 1.0)
    public = {k: v for k, v in song.items() if not k.startswith("_")}
    public.update(sl)
    return public


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    last = [""]

    def show(stage, frac):
        if stage != last[0]:
            last[0] = stage
            print(f"[{time.time() - t0:6.1f}s] {stage}", flush=True)

    r = analyze(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "nodrums", show)
    print(f"{r['title']}: {r['meter']} {r['bpm']} bpm stable={r['stable']} drift={r['drift']} "
          f"{r['beatsPerBar']} beats/bar x{r['stepsPerBeat']} key={r['key']} tuning={r['tuning']}c")
    print("sections", [(round(s['start']), s['label']) for s in r["sections"]])
    print("drums", {k: v for k, v in r["drums"].items() if k != "groove"})
    if r["drums"]["groove"]:
        for k, v in r["drums"]["groove"].items():
            print(f"  {k:5s}", "".join("X" if p > 0.6 else "x" if p > 0.3 else "." for p in v))
    print(len(r["slices"]), "slices; pads:")
    for i in r["pads"]:
        s = r["slices"][i]
        print(f"  {s['t']:7.2f}s beat {s['beat']:7.2f} score {s['score']:.2f} {s.get('note', '-'):4s} conf {s.get('conf', 0):.2f} sec {s['sec']}")
