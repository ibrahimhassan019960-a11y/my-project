# Original score synthesised from scratch (no samples) -> no licensing issues.
import numpy as np, wave
from scipy.signal import lfilter, butter, fftconvolve

SR = 48000
import os
VER = os.environ.get('VER', '15' if os.environ.get('V15') == '1' else '30')
V15 = VER == '15'
DUR = {'30': 30.0, '15': 15.0, '30v': 31.5, '15v': 15.0}[VER]
N = int(SR * DUR)
BEAT = 0.5                     # 120 BPM, 15 video frames per beat
DROP, END = 12.5, 25.5
rng = np.random.default_rng(7)
mix = np.zeros((N, 2))

def at(t): return int(t * SR)
def add(sig, t, gain=1.0, pan=0.0):
    i = at(t)
    if i >= N: return
    sig = sig[:N - i]
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    if sig.ndim == 1:
        mix[i:i + len(sig), 0] += sig * gain * l * 1.414
        mix[i:i + len(sig), 1] += sig * gain * r * 1.414
    else:
        mix[i:i + len(sig)] += sig * gain

def env(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(t / max(a, 1e-4), 1) * np.exp(-t / d)

def lp(x, fc, order=2):
    b, a = butter(order, min(fc / (SR / 2), 0.99)); return lfilter(b, a, x)
def hp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), 'high'); return lfilter(b, a, x)

def kick():
    n = at(0.45); t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 7) + 0.3 * lp(rng.standard_normal(n), 3000) * np.exp(-t * 120)
    return np.tanh(s * 1.6)
def clap():
    n = at(0.35); s = hp(lp(rng.standard_normal(n), 6000), 900)
    e = np.zeros(n)
    for o in (0, 0.012, 0.024): e += env(n, 0.001, 0.012) * 0 + np.roll(env(n, 0.0005, 0.01), at(o))
    e += np.roll(env(n, 0.001, 0.09), at(0.03))
    return s * e * 0.8
def hat(open_=False):
    n = at(0.25 if open_ else 0.06); s = hp(rng.standard_normal(n), 7500, 4)
    return s * env(n, 0.0005, 0.08 if open_ else 0.018) * 0.5

def saw(f, n, det=0.0):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in (-det, 0, det):
        out += 2 * ((t * f * (1 + d)) % 1) - 1
    return out / 3

def note(f): return f
CH = [  # Am, F, C, G (pad voicings)
    [220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 261.63, 329.63], [196.0, 246.94, 293.66]]
ROOT = [55.0, 43.65, 65.41, 49.0]

def pad(chord, dur, fc=1800):
    n = at(dur); s = np.zeros(n)
    for f in chord: s += saw(f, n, 0.004) + 0.5 * saw(f * 2, n, 0.006)
    s = lp(s, fc)
    t = np.arange(n) / SR
    e = np.minimum(t / 0.35, 1) * np.minimum((dur - t) / 0.3, 1).clip(0, 1)
    st = np.stack([s * e, np.roll(s, 300) * e], 1)
    return st * 0.09
def bass(f, dur):
    n = at(dur); t = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * t) + 0.35 * lp(saw(f, n), 400)
    e = np.minimum(t / 0.01, 1) * np.minimum((dur - t) / 0.05, 1).clip(0, 1)
    return np.tanh(s * e * 1.3) * 0.55
def impact():
    n = at(2.5); t = np.arange(n) / SR
    f = 32 + 90 * np.exp(-t * 9)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    nz = lp(rng.standard_normal(n), 2500) * np.exp(-t * 6) * 0.5
    return np.tanh((boom + nz) * 1.5)
def whoosh(dur=0.6, up=True):
    n = at(dur); x = rng.standard_normal(n); t = np.arange(n) / SR
    out = np.zeros(n); seg = at(0.02)
    for i in range(0, n, seg):
        k = i / n; fc = 300 + 5000 * (k if up else 1 - k) ** 2
        out[i:i + seg] = lp(x[i:i + seg], fc, 1)
    e = np.sin(np.pi * t / dur) ** 2
    return out * e * 0.7
def riser(dur):
    n = at(dur); t = np.arange(n) / SR
    x = whoosh(dur, True) * np.linspace(0.2, 1.4, n)
    tone = np.sin(2 * np.pi * np.cumsum(200 + 900 * (t / dur) ** 2) / SR) * 0.15 * (t / dur)
    return x + tone

def reverb(x, sec=1.4, wet=0.25):
    n = at(sec); ir = rng.standard_normal((n, 2)) * np.exp(-np.arange(n) / SR * 4)[:, None]
    ir[:, 0] = lp(ir[:, 0], 5000); ir[:, 1] = lp(ir[:, 1], 5000)
    y = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    y /= np.abs(y).max() + 1e-9; y *= np.abs(x).max()
    return x * (1 - wet) + y * wet

K, C, Hc, Ho = kick(), clap(), hat(), hat(True)
music = np.zeros((N, 2))
def section(t0, t1, full=True, drums=True):
    global mix
    b = t0
    while b < t1 - 1e-6:
        bar = int(round(b / 2.0)) % 4
        k = int(round((b % 2.0) / BEAT))
        if k == 0:
            add(pad(CH[bar], min(2.0, t1 - b) + 0.05, 2400 if full else 900), b, 1.0)
        if full:
            add(bass(ROOT[bar], 0.22), b, 0.9); add(bass(ROOT[bar], 0.22), b + 0.25, 0.6)
        if drums:
            if full or k == 0: add(K if full else lp(K, 900), b, 0.95 if full else 0.6)
            if full and k in (1, 3): add(C, b, 0.55, 0.05)
            add(Hc, b + 0.25, 0.35 if full else 0.15, 0.3)
            if full: add(Hc, b, 0.18, -0.3); add(Hc, b + 0.125, 0.1, 0.4); add(Hc, b + 0.375, 0.1, -0.4)
        b += BEAT

def plan(hook, rs, drop, end, cuts):
    """hook groove -> filtered BEFORE bed -> riser from rs -> drop -> tagline at end"""
    add(impact(), 0.0, 0.8)
    section(0.0, hook, True)
    add(whoosh(0.5, False), hook - 0.25, 0.6)
    add(impact(), hook, 0.45)
    section(hook, rs, False)
    add(riser(drop - rs), rs, 0.55)
    r, step = rs, 0.25
    while r < drop - 0.1:
        add(C, r, 0.18 + 0.3 * (r - rs) / (drop - rs)); r += step; step = max(0.0625, step * 0.8)
    add(impact(), drop, 1.0)
    section(drop, end, True)
    for t in cuts: add(whoosh(0.5), t - 0.35, 0.35)
    add(impact(), end, 0.9)
    add(pad(CH[0], DUR - end + 0.1, 1600), end, 1.4)
    add(bass(ROOT[0], 3.0) * np.exp(-np.arange(at(3.0)) / SR * 0.9), end, 0.6)
    add(Ho, end, 0.4)
    add(whoosh(0.8, True), end + 1.05, 0.3)

if VER == '30v':
    plan(4.0, 12.5, 14.0, 27.0, (17.0, 20.0, 24.0))
elif VER == '15v':
    plan(1.5, 6.0, 7.0, 11.0, (9.0,))
elif V15:
    add(impact(), 0.0, 0.8)
    section(0.0, 1.5, True)
    add(whoosh(0.5, False), 1.25, 0.6)
    add(impact(), 1.5, 0.45)
    section(1.5, 4.0, False)
    add(riser(1.0), 4.0, 0.55)
    r, step = 4.0, 0.2
    while r < 4.9:
        add(C, r, 0.18 + 0.3 * (r - 4.0) / 0.9); r += step; step = max(0.0625, step * 0.8)
    add(impact(), 5.0, 1.0)
    section(5.0, 11.5, True)
    for t in (7.5, 9.5): add(whoosh(0.5), t - 0.35, 0.35)
    add(impact(), 11.5, 0.9)
    add(pad(CH[0], 3.6, 1600), 11.5, 1.4)
    add(bass(ROOT[0], 3.0) * np.exp(-np.arange(at(3.0)) / SR * 0.9), 11.5, 0.6)
    add(Ho, 11.5, 0.4)
    add(whoosh(0.8, True), 12.55, 0.3)
else:
    # hook: hit + full groove
    add(impact(), 0.0, 0.8)
    section(0.0, 4.0, True)
    # BEFORE: filtered, sparse
    add(whoosh(0.5, False), 3.75, 0.6)
    add(impact(), 4.0, 0.45)
    section(4.0, 11.0, False)
    for t in (6.0, 8.0): add(whoosh(0.45), t - 0.3, 0.25)
    # transition card: riser + snare roll, then cut
    add(riser(1.5), 11.0, 0.55)
    r = 11.0
    step = 0.25
    while r < 12.4:
        add(C, r, 0.18 + 0.3 * (r - 11) / 1.4); r += step; step = max(0.0625, step * 0.8)
    # DROP / AFTER
    add(impact(), DROP, 1.0)
    section(DROP, END, True)
    for t in (15.5, 18.5, 22.5): add(whoosh(0.5), t - 0.35, 0.35)
    # ending
    add(impact(), END, 0.9)
    add(pad(CH[0], 4.6, 1600), END, 1.4)
    add(bass(ROOT[0], 3.5) * np.exp(-np.arange(at(3.5)) / SR * 0.9), END, 0.6)
    add(Ho, END, 0.4)
    add(whoosh(0.8, True), 25.55 + 1.0, 0.3)  # under the "فرق كبير" pop

mix = reverb(mix, 1.2, 0.18)
# fade out
fo = at(1.2); mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
mix /= np.abs(mix).max() + 1e-9
mix = np.tanh(mix * 1.4) / np.tanh(1.4) * 0.89
with wave.open({'30': 'music.wav', '15': 'music15.wav'}.get(VER, f'music_{VER}.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print('ok')
