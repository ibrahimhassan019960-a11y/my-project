# Voice-led mix: cleaned-up technician voice on top, music ducked under it, light ambience elsewhere.
import numpy as np, wave, subprocess, sys
SR = 48000
VER = sys.argv[1]
def rd(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32768
def clean(p):
    o = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', p, '-af',
        'highpass=f=90,lowpass=f=9000,afftdn=nf=-25,acompressor=threshold=-22dB:ratio=3:attack=5:release=90:makeup=4,equalizer=f=2800:t=q:w=1.2:g=3',
        '-ac', '2', '-ar', str(SR), '-f', 's16le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(o, np.int16).reshape(-1, 2) / 32768
if VER == '30v':
    dur, voice = 35.5, [('v1', 4.0, 99), ('v2', 7.3, 153)]
    amb = [('s1', 0, 60, .10), ('s2', 2.0, 60, .10), ('s6b', 14.0, 90, .10), ('s7', 17.0, 90, .10), ('s9', 24.0, 90, .12), ('s11', 27.0, 135, .10)]
else:
    dur, voice = 19.0, [('v2b', 1.5, 165)]
    amb = [('s1', 0, 45, .10), ('s6b', 7.0, 60, .10), ('s7', 9.0, 60, .10), ('s11', 11.0, 120, .10)]
N = int(dur * SR)
mus = rd(f'music_{VER}.wav')[:N]
duck = np.ones(N)
vo = np.zeros((N, 2)); am = np.zeros((N, 2))
def place(dst, a, t, nf, g):
    i = int(t * SR); L = min(len(a), int(nf / 30 * SR), N - i)
    e = np.ones(L); r = int(0.03 * SR); e[:r] = np.linspace(0, 1, r); e[-r:] = np.linspace(1, 0, r)
    dst[i:i + L] += a[:L] * e[:, None] * g
    return i, i + L
for n, t, nf in voice:
    v = clean(f'seg/{n}.wav')
    v = v / (np.sqrt(np.mean(v ** 2)) + 1e-9) * 0.12      # consistent voice level
    i0, i1 = place(vo, v, t, nf, 1.0)
    duck[i0:i1] = 0.22                                     # ~ -13 dB music under speech
for n, t, nf, g in amb: place(am, rd(f'seg/{n}.wav'), t, nf, g)
k = int(0.25 * SR); ker = np.ones(k) / k
duck = np.convolve(np.pad(duck, (k, k), mode='edge'), ker, 'same')[k:-k]  # smooth ramps
out = mus * duck[:, None] * 0.8 + vo + am
out /= max(1.0, np.abs(out).max() / 0.95)
w = wave.open(f'mix_{VER}.wav', 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((out * 32767).astype(np.int16).tobytes())
print('mix', VER, len(out) / SR)
