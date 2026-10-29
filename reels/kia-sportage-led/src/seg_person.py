# Person + held-strip matte for the workshop shots (MediaPipe selfie-segmentation model via LiteRT).
import cv2, numpy as np, sys
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path='models/selfie_segmentation_landscape.tflite'); it.allocate_tensors()
inp, out = it.get_input_details()[0]['index'], it.get_output_details()[0]['index']
def person(f):
    x = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (256, 144), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    it.set_tensor(inp, x[None]); it.invoke()
    return cv2.resize(it.get_tensor(out)[0, ..., 0], (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
def strip(f, pm):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    r = (((hsv[..., 0] < 10) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 50)).astype(np.uint8)
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    near = cv2.dilate((pm > 0.5).astype(np.uint8), np.ones((45, 45), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(r)
    keep = np.zeros_like(r)
    for i in range(1, n):
        comp = lab == i
        if st[i, 4] > 300 and (near[comp].any()): keep[comp] = 1
    return cv2.dilate(keep, np.ones((7, 7), np.uint8)).astype(np.float32)
for n in sys.argv[1:]:
    c = cv2.VideoCapture(f'seg/{n}.mp4'); ms = []; ema = None
    while True:
        ok, f = c.read()
        if not ok: break
        p = person(f)
        ema = p if ema is None else 0.6 * p + 0.4 * ema      # temporal smoothing against flicker
        m = np.maximum(np.clip((ema - 0.3) / 0.4, 0, 1), strip(f, ema))
        m = cv2.GaussianBlur(m, (0, 0), 2.5)
        ms.append((m * 255).astype(np.uint8))
    np.save(f'seg/{n}_mask.npy', np.stack(ms)); print(n, len(ms), flush=True)
