import cv2, numpy as np, json, subprocess, sys, math, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
PL = json.load(open('plates.json'))
RED = json.load(open('red.json'))
CAIRO, MONT = 'fonts/Cairo.ttf', 'fonts/Montserrat.ttf'
ACCENT = (225, 22, 34)          # RGB, sampled from the real LED colour
LED_CORE = (255, 75, 62)
PREVIEW = len(sys.argv) > 1 and sys.argv[1] == 'preview'

def ease(t):  # easeOutCubic
    t = min(max(t, 0), 1); return 1 - (1 - t) ** 3
def eio(t):
    t = min(max(t, 0), 1); return t * t * (3 - 2 * t)

def load(n):
    c = cv2.VideoCapture(f'seg/{n}.mp4'); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr

# ---------- privacy: blur the customer's licence plate ----------
def blur_plate(f, n, i):
    if n not in PL: return f
    i = min(i, len(PL[n]) - 1)
    x, y, w, h = PL[n][i]
    p = 14
    x0, y0 = int(max(0, x - p)), int(max(0, y - p))
    x1, y1 = int(min(f.shape[1], x + w + p)), int(min(f.shape[0], y + h + p))
    if y + h > f.shape[0] - 25: y1 = f.shape[0]
    if x1 - x0 < 4 or y1 - y0 < 4: return f
    roi = f[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, (x1 - x0) // 14), max(1, (y1 - y0) // 14)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    pix = cv2.GaussianBlur(pix, (0, 0), 6)
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.rectangle(m, (6, 6), (x1 - x0 - 7, y1 - y0 - 7), 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
    f = f.copy()
    f[y0:y1, x0:x1] = (pix * m + roi * (1 - m)).astype(np.uint8)
    return f

def crop_scale(f, cx, cy, cw, ch, ow, oh):
    sh, sw = f.shape[:2]
    cw = min(cw, sw); ch = min(ch, sh)
    x0 = min(max(cx - cw / 2, 0), sw - cw); y0 = min(max(cy - ch / 2, 0), sh - ch)
    M = np.float32([[ow / cw, 0, -x0 * ow / cw], [0, oh / ch, -y0 * oh / ch]])
    out = cv2.warpAffine(f, M, (ow, oh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    bl = cv2.GaussianBlur(out, (0, 0), 1.6)
    out = cv2.addWeighted(out, 1.45, bl, -0.45, 0)
    return out, (x0, y0, ow / cw)

def background(f):
    s = cv2.resize(f, (54, 96), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 3)
    b = cv2.resize(s, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) * 0.32
    return b

YY, XX = np.mgrid[0:H, 0:W]
VIG = (1 - 0.55 * (((XX - W / 2) / (W * 0.75)) ** 2 + ((YY - H / 2) / (H * 0.62)) ** 2)).clip(0.35, 1)[..., None].astype(np.float32)

# ---------- text sprites ----------
_cache = {}
def font(path, size, wght):
    F = ImageFont.truetype(path, size)
    axes = F.get_variation_axes()
    F.set_variation_by_axes([wght] + [0] * (len(axes) - 1))
    return F

def sprite(txt, path=CAIRO, size=60, wght=800, fill=(255, 255, 255), track=0, glow=None, shadow=True, pad=40):
    key = (txt, path, size, wght, fill, track, glow, shadow)
    if key in _cache: return _cache[key]
    F = font(path, size, wght)
    ar = path == CAIRO
    if track and not ar:
        widths = [F.getlength(ch) for ch in txt]
        tw = int(sum(widths) + track * (len(txt) - 1))
    else:
        tw = int(F.getlength(txt, direction='rtl' if ar else None) if ar else F.getlength(txt))
    asc, desc = F.getmetrics()
    th = asc + desc
    img = Image.new('RGBA', (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    def draw(dr, col):
        if track and not ar:
            x = pad
            for ch, wd in zip(txt, widths):
                dr.text((x, pad), ch, font=F, fill=col); x += wd + track
        else:
            dr.text((pad, pad), txt, font=F, fill=col, direction='rtl' if ar else None, features=['-liga'] if not ar else None)
    layers = []
    if shadow:
        sh = Image.new('RGBA', img.size, (0, 0, 0, 0)); draw(ImageDraw.Draw(sh), (0, 0, 0, 200))
        sh = sh.filter(ImageFilter.GaussianBlur(10)); layers.append(sh)
    if glow:
        g = Image.new('RGBA', img.size, (0, 0, 0, 0)); draw(ImageDraw.Draw(g), glow + (255,))
        g = g.filter(ImageFilter.GaussianBlur(14)); layers.append(g); layers.append(g)
    for l in layers: img = Image.alpha_composite(img, l)
    draw(d, fill + (255,)) if False else None
    top = Image.new('RGBA', img.size, (0, 0, 0, 0)); draw(ImageDraw.Draw(top), fill + (255,))
    img = Image.alpha_composite(img, top)
    a = np.asarray(img).astype(np.float32) / 255
    rgb = a[..., :3][..., ::-1] * a[..., 3:4]  # premultiplied BGR
    _cache[key] = (rgb, a[..., 3:4])
    return _cache[key]

def pill(txt, bg, fg, size=58, track=10, glow=None):
    key = ('pill', txt, bg, fg, size, glow)
    if key in _cache: return _cache[key]
    F = font(MONT, size, 900)
    widths = [F.getlength(c) for c in txt]
    tw = int(sum(widths) + track * (len(txt) - 1))
    padx, pady, m = 46, 18, 40
    asc, desc = F.getmetrics()
    bw, bh = tw + 2 * padx, size + 2 * pady
    img = Image.new('RGBA', (bw + 2 * m, bh + 2 * m), (0, 0, 0, 0))
    if glow:
        g = Image.new('RGBA', img.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle((m, m, m + bw, m + bh), radius=bh // 2, fill=glow + (230,))
        img = Image.alpha_composite(img, g.filter(ImageFilter.GaussianBlur(18)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((m, m, m + bw, m + bh), radius=bh // 2, fill=bg)
    x = m + padx
    ty = m + pady - (asc - size) - int(size * 0.12)
    for c, wd in zip(txt, widths):
        d.text((x, ty), c, font=F, fill=fg); x += wd + track
    a = np.asarray(img).astype(np.float32) / 255
    _cache[key] = (a[..., :3][..., ::-1] * a[..., 3:4], a[..., 3:4])
    return _cache[key]

def put(frame, spr, cx, cy, op=1.0, scale=1.0, clipx=None):
    rgb, a = spr
    if op <= 0.003: return
    if scale != 1.0:
        hh, ww = a.shape[:2]
        nw, nh = max(1, int(ww * scale)), max(1, int(hh * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)[..., None]
    h, w = a.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + w), min(H, y0 + h)
    if clipx is not None:  # reveal mask (wipe) in absolute x
        fx0 = max(fx0, clipx[0]); fx1 = min(fx1, clipx[1])
    if fx1 <= fx0 or fy1 <= fy0: return
    sr = rgb[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] * 255 * op
    sa = a[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] * op
    frame[fy0:fy1, fx0:fx1] = frame[fy0:fy1, fx0:fx1] * (1 - sa) + sr

def text_in(frame, spr, cx, cy, t, t_in=0.0, t_out=None, dy=40):
    """slide-up + fade in at t_in, fade out at t_out (seconds within shot)"""
    k = ease((t - t_in) / 0.45)
    op = k
    if t_out is not None: op *= 1 - eio((t - t_out) / 0.3)
    put(frame, spr, cx, cy + (1 - k) * dy, op)

def accent_line(frame, cx, cy, width, t, t_in=0.15, thick=6):
    k = ease((t - t_in) / 0.5)
    w = int(width * k)
    if w < 2: return
    ov = np.zeros((40, width + 80, 3), np.float32)
    cv2.line(ov, (40 + (width - w) // 2, 20), (40 + (width + w) // 2, 20), ACCENT[::-1], thick, cv2.LINE_AA)
    g = cv2.GaussianBlur(ov, (0, 0), 7)
    ov = np.clip(ov + g * 1.6, 0, 255)
    y0, x0 = int(cy - 20), int(cx - width / 2 - 40)
    sub = frame[y0:y0 + 40, x0:x0 + width + 80]
    frame[y0:y0 + 40, x0:x0 + width + 80] = np.clip(sub + ov, 0, 255)

def add_glow_layer(frame, ov, strength=1.6, sigma=9):
    g = cv2.GaussianBlur(ov, (0, 0), sigma)
    np.clip(frame + ov + g * strength, 0, 255, out=frame)

# ---------- shots ----------
WIN_W = (740, 700)     # window height, source crop width (wide car shots)
SHOTS = [
    dict(seg='s1', n=60, kind='T', crop=[(424, 225, 478), (424, 205, 410)], flash=True),
    dict(seg='s2', n=60, kind='W', follow=True, crop=[(None, 239, 700), (None, 239, 660)], rgbout=True),
    dict(seg='s3', n=90, kind='T', crop=[(330, 239, 478), (330, 230, 450)], flash=True, rgbin=True),
    dict(seg='s4', n=60, kind='T', crop=[(300, 239, 478), (300, 239, 478)]),
    dict(seg='s5', n=60, kind='W', crop=[(500, 239, 700), (500, 239, 660)]),
    dict(seg=None, n=45, kind='CARD'),
    dict(seg='s6', n=90, kind='SPLIT', flash=True),
    dict(seg='s7', n=90, kind='W', crop=[(402, 239, 700), (402, 239, 700)]),
    dict(seg='s8', n=120, kind='W', crop=[(424, 200, 700), (424, 220, 640)]),
    dict(seg='s9', n=90, kind='W', crop=[(470, 239, 700), (470, 239, 680)]),
    dict(seg='s11', n=135, kind='W', crop=[(402, 239, 700), (402, 226, 560)], flash=True),
]

# 15-second cut: hook -> BEFORE -> split (drop on beat 10) -> AFTER -> detail -> tagline
SHOTS15 = [
    dict(seg='s1', n=45, kind='T', crop=[(424, 225, 478), (424, 210, 420)], flash=True, rgbout=True),
    dict(seg='s3', n=60, kind='T', crop=[(330, 239, 478), (330, 232, 455)], flash=True, rgbin=True),
    dict(seg='s4', n=45, kind='T', crop=[(300, 239, 478), (300, 239, 478)]),
    dict(seg='s6', n=75, kind='SPLIT', flash=True),
    dict(seg='s7', n=60, kind='W', crop=[(402, 239, 700), (402, 239, 700)]),
    dict(seg='s8', n=60, kind='W', crop=[(424, 200, 700), (424, 215, 660)]),
    dict(seg='s11', n=105, kind='W', crop=[(402, 239, 700), (402, 228, 580)], flash=True),
]
V15 = os.environ.get('V15') == '1'
if V15: SHOTS = SHOTS15

S = {}
for n in ['s1', 's2', 's3', 's4', 's5', 's6a', 's6b', 's7', 's8', 's9', 's11']:
    S[n] = load(n)

# text sprites
T_KIA = sprite('KIA SPORTAGE', MONT, 66, 800, track=14)
T_SUB1 = sprite('إضاءة LED للمسطرة الخلفية', CAIRO, 62, 800)
T_DIFF = sprite('شوف الفرق.. قبل وبعد', CAIRO, 70, 900)
T_REAR = sprite('REAR LED STRIP', MONT, 40, 700, fill=(235, 235, 235), track=12)
P_BEFORE = pill('BEFORE', (245, 245, 245, 255), (20, 20, 20))
P_AFTER = pill('AFTER', ACCENT + (255,), (255, 255, 255), glow=ACCENT)
P_BEFORE_S = pill('BEFORE', (245, 245, 245, 235), (20, 20, 20), size=40, track=7)
P_AFTER_S = pill('AFTER', ACCENT + (240,), (255, 255, 255), size=40, track=7, glow=ACCENT)
T_C3 = sprite('المسطرة الأصلية.. بدون إضاءة', CAIRO, 60, 800)
T_C4 = sprite('شكلها الأصلي من غير أي إضاءة', CAIRO, 60, 800)
T_C5 = sprite('المسطرة في الورشة قبل التعديل', CAIRO, 60, 800)
T_CARD1 = sprite('إضافة إضاءة LED بطول المسطرة', CAIRO, 62, 900)
T_CARD2 = sprite('رسم توضيحي', CAIRO, 34, 600, fill=(170, 170, 170), shadow=False)
T_C7 = sprite('إضاءة LED بطول المسطرة', CAIRO, 62, 800)
T_LED = sprite('LED', MONT, 44, 900, fill=(255, 255, 255), track=6, glow=ACCENT)
T_C8a = sprite('تشطيب نظيف..', CAIRO, 64, 900)
T_C8b = sprite('وإضاءة متناسقة مع الفوانيس', CAIRO, 60, 800)
T_C9 = sprite('شغل حقيقي.. على نفس العربية', CAIRO, 62, 800)
T_END1 = sprite('تفاصيل صغيرة..', CAIRO, 96, 1000, glow=(150, 0, 10))
T_END2 = sprite('فرق كبير', CAIRO, 120, 1000, fill=(255, 255, 255), glow=ACCENT)
T_END3 = sprite('KIA SPORTAGE  •  REAR LED STRIP', MONT, 36, 700, fill=(220, 220, 220), track=6)

def window(frame, img, top, punch=1.0):
    h, w = img.shape[:2]
    if punch != 1.0:
        nw, nh = int(w * punch), int(h * punch)
        big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = big[(nh - h) // 2:(nh - h) // 2 + h, (nw - w) // 2:(nw - w) // 2 + w]
    # drop shadow band
    frame[max(0, top - 30):top, :] *= np.linspace(1, 0.55, min(30, top))[:, None, None]
    frame[top + h:top + h + 30, :] *= np.linspace(0.55, 1, 30)[:, None, None]
    frame[top:top + h, :] = img.astype(np.float32)

def lerp(a, b, k): return a + (b - a) * k

def render_card(i, n, base):
    t = i / FPS
    frame = base.copy() * 0.55
    # curved strip illustration
    k = eio(i / (n - 8))
    xs = np.linspace(120, 960, 200)
    ys = 900 + 70 * ((xs - 540) / 420) ** 2 - 40
    pts = np.stack([xs, ys], 1).astype(np.int32)
    body = np.zeros_like(frame)
    cv2.polylines(body, [pts], False, (70, 70, 78), 34, cv2.LINE_AA)
    cv2.polylines(body, [pts], False, (40, 40, 48), 22, cv2.LINE_AA)
    a_in = ease(i / 8)
    frame[:] = frame * (1 - (body.sum(2, keepdims=True) > 0) * a_in) + body * a_in
    m = int(len(pts) * k)
    if m > 1:
        ov = np.zeros_like(frame)
        cv2.polylines(ov, [pts[:m]], False, ACCENT[::-1], 14, cv2.LINE_AA)
        cv2.polylines(ov, [pts[:m]], False, LED_CORE[::-1], 6, cv2.LINE_AA)
        hx, hy = pts[m - 1]
        cv2.circle(ov, (int(hx), int(hy)), 10, (230, 230, 255), -1, cv2.LINE_AA)
        add_glow_layer(frame, ov, 2.2, 16)
    text_in(frame, T_CARD1, 540, 700, t, 0.05)
    text_in(frame, T_CARD2, 540, 1060, t, 0.25)
    return frame

def render(si, sh, i, gidx):
    t = i / FPS
    n = sh['n']
    k = i / max(1, n - 1)
    kind = sh['kind']
    if kind == 'CARD':
        return render_card(i, n, CARD_BASE)
    if kind == 'SPLIT':
        fa = S['s6a'][min(i, len(S['s6a']) - 1)]
        fb = blur_plate(S['s6b'][min(i, len(S['s6b']) - 1)], 's6b', i)
        frame = background(fb)
        ph = 620
        ia, _ = crop_scale(fa, 400, 239, 832, 478, W, ph)
        ib, _ = crop_scale(fb, 406, 239, 832, 478, W, ph)
        reveal = ease(i / 12)
        topA, topB = 330, 330 + ph + 14
        frame[topA:topA + ph] = ia
        # AFTER panel wipes in from the left
        wx = int(W * reveal)
        frame[topB:topB + ph, :wx] = ib[:, :wx]
        ov = np.zeros((H, W, 3), np.float32)
        lw = int(W * ease((i - 4) / 14))
        if lw > 0:
            cv2.line(ov, ((W - lw) // 2, topA + ph + 7), ((W + lw) // 2, topA + ph + 7), ACCENT[::-1], 6, cv2.LINE_AA)
        add_glow_layer(frame, ov, 2.0, 10)
        put(frame, P_BEFORE_S, 185, topA + 60, ease((i - 2) / 8))
        put(frame, P_AFTER_S, 165, topB + 60, ease((i - 10) / 8))
        return frame

    segname = sh['seg']
    fr = S[segname]
    src = blur_plate(fr[min(i, len(fr) - 1)], segname, i)
    frame = background(src)
    (cx0, cy0, cw0), (cx1, cy1, cw1) = sh['crop']
    kk = eio(k)
    cw = lerp(cw0, cw1, kk); cy = lerp(cy0, cy1, kk)
    if sh.get('follow'):
        r = RED[segname][min(i, len(RED[segname]) - 1)]
        cx = (r[0] + r[1]) / 2 if r else 424
    else:
        cx = lerp(cx0, cx1, kk)
    punch = 1 + 0.07 * (1 - ease(i / 9)) if i < 9 else 1.0
    if kind == 'T':
        ow, oh, top = W, 1080, 340
        img, geo = crop_scale(src, cx, cy, cw, cw, ow, oh)
    else:
        ow, oh, top = W, WIN_W[0], 470
        img, geo = crop_scale(src, cx, cy, cw, cw * oh / ow, ow, oh)
    # rgb split (rewind/glitch) transitions
    amt = 0
    if sh.get('rgbout') and i > n - 7: amt = (i - (n - 7)) * 5
    if sh.get('rgbin') and i < 5: amt = (5 - i) * 6
    if amt:
        img = img.copy()
        img[..., 2] = np.roll(img[..., 2], amt, 1); img[..., 0] = np.roll(img[..., 0], -amt, 1)
    window(frame, img, top, punch)
    bottom = top + oh
    seg = segname

    if seg == 's1':
        put(frame, T_KIA, 540, 250 + (1 - ease(t / 0.5)) * -30, ease(t / 0.4))
        accent_line(frame, 540, 1448, 520, t, 0.2)
        text_in(frame, T_SUB1, 540, 1520, t, 0.25)
    elif seg == 's2':
        put(frame, T_KIA, 540, 250, 1.0)
        put(frame, T_REAR, 540, 330, ease(t / 0.4))
        text_in(frame, T_DIFF, 540, bottom + 95, t, 0.1)
    elif seg in ('s3', 's4', 's5'):
        if seg == 's3':
            sc = 1 + 0.5 * (1 - ease(t / 0.35))
            put(frame, P_BEFORE, 540, 250, ease(t / 0.2), sc)
        else:
            put(frame, P_BEFORE, 540, 250, 1.0)
        cap = {'s3': T_C3, 's4': T_C4, 's5': T_C5}[seg]
        text_in(frame, cap, 540, (bottom + 85) if kind == 'W' else 1500, t, 0.08)
    elif seg == 's7':
        sc = 1 + 0.5 * (1 - ease(t / 0.35))
        put(frame, P_AFTER, 540, 270, ease(t / 0.2), sc)
        text_in(frame, T_C7, 540, bottom + 95, t, 0.3)
        # animated pointers at the real strip position
        r = RED['s7'][min(i, len(RED['s7']) - 1)]
        x0, y0, s = geo
        sy = top + (r[3] - y0) * s
        ov = np.zeros((H, W, 3), np.float32)
        ap = ease((t - 0.45) / 0.4)
        if ap > 0:
            bob = 10 * math.sin(t * 6)
            for sx in (330, 540, 750):
                ax = sx
                ty = sy - 40 + bob - (1 - ap) * 40
                pts = np.array([[ax - 22, ty - 26], [ax, ty], [ax + 22, ty - 26]], np.int32)
                cv2.polylines(ov, [pts], False, (255, 255, 255), 6, cv2.LINE_AA)
            add_glow_layer(frame, ov * ap, 0.8, 6)
    elif seg == 's8':
        text_in(frame, T_C8a, 540, 330, t, 0.15)
        text_in(frame, T_C8b, 540, bottom + 95, t, 0.5)
    elif seg == 's9':
        text_in(frame, T_C9, 540, bottom + 95, t, 0.15)
    elif seg == 's11':
        dark = eio((t - 0.3) / 1.0) * 0.35
        frame *= (1 - dark)
        text_in(frame, T_END1, 540, 1370, t, 0.5, dy=50)
        # second line pops with scale
        k2 = ease((t - 1.1) / 0.45)
        put(frame, T_END2, 540, 1500, k2, 1 + 0.35 * (1 - k2))
        accent_line(frame, 540, 1590, 420, t, 1.4)
        text_in(frame, T_END3, 540, 330, t, 1.6)
    return frame

# card background: a blurred frame of the finished car
CARD_BASE = background(blur_plate(S['s11'][60], 's11', 60)) * 1.6

out = ('reel_video15.mp4' if V15 else 'reel_video.mp4') if not PREVIEW else 'preview.mp4'
enc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-preset', 'slow' if not PREVIEW else 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
g = 0
total = sum(s['n'] for s in SHOTS)
snaps = [int(x) for x in sys.argv[2:]] if PREVIEW else []
for si, sh in enumerate(SHOTS):
    for i in range(sh['n']):
        if PREVIEW and snaps and g not in snaps:
            g += 1; continue
        f = render(si, sh, i, g)
        f *= VIG
        if sh.get('flash') and i < 6:
            f += (1 - i / 6) ** 2 * 140
        # global fade in / out
        if g < 6: f *= g / 6
        if g > total - 16: f *= max(0, (total - 1 - g) / 15)
        f8 = np.clip(f, 0, 255).astype(np.uint8)
        if PREVIEW and snaps: cv2.imwrite(f'frames/p_{g:04d}.jpg', f8)
        enc.stdin.write(f8.tobytes())
        g += 1
    print('shot', si, 'done', g, flush=True)
enc.stdin.close(); enc.wait()
print('total frames', total)
