"""
Demo/test görüntüleri üretir.

Gerçek bir portre fotoğrafına (MediaPipe örnek görselleri) gerçek el fotoğrafları
kesilip yapıştırılır. Parmak ucu bıyık bölgesine, yanağa, alına, çeneye veya yüzden
uzağa konularak pozitif ve negatif senaryolar oluşturulur.

Kullanım:  python tests/make_demo_images.py
Kaynak görseller indirilemezse demo/kaynak klasöründekiler kullanılır.
"""
import os
import sys
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "demo", "kaynak")
OUT = os.path.join(ROOT, "demo", "girdiler")
BASE_URL = "https://storage.googleapis.com/mediapipe-assets/"
FILES = ["portrait.jpg", "pointing_up.jpg", "fist.jpg", "thumb_up.jpg", "victory.jpg"]


def fetch():
    os.makedirs(SRC, exist_ok=True)
    for f in FILES:
        p = os.path.join(SRC, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(BASE_URL + f, p)


def hand_cutout(img):
    """Düz duvar önündeki eli maskeyle ayırır (renk farkı + GrabCut)."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:10].reshape(-1, 3), lab[:, :10].reshape(-1, 3),
                             lab[:, -10:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    diff = np.linalg.norm((lab - bg)[:, :, 1:], axis=2)  # a,b kanalları: ten rengi
    mask = (diff > 8).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == k).astype(np.uint8)
    gc = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    gc[cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0] = cv2.GC_FGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
    m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(m, (5, 5), 0)


def paste(base, hand, mask, anchor_src, anchor_dst, scale, angle):
    """El görselini, anchor_src noktası anchor_dst'e gelecek şekilde döndürüp yapıştırır."""
    M = cv2.getRotationMatrix2D(tuple(map(float, anchor_src)), angle, scale)
    M[:, 2] += np.array(anchor_dst, dtype=np.float64) - np.array(anchor_src, dtype=np.float64)
    H, W = base.shape[:2]
    hw = cv2.warpAffine(hand, M, (W, H))
    mw = cv2.warpAffine(mask, M, (W, H)).astype(np.float32)[:, :, None] / 255.0
    return (base * (1 - mw) + hw * mw).astype(np.uint8)


def webcam_frame(det, mp, img):
    """Portreyi, laptop kamerası kadrajına benzeyecek şekilde yüze yakın 640x480 kırpar."""
    f = det.face.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).face_landmarks[0]
    H, W = img.shape[:2]
    cx, cy = f[1].x * W, f[1].y * H
    fw = abs(f[454].x - f[234].x) * W
    cw = fw * 2.6
    ch = cw * 0.75
    x0, y0 = int(max(0, cx - cw / 2)), int(max(0, cy - ch * 0.5))
    crop = img[y0:int(y0 + ch), x0:int(x0 + cw)]
    return cv2.resize(crop, (640, 480), interpolation=cv2.INTER_CUBIC)


def add_mustache(img, face, seed=7):
    """Yüze kıl tellerinden oluşan gerçekçi koyu bir bıyık çizer (kullanıcının bıyığını taklit)."""
    H, W = img.shape[:2]
    P = lambda i: np.array([face[i].x * W, face[i].y * H])  # noqa: E731
    rng = np.random.default_rng(seed)
    lip = [P(i) for i in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]]
    nose = P(2)
    fw = np.linalg.norm(P(234) - P(454))
    layer = img.copy()
    for _ in range(1400):
        k = rng.uniform(0, len(lip) - 1)
        a, b = lip[int(k)], lip[min(int(k) + 1, len(lip) - 1)]
        base = a + (b - a) * (k - int(k))
        top = nose + (base - nose) * 0.35
        p = top + (base - top) * rng.uniform(0.05, 1.0)
        side = (p[0] - nose[0]) / (fw * 0.3)
        ang = np.deg2rad(90 + 35 * np.clip(side, -1, 1) + rng.normal(0, 10))
        ln = fw * rng.uniform(0.03, 0.06)
        q = p + ln * np.array([np.cos(ang), np.sin(ang)])
        c = int(rng.uniform(15, 45))
        cv2.line(layer, tuple(p.astype(int)), tuple(q.astype(int)), (c, c, c + 5), 1, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.85, img, 0.15, 0)


def main():
    fetch()
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, ROOT)
    from trikokontrol.detector import MustacheDetector

    det = MustacheDetector(video_mode=False)
    import mediapipe as mp

    portrait = webcam_frame(det, mp, cv2.imread(os.path.join(SRC, "portrait.jpg")))
    detect_face = lambda im: det.face.detect(mp.Image(  # noqa: E731
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))).face_landmarks[0]
    portrait = add_mustache(portrait, detect_face(portrait))
    face = detect_face(portrait)
    H, W = portrait.shape[:2]
    P = lambda i: (face[i].x * W, face[i].y * H)  # noqa: E731
    fw = np.linalg.norm(np.subtract(P(234), P(454)))
    mus_l = np.add(np.multiply(P(2), 0.4), np.multiply(P(61), 0.6))   # bıyığın sol yanı
    mus_r = np.add(np.multiply(P(2), 0.4), np.multiply(P(291), 0.6))  # bıyığın sağ yanı
    mus_c = np.add(np.multiply(P(2), 0.5), np.multiply(P(0), 0.5))    # bıyık ortası

    hands = {}
    for name in ["pointing_up", "fist", "thumb_up", "victory"]:
        img = cv2.imread(os.path.join(SRC, name + ".jpg"))
        m = hand_cutout(img)
        res = det.hand.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        lm = res.hand_landmarks[0]
        h, w = img.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in lm])
        hand_w = np.linalg.norm(pts[5] - pts[17]) * 1.25  # avuç genişliği
        hands[name] = (img, m, pts, hand_w)

    def put(name, tip_idx, dst, angle, size=0.55, base=None, pts_override=None):
        img, m, pts, hw = hands[name]
        src = pts[tip_idx] if pts_override is None else pts_override
        scale = (fw * size) / hw
        return paste(portrait if base is None else base, img, m, src, dst, scale, angle)

    thumb_index_mid = lambda name: (hands[name][2][4] + hands[name][2][8]) / 2  # noqa: E731

    scenarios = {
        # --- POZİTİF: bıyığa dokunma / yolma ---
        "poz_01_isaret_parmagi_biyik_sol.jpg": put("pointing_up", 8, mus_l, -60),
        "poz_02_isaret_parmagi_biyik_sag.jpg": put("pointing_up", 8, mus_r, 60),
        "poz_03_isaret_parmagi_biyik_orta.jpg": put("pointing_up", 8, mus_c, 0),
        "poz_04_yumruk_cimdik_biyik.jpg": put("fist", 4, mus_l, -20, pts_override=thumb_index_mid("fist")),
        "poz_05_basparmak_biyik.jpg": put("thumb_up", 4, mus_r, 35),
        "poz_06_iki_parmak_biyik.jpg": put("victory", 8, mus_l, -15),
        "poz_07_isaret_parmagi_egik_orta.jpg": put("pointing_up", 8, mus_c, -30),
        "poz_08_isaret_parmagi_egik_orta2.jpg": put("pointing_up", 8, mus_c, 30),
        "poz_09_yumruk_cimdik_sag.jpg": put("fist", 4, mus_r, 25, pts_override=thumb_index_mid("fist")),
        "poz_10_basparmak_biyik_sol.jpg": put("thumb_up", 4, mus_l, -35),
        # --- NEGATİF: yüzün başka yerleri veya yüzden uzak ---
        "neg_01_el_yok.jpg": portrait.copy(),
        "neg_02_parmak_alinda.jpg": put("pointing_up", 8, P(151), 75),
        "neg_03_parmak_yanakta.jpg": put("pointing_up", 8, np.add(P(234), [fw * 0.12, fw * 0.05]), -70),
        "neg_04_parmak_cenede.jpg": put("pointing_up", 8, P(152), 0),
        "neg_05_el_yuzden_uzak.jpg": put("pointing_up", 8, (W * 0.12, H * 0.75), 10),
        "neg_06_parmak_gozde.jpg": put("pointing_up", 8, P(159), 110),
        "neg_07_yumruk_omuzda.jpg": put("fist", 8, (W * 0.85, H * 0.8), 0),
    }
    for fn, im in scenarios.items():
        cv2.imwrite(os.path.join(OUT, fn), im)
        print("yazildi:", fn)

    # Hareket dizisi: yumruk aşağıdan (kadraj dışından) yükselip bıyığa gelir.
    seq_dir = os.path.join(OUT, "dizi")
    os.makedirs(seq_dir, exist_ok=True)
    start = np.array([mus_l[0] - fw * 0.1, H * 1.05])
    for i in range(10):
        k = min(1.0, i / 6)
        dst = start + (np.array(mus_l) - start) * k
        im = put("fist", 4, dst, -20, pts_override=thumb_index_mid("fist"))
        cv2.imwrite(os.path.join(seq_dir, f"yumruk_yaklasma_{i:02d}.jpg"), im)
    print("yazildi: dizi/yumruk_yaklasma_*.jpg")
    det.close()


if __name__ == "__main__":
    main()
