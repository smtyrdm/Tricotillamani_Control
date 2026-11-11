"""Demo görüntülerinden kamera akışını taklit eden bir test videosu üretir (demo/demo_video.mp4).
Kullanım: python tests/make_demo_video.py   ->   python baslat.py --video demo/demo_video.mp4
"""
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "demo", "girdiler")
FPS = 15
PLAN = [  # (görüntü, saniye)
    ("neg_01_el_yok.jpg", 3.0),
    ("neg_05_el_yuzden_uzak.jpg", 1.0),
    ("neg_03_parmak_yanakta.jpg", 1.5),
    ("neg_01_el_yok.jpg", 1.0),
    ("poz_01_isaret_parmagi_biyik_sol.jpg", 2.5),   # bıyığa dokunma -> alarm
    ("neg_01_el_yok.jpg", 3.0),                     # el çekildi -> alarm kapanır
    ("poz_04_yumruk_cimdik_biyik.jpg", 2.0),        # yolma -> alarm
    ("neg_01_el_yok.jpg", 3.0),
]


def main():
    out = os.path.join(ROOT, "demo", "demo_video.mp4")
    w = None
    for name, sec in PLAN:
        img = cv2.imread(os.path.join(IN, name))
        if w is None:
            w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (img.shape[1], img.shape[0]))
        for _ in range(int(sec * FPS)):
            w.write(img)
    w.release()
    print("yazildi:", out)


if __name__ == "__main__":
    main()
