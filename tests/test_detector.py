"""
Demo görüntüleri üzerinde algılayıcı testi.

  python tests/test_detector.py            -> tüm testleri çalıştırır, sonuç görsellerini demo/sonuclar'a yazar
  python -m pytest tests                   -> pytest ile

Beklenti: 'poz_' görsellerinde TEMAS/YOLMA, 'neg_' görsellerinde YOK/YAKIN.
"""
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trikokontrol.alarm import AlarmLogic  # noqa: E402
from trikokontrol.detector import (STATE_NEAR, STATE_NONE, STATE_PULL,  # noqa: E402
                                   STATE_TOUCH, MustacheDetector, classify, draw)

IN = os.path.join(ROOT, "demo", "girdiler")
OUT = os.path.join(ROOT, "demo", "sonuclar")
ALARM = {STATE_TOUCH, STATE_PULL}


def run_images():
    if not glob.glob(os.path.join(IN, "*.jpg")):
        from tests import make_demo_images  # noqa
        make_demo_images.main()
    os.makedirs(OUT, exist_ok=True)
    det = MustacheDetector(video_mode=False)
    clean = cv2.imread(os.path.join(IN, "neg_01_el_yok.jpg"))
    results = []
    for f in sorted(glob.glob(os.path.join(IN, "*.jpg"))):
        name = os.path.basename(f)
        img = cv2.imread(f)
        # Gerçek kullanımdaki gibi: önce kamera birkaç kare elsiz yüzü görür
        det.reset_baseline()
        for _ in range(8):
            det.process(clean)
        r = det.process(img)
        expected_alarm = name.startswith("poz_")
        ok = (r.state in ALARM) == expected_alarm
        out = draw(img, r)
        cv2.putText(out, "DOGRU" if ok else "HATALI", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 180, 0) if ok else (0, 0, 255), 3)
        cv2.imwrite(os.path.join(OUT, name), out)
        results.append((name, r.state, r.hands, r.face_found, ok))
    det.close()
    return results


def test_demo_images():
    results = run_images()
    neg = [r for r in results if r[0].startswith("neg_")]
    pos = [r for r in results if r[0].startswith("poz_")]
    false_alarms = [r for r in neg if not r[4]]
    assert not false_alarms, f"Yanlış alarm: {false_alarms}"
    hit_rate = sum(r[4] for r in pos) / len(pos)
    assert hit_rate >= 0.85, f"Yakalama oranı düşük: {hit_rate:.0%} {[r for r in pos if not r[4]]}"


def test_geometry_synthetic():
    """Model olmadan saf geometri testi: kare bölge, sentetik el."""
    zone = np.array([[100, 100], [200, 100], [200, 130], [100, 130]], np.float32)
    fw = 300.0

    def hand(tip, thumb=None):
        h = np.full((21, 2), 1000.0, np.float32)
        h[8] = tip
        h[7] = (tip[0], tip[1] + 30)
        h[4] = thumb if thumb is not None else (tip[0] + 200, tip[1] + 200)
        h[3] = (h[4][0] + 10, h[4][1] + 10)
        return h

    assert classify(zone, fw, [hand((150, 115))])[0] == STATE_TOUCH
    assert classify(zone, fw, [hand((150, 115), thumb=(160, 118))])[0] == STATE_PULL
    assert classify(zone, fw, [hand((150, 300))])[0] == STATE_NONE
    assert classify(zone, fw, [hand((150, 170))])[0] == STATE_NEAR  # 40px = 0.13 yüz genişliği
    assert classify(zone, fw, [])[0] == STATE_NONE


def test_alarm_timing():
    """Kısa temaslarda (ör. kaşıma) alarm çalmaz, süren temasta çalar, el çekilince durur."""
    a = AlarmLogic(touch_hold_s=0.5, pull_hold_s=0.15, release_s=1.0, escalate_s=4.0)
    fps = 15
    t = [0.0]

    def feed(state, seconds):
        out = []
        for _ in range(int(round(seconds * fps))):
            out.append(a.update(state, t[0]))
            t[0] += 1 / fps
        return out

    assert set(feed(STATE_NONE, 1.0)) == {"idle"}
    assert set(feed(STATE_TOUCH, 0.3)) == {"idle"}          # kısa dokunuş (kaşıma) -> alarm yok
    assert set(feed(STATE_NONE, 1.0)) == {"idle"}
    s = feed(STATE_TOUCH, 1.0)                              # süren temas -> alarm
    assert s[0] == "idle" and s[-1] == "alarm"
    assert a.events == 1
    # algılamada tek karelik kopma alarmı kesmez
    assert feed(STATE_NONE, 1 / fps) == ["alarm"]
    assert feed(STATE_TOUCH, 0.5)[-1] == "alarm"
    s = feed(STATE_NONE, 1.5)                               # el çekildi -> ~1 sn sonra kapanır
    assert s[0] == "alarm" and s[-1] == "idle"
    # yolma (çimdik) daha hızlı tetikler
    s = feed(STATE_PULL, 0.3)
    assert s[0] == "idle" and s[-1] == "alarm"
    assert a.events == 2
    # el bırakılmazsa üst seviye (ekranı kapat/kilitle)
    assert feed(STATE_PULL, 4.5)[-1] == "escalate"


def test_video_sequence():
    """Video modunda: el yok -> el bıyıkta (yüz kapanınca hafızadaki bölgeyle) -> alarm."""
    if not os.path.exists(os.path.join(IN, "neg_01_el_yok.jpg")):
        run_images()
    det = MustacheDetector(video_mode=True)
    alarm = AlarmLogic(touch_hold_s=0.4, pull_hold_s=0.15, release_s=1.0)
    clean = cv2.imread(os.path.join(IN, "neg_01_el_yok.jpg"))
    touch = cv2.imread(os.path.join(IN, "poz_03_isaret_parmagi_biyik_orta.jpg"))
    seq = [clean] * 10 + [touch] * 15 + [clean] * 25
    states = []
    t = 0.0
    for fr in seq:
        r = det.process(fr, now=t)
        states.append(alarm.update(r.state, t))
        t += 1 / 15
    det.close()
    assert states[:10] == ["idle"] * 10
    assert "alarm" in states[10:25]
    assert states[-1] == "idle"


if __name__ == "__main__":
    res = run_images()
    ok = sum(r[4] for r in res)
    print(f"{'GORSEL':45s} {'DURUM':7s} EL  YUZ  SONUC")
    for name, st, hands, face, good in res:
        print(f"{name:45s} {st:7s} {hands:<3d} {str(face):5s} {'DOGRU' if good else 'HATALI'}")
    print(f"\nToplam: {ok}/{len(res)} doğru")
    test_geometry_synthetic()
    test_alarm_timing()
    test_video_sequence()
    print("Geometri, alarm zamanlama ve video dizisi testleri: GEÇTİ")
