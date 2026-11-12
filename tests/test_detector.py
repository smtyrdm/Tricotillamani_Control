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


# ---------------------------------------------------------------- kullanıcı geri bildirimi
def _portrait_geom():
    import mediapipe as mp
    from trikokontrol.detector import _to_px, face_geometry
    det = MustacheDetector(video_mode=False)
    img = cv2.imread(os.path.join(IN, "neg_01_el_yok.jpg"))
    fr = det.face.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    det.close()
    face_px = _to_px(fr.face_landmarks[0], img.shape[1], img.shape[0])
    return face_px, face_geometry(face_px, 1.25, 1.45)


def _fake_hand(wrist, tip, palm_len):
    """Bilekten parmak ucuna uzanan basit bir el iskeleti üretir."""
    wrist, tip = np.array(wrist, np.float32), np.array(tip, np.float32)
    d = (tip - wrist) / (np.linalg.norm(tip - wrist) + 1e-6)
    n = np.array([-d[1], d[0]], np.float32)
    h = np.zeros((21, 2), np.float32)
    h[0] = wrist
    for f, off in zip(range(5), (-0.45, -0.2, 0.0, 0.2, 0.4)):
        base = wrist + d * palm_len + n * off * palm_len
        for j in range(1, 5):
            idx = 1 + f * 4 + (j - 1)
            h[idx] = base + (tip - base) * (j / 4) if f == 1 else base + d * palm_len * 0.25 * j
    h[9] = wrist + d * palm_len
    return h


def test_phantom_hands_rejected():
    """Ekran görüntüsündeki hata: elin yokken yanak / kulak 'el' sanılıp alarm veriyordu."""
    from trikokontrol.detector import FACE_LEFT, hand_is_plausible
    face_px, g = _portrait_geom()
    fw = g.fw
    nose = face_px[1]
    # 1) yanakta küçük hayali el: bilek yanakta (dudak üstü), parmaklar burun/bıyık yanında
    cheek = face_px[205]
    phantom = _fake_hand(cheek, (nose[0], nose[1] + fw * 0.12), fw * 0.18)
    assert not hand_is_plausible(phantom, g)
    # 2) kulakta küçük hayali el
    ear = face_px[FACE_LEFT] + np.array([-fw * 0.05, 0], np.float32)
    phantom_ear = _fake_hand(ear + [0, fw * 0.15], ear, fw * 0.12)
    assert not hand_is_plausible(phantom_ear, g)
    # 3) gerçek el: aşağıdan gelip bıyığa dokunuyor -> kabul
    lip = face_px[0]
    real = _fake_hand(lip + [0, fw * 1.0], lip + [0, -fw * 0.05], fw * 0.55)
    assert hand_is_plausible(real, g)
    # 4) gerçek el: yandan gelip bıyığa dokunuyor -> kabul
    real_side = _fake_hand(face_px[61] + [-fw * 0.9, fw * 0.3], face_px[61], fw * 0.5)
    assert hand_is_plausible(real_side, g)
    # sınıflandırmada da hayali el alarm üretmemeli
    st = classify(g.zone, fw, [real])[0]
    assert st in (STATE_TOUCH, STATE_PULL)


def test_head_turn_no_occlusion_alarm():
    """Baş yana dönükken örtülme kontrolü (görüntü değişimi) alarm üretmemeli."""
    det = MustacheDetector(video_mode=False)
    clean = cv2.imread(os.path.join(IN, "neg_01_el_yok.jpg"))
    for _ in range(8):
        det.process(clean)
    # Başın dönmesini taklit: görüntüyü yatayda sıkıştır (yüz daralır, bıyık görüntüsü değişir)
    # + yanağa yakın bir el
    img = cv2.imread(os.path.join(IN, "neg_03_parmak_yanakta.jpg"))
    h, w = img.shape[:2]
    stretched = cv2.resize(img[:, : int(w * 0.85)], (w, h))
    r = det.process(stretched)
    det.close()
    assert r.state not in (STATE_TOUCH, STATE_PULL), r.state


def test_zone_covers_full_mustache():
    """Bıyık bölgesi, demo yüze çizilen bıyığın tamamını (koyu pikseller) kapsamalı."""
    face_px, g = _portrait_geom()
    img = cv2.imread(os.path.join(IN, "neg_01_el_yok.jpg"))
    # bıyık pikselleri: burun altı ile dudak arasındaki çok koyu pikseller
    x0, x1 = int(face_px[61][0] - 20), int(face_px[291][0] + 20)
    y0, y1 = int(face_px[2][1]), int(face_px[13][1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray[y0:y1, x0:x1] < 50)
    contour = g.zone.reshape(-1, 1, 2)
    inside = [cv2.pointPolygonTest(contour, (float(x + x0), float(y + y0)), True) >= -g.fw * 0.02
              for x, y in zip(xs, ys)]
    assert len(inside) > 50
    assert np.mean(inside) >= 0.95, np.mean(inside)


# ---------------------------------------------------------------- v1.2: yüze dışarıdan giriş
def _run_video(frames, hold=0.4):
    det = MustacheDetector(video_mode=True)
    alarm = AlarmLogic(touch_hold_s=hold, pull_hold_s=0.15, release_s=1.0)
    t, out = 0.0, []
    for fr in frames:
        r = det.process(fr, now=t)
        out.append((r.state, alarm.update(r.state, t), len(r.rejected_hands)))
        t += 1 / 15
    det.close()
    return out


def test_fist_entering_from_below_alarms():
    """Yumruk kadrajın altından yükselip bıyığa gelirse (gerçek el davranışı) alarm çalmalı."""
    seq_dir = os.path.join(IN, "dizi")
    if not glob.glob(os.path.join(seq_dir, "*.jpg")):
        from tests import make_demo_images
        make_demo_images.main()
    clean = cv2.imread(os.path.join(IN, "neg_01_el_yok.jpg"))
    seq = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))]
    frames = [clean] * 8 + [f for f in seq for _ in range(2)] + [seq[-1]] * 15
    out = _run_video(frames)
    assert any(a == "alarm" for _, a, _ in out), [s for s, _, _ in out]


def test_hand_tracks_entry_rule():
    """
    Yüzün içinde 'beliren' ve hep orada kalan el (yüz şekillerinden uydurulan sahte el)
    asla güvenilir sayılmaz; yüzün dışından gelip içeri giren el güvenilir sayılır.
    """
    from trikokontrol.detector import HandTracks
    face_px, g = _portrait_geom()
    fw = g.fw
    lip = face_px[0]
    inside = _fake_hand(lip + [fw * 0.05, fw * 0.15], lip, fw * 0.4)      # bilek yüzün içinde
    outside = _fake_hand(lip + [0, fw * 1.2], lip + [0, fw * 0.4], fw * 0.5)  # bilek çenenin altında
    tr = HandTracks()
    t = 0.0
    for _ in range(60):  # 4 sn boyunca yüzün içinde duran sahte el
        assert tr.update([inside], g, t) == [False]
        t += 1 / 15
    tr.reset()
    # gerçek el: aşağıdan yükselip bıyığa gelir, bileği sonunda yüzün içine girer
    for k in np.linspace(0, 1, 10):
        hand = outside + (inside - outside) * k
        res = tr.update([hand], g, t)
        t += 1 / 15
    assert res == [True]
    # el 1 sn kaybolursa güven sıfırlanır; yüzün içinde yeniden beliren el güvenilmez
    t += 1.0
    assert tr.update([inside], g, t) == [False]


def test_thin_v_phantom_shape_rejected():
    """Ekran görüntüsündeki sahte el: gözlük sapından ağız kenarına inen ince 'V' iskelet."""
    from trikokontrol.detector import hand_shape_ok
    face_px, g = _portrait_geom()
    fw = g.fw
    wrist = face_px[61] + np.array([0, -fw * 0.02], np.float32)       # ağız kenarı
    top = face_px[127] + np.array([fw * 0.05, -fw * 0.1], np.float32)  # gözlük sapı / şakak
    h = np.zeros((21, 2), np.float32)
    for i in range(21):  # tüm noktalar bilekten şakağa giden ince iki çizgi üzerinde
        k = (i % 5 + 1) / 5
        side = np.array([fw * 0.03 * (1 if i % 2 else -1), 0], np.float32)
        h[i] = wrist + (top - wrist) * k + side * k
    h[0] = wrist
    assert not hand_shape_ok(h)
    # gerçek el iskeletleri (demo görüntülerdeki) şekil kontrolünden geçmeli
    det = MustacheDetector(video_mode=False)
    for f in ["poz_01_isaret_parmagi_biyik_sol.jpg", "poz_04_yumruk_cimdik_biyik.jpg",
              "poz_05_basparmak_biyik.jpg"]:
        r = det.process(cv2.imread(os.path.join(IN, f)))
        assert r.hand_points and all(hand_shape_ok(hp) for hp in r.hand_points), f
    det.close()
