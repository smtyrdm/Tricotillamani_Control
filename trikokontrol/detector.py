"""
Bıyık bölgesi + parmak ucu algılayıcı.

Mantık:
  1. MediaPipe Face Landmarker ile yüzün 478 noktası bulunur.
  2. Burun altı (subnazal) ile üst dudak arasındaki noktalardan bir "bıyık bölgesi"
     çokgeni oluşturulur. Bu bölge yalnızca bıyığı kapsar; yanak, alın, çene vb. hariçtir.
  3. MediaPipe Hand Landmarker ile ellerin 21 noktası bulunur.
  4. Başparmak / işaret / orta parmak uçlarından biri bıyık bölgesine (küçük bir pay ile)
     girerse "TEMAS" sayılır. Başparmak ile işaret parmağı birbirine yakın (çimdik) ve
     bıyıkta ise "YOLMA" sayılır.
  5. El bıyığı kapattığında yüz kaybolabilir; bu yüzden son görülen bıyık bölgesi
     birkaç saniye hafızada tutulur.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .paths import model_path

# Bıyık bölgesini çevreleyen yüz noktaları (MediaPipe Face Mesh indeksleri)
# Üst dudağın dış kenarı (ağız köşesinden ağız köşesine)
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
# Burun altı / burun kanadı tabanı (sağdan sola, çokgeni kapatmak için)
NOSE_BASE = [410, 322, 391, 393, 164, 167, 165, 92, 186]
NOSE_EXTRA = [2, 98, 327, 97, 326]
MUSTACHE_IDX = UPPER_LIP + NOSE_BASE + NOSE_EXTRA

FACE_LEFT, FACE_RIGHT = 234, 454  # yüz genişliği için
CHIN, FOREHEAD = 152, 10

# Kontrol edilen parmak uçları
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
INDEX_DIP, THUMB_IP = 7, 3
CHECK_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, INDEX_DIP, THUMB_IP]

STATE_NONE = "YOK"
STATE_NEAR = "YAKIN"
STATE_TOUCH = "TEMAS"
STATE_PULL = "YOLMA"


@dataclass
class DetectionResult:
    state: str = STATE_NONE
    face_found: bool = False
    face_from_memory: bool = False
    hands: int = 0
    zone: Optional[np.ndarray] = None  # piksel cinsinden çokgen (N,2)
    face_width: float = 0.0
    hand_points: List[np.ndarray] = field(default_factory=list)  # her el için (21,2)
    hit_points: List[Tuple[int, int]] = field(default_factory=list)
    min_distance: float = float("inf")  # parmak ucunun bölgeye uzaklığı (yüz genişliği oranı)
    occlusion: float = 0.0  # bıyık bölgesinin ne kadarının örtüldüğü (0-1)
    occluded_touch: bool = False  # temas, örtülme yedek kontrolüyle mi bulundu


PATCH_W, PATCH_H = 64, 24
PATCH_ANCHORS = [61, 291, 2]  # sol ağız köşesi, sağ ağız köşesi, burun altı
PATCH_DST = np.array([[0, PATCH_H - 2], [PATCH_W, PATCH_H - 2], [PATCH_W / 2, 1]], np.float32)


def mustache_patch(rgb: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Bıyık bölgesini yüz pozisyonundan bağımsız sabit boyutlu bir yamaya çevirir (Lab)."""
    M = cv2.getAffineTransform(anchors.astype(np.float32), PATCH_DST)
    patch = cv2.warpAffine(rgb, M, (PATCH_W, PATCH_H), flags=cv2.INTER_AREA,
                           borderMode=cv2.BORDER_REPLICATE)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    return cv2.cvtColor(patch.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _to_px(landmarks, w: int, h: int, idx=None) -> np.ndarray:
    if idx is None:
        return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx], dtype=np.float32)


def mustache_zone(face_px: np.ndarray) -> Tuple[np.ndarray, float]:
    """Yüz noktalarından bıyık bölgesi (dışbükey zarf) ve yüz genişliğini döndürür."""
    pts = face_px[MUSTACHE_IDX]
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    fw = float(np.linalg.norm(face_px[FACE_LEFT] - face_px[FACE_RIGHT]))
    return hull, max(fw, 1.0)


def classify(zone: np.ndarray, face_width: float, hands_px: List[np.ndarray],
             margin: float = 0.07, near_margin: float = 0.25,
             pinch_ratio: float = 0.22) -> Tuple[str, List[Tuple[int, int]], float]:
    """
    Saf geometri: bölge + el noktaları -> durum.
    margin      : bölgenin dışına taşan tolerans (yüz genişliği oranı)
    near_margin : "YAKIN" uyarısı için tolerans
    pinch_ratio : başparmak-işaret arası bu orandan küçükse çimdik sayılır
    """
    state = STATE_NONE
    hits: List[Tuple[int, int]] = []
    best = float("inf")
    contour = zone.reshape(-1, 1, 2).astype(np.float32)
    for hand in hands_px:
        tip_d = {}
        for t in CHECK_TIPS:
            p = hand[t]
            # pointPolygonTest: içeride pozitif, dışarıda negatif mesafe
            d = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)
            dist_ratio = max(0.0, -d) / face_width
            tip_d[t] = dist_ratio
            best = min(best, dist_ratio)
            if dist_ratio <= margin:
                hits.append((int(p[0]), int(p[1])))
        touching = any(tip_d[t] <= margin for t in CHECK_TIPS)
        if touching:
            pinch = np.linalg.norm(hand[THUMB_TIP] - hand[INDEX_TIP]) / face_width
            pinch_in_zone = (tip_d[THUMB_TIP] <= margin * 1.5 or tip_d[INDEX_TIP] <= margin * 1.5)
            if pinch <= pinch_ratio and pinch_in_zone:
                state = STATE_PULL
            elif state != STATE_PULL:
                state = STATE_TOUCH
        elif state == STATE_NONE and min(tip_d.values()) <= near_margin:
            state = STATE_NEAR
    return state, hits, best


class MustacheDetector:
    """MediaPipe modellerini sarmalayan, kare kare çalışan algılayıcı."""

    def __init__(self, video_mode: bool = True, zone_memory_s: float = 3.0,
                 margin: float = 0.07, near_margin: float = 0.25, pinch_ratio: float = 0.22,
                 crop_scale: float = 2.4, occlusion_check: bool = True,
                 occlusion_threshold: float = 0.40):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        self.video_mode = video_mode
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("face_landmarker.task")),
            running_mode=mode, num_faces=1,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
            running_mode=mode, num_hands=2,
            min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35))
        # Yüz çevresine yakınlaştırılmış ikinci el algılayıcı: yüze yakın / yüzü örten
        # elleri (yumruk, çimdik) tam karede kaçırmamak için.
        self.hand_crop = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
        self.crop_scale = crop_scale
        self.zone_memory_s = zone_memory_s
        self.margin = margin
        self.near_margin = near_margin
        self.pinch_ratio = pinch_ratio
        self.occlusion_check = occlusion_check
        self.occlusion_threshold = occlusion_threshold
        self._baseline: Optional[np.ndarray] = None
        self._baseline_n = 0
        self._last_anchors: Optional[np.ndarray] = None
        self._last_zone: Optional[np.ndarray] = None
        self._last_fw = 0.0
        self._last_zone_t = 0.0
        self._t0 = time.monotonic()
        self._last_ts = -1

    def close(self):
        try:
            self.face.close()
            self.hand.close()
            self.hand_crop.close()
        except Exception:
            pass

    def process(self, frame_bgr: np.ndarray, now: Optional[float] = None) -> DetectionResult:
        now = time.monotonic() if now is None else now
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video_mode:
            ts = int((now - self._t0) * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            fr = self.face.detect_for_video(image, ts)
            hr = self.hand.detect_for_video(image, ts)
        else:
            fr = self.face.detect(image)
            hr = self.hand.detect(image)

        res = DetectionResult()
        if fr.face_landmarks:
            face_px = _to_px(fr.face_landmarks[0], w, h)
            zone, fw = mustache_zone(face_px)
            self._last_zone, self._last_fw, self._last_zone_t = zone, fw, now
            self._last_anchors = face_px[PATCH_ANCHORS]
            res.face_found = True
        elif self._last_zone is not None and now - self._last_zone_t <= self.zone_memory_s:
            zone, fw = self._last_zone, self._last_fw
            res.face_from_memory = True
        else:
            zone, fw = None, 0.0

        res.hand_points = [_to_px(hl, w, h) for hl in hr.hand_landmarks]
        if zone is not None and self.crop_scale:
            res.hand_points += self._crop_hands(rgb, zone, fw, res.hand_points)
        res.hands = len(res.hand_points)
        res.zone, res.face_width = zone, fw
        if zone is not None and res.hand_points:
            res.state, res.hit_points, res.min_distance = classify(
                zone, fw, res.hand_points, self.margin, self.near_margin, self.pinch_ratio)
        if zone is not None and self.occlusion_check:
            self._occlusion_fallback(rgb, res)
        return res

    def reset_baseline(self):
        self._baseline, self._baseline_n = None, 0

    def _occlusion_fallback(self, rgb, res: DetectionResult):
        """
        Yedek kontrol: el bıyığın hemen yanındaysa ve bıyık bölgesinin görüntüsü
        (elsiz haldeki öğrenilmiş görüntüsüne göre) belirgin şekilde değiştiyse,
        parmak uçları doğru bulunamamış olsa bile TEMAS say.
        """
        zone, fw = res.zone, res.face_width
        contour = zone.reshape(-1, 1, 2).astype(np.float32)
        hand_d = float("inf")
        for hp in res.hand_points:
            for p in hp:
                d = -cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True) / fw
                hand_d = min(hand_d, d)
        if self._last_anchors is None:
            return
        patch = mustache_patch(rgb, self._last_anchors)
        hand_far = hand_d > 1.0
        if res.face_found and hand_far:
            # Elsiz görüntüyü öğren (ışık değişimine uyum için yavaşça güncellenir)
            if self._baseline is None:
                self._baseline = patch.copy()
            else:
                self._baseline = 0.9 * self._baseline + 0.1 * patch
            self._baseline_n += 1
            return
        if hand_d > 0.5:
            return
        if res.face_from_memory and hand_d <= 0.3 and res.state not in (STATE_TOUCH, STATE_PULL):
            # El bıyık bölgesinin yanındayken yüz kayboldu: büyük ihtimalle el yüzü örtüyor
            res.state, res.occluded_touch = STATE_TOUCH, True
        if self._baseline is None or self._baseline_n < 5:
            return
        changed = (np.linalg.norm(patch - self._baseline, axis=2) > 15.0).astype(np.float32)
        # Yerel örtülme: bıyık yamasında parmak genişliğinde (~10 sütun) en çok değişen şerit
        cols = changed.mean(axis=0)
        res.occlusion = float(np.convolve(cols, np.ones(10) / 10, mode="valid").max())
        if res.occlusion >= self.occlusion_threshold and res.state not in (STATE_TOUCH, STATE_PULL):
            res.state, res.occluded_touch = STATE_TOUCH, True

    def _crop_hands(self, rgb, zone, fw, existing, size=320):
        """Bıyık merkezli kare kırpımda el arar, bulunanları tam kare koordinatına çevirir."""
        cx, cy = zone.mean(axis=0)
        side = fw * self.crop_scale
        x0, y0 = cx - side / 2, cy - side / 2
        k = size / side
        M = np.array([[k, 0, -x0 * k], [0, k, -y0 * k]], dtype=np.float32)
        crop = cv2.warpAffine(rgb, M, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        r = self.hand_crop.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                 data=np.ascontiguousarray(crop)))
        out = []
        for hl in r.hand_landmarks:
            pts = np.array([(lm.x * size / k + x0, lm.y * size / k + y0) for lm in hl], np.float32)
            # tam karede zaten bulunmuş elin kopyası ise ekleme
            dup = any(np.median(np.linalg.norm(e - pts, axis=1)) < fw * 0.15 for e in existing)
            if not dup:
                out.append(pts)
        return out


HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

STATE_COLORS = {STATE_NONE: (0, 200, 0), STATE_NEAR: (0, 200, 255),
                STATE_TOUCH: (0, 0, 255), STATE_PULL: (255, 0, 255)}


def draw(frame: np.ndarray, res: DetectionResult, label: bool = True) -> np.ndarray:
    out = frame.copy()
    color = STATE_COLORS.get(res.state, (255, 255, 255))
    if res.zone is not None:
        cv2.polylines(out, [res.zone.astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
    for hp in res.hand_points:
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, tuple(hp[a].astype(int)), tuple(hp[b].astype(int)), (255, 255, 0), 2)
        for t in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP):
            cv2.circle(out, tuple(hp[t].astype(int)), 5, (255, 0, 0), -1)
    for p in res.hit_points:
        cv2.circle(out, p, 10, (0, 0, 255), 3)
    if label:
        draw_label(out, res)
    return out


def draw_label(out: np.ndarray, res: DetectionResult) -> None:
    color = STATE_COLORS.get(res.state, (255, 255, 255))
    txt = f"Durum: {res.state}"
    if res.face_from_memory:
        txt += " (yuz hafizadan)"
    if res.occluded_touch:
        txt += f" (bolge ortuldu %{int(res.occlusion * 100)})"
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
