"""
Trikotillomani Kontrol - ana uygulama (Tkinter arayüzü).

Kamera görüntüsü arka planda işlenir. Parmak bıyık bölgesine gidip orada kalırsa
(veya çimdik/yolma hareketi yapılırsa):
  1) Siren sesi çalar
  2) Ekranın tamamı kırmızı-siyah yanıp sönen bir uyarıyla kaplanır
  3) El hâlâ çekilmezse monitör kapatılır (isteğe bağlı: bilgisayar kilitlenir)
El bıyıktan uzaklaşınca her şey kendiliğinden kapanır.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from . import __version__, config
from .alarm import ALARM, ESCALATE, IDLE, AlarmLogic, SoundPlayer, lock_screen, monitor_off
from .detector import STATE_PULL, STATE_TOUCH, MustacheDetector, draw, draw_label
from .paths import user_dir

LOG_PATH = os.path.join(user_dir(), "kayitlar.csv")


def open_camera(idx, w: int, h: int):
    if isinstance(idx, str):  # test için video dosyası
        return cv2.VideoCapture(idx)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


class Worker(threading.Thread):
    """Kamera + algılama döngüsü (arayüzü dondurmamak için ayrı iş parçacığında)."""

    def __init__(self, cfg: dict, out: "queue.Queue", source=None):
        super().__init__(daemon=True)
        self.cfg = cfg
        # kaynak: kamera numarası veya (test için) video dosyası yolu
        self.source = cfg["kamera_no"] if source is None else source
        self.out = out
        self.stop_evt = threading.Event()
        self.paused = threading.Event()
        self.error = None
        self.alarm = AlarmLogic(touch_hold_s=cfg["temas_bekleme_sn"],
                                pull_hold_s=cfg["yolma_bekleme_sn"],
                                release_s=cfg["birakma_sn"],
                                escalate_s=cfg["ust_seviye_sn"])
        self.fps = 0.0

    def apply_cfg(self, cfg: dict):
        self.cfg = cfg
        self.alarm.touch_hold_s = cfg["temas_bekleme_sn"]
        self.alarm.pull_hold_s = cfg["yolma_bekleme_sn"]
        self.alarm.release_s = cfg["birakma_sn"]
        self.alarm.escalate_s = cfg["ust_seviye_sn"]
        if hasattr(self, "det"):
            self.det.margin = cfg["hassasiyet"]
            self.det.occlusion_check = cfg["ortu_algilama"]
            self.det.zone_scale_x = cfg["biyik_bolgesi"]
            self.det.zone_scale_y = cfg["biyik_bolgesi"] * 1.16

    def run(self):
        try:
            self.det = MustacheDetector(video_mode=True, margin=self.cfg["hassasiyet"],
                                        occlusion_check=self.cfg["ortu_algilama"],
                                        zone_scale_x=self.cfg["biyik_bolgesi"],
                                        zone_scale_y=self.cfg["biyik_bolgesi"] * 1.16)
        except Exception as e:
            self.error = f"Yapay zekâ modelleri yüklenemedi:\n{e}"
            return
        cap = open_camera(self.source, self.cfg["kamera_genislik"], self.cfg["kamera_yukseklik"])
        if not cap.isOpened():
            self.error = (f"Kamera açılamadı (kamera_no={self.cfg['kamera_no']}).\n"
                          "Kameranın takılı olduğundan ve başka bir programın kullanmadığından emin olun.\n"
                          "Birden fazla kamera varsa ayarlar.json içinde kamera_no değerini 1 yapın.")
            return
        fails = 0
        last = time.monotonic()
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok and isinstance(self.source, str):  # video bitti -> başa sar
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if isinstance(self.source, str):
                time.sleep(1 / 15)
            if not ok:
                fails += 1
                if fails > 50:
                    self.error = "Kameradan görüntü alınamıyor."
                    break
                time.sleep(0.05)
                continue
            fails = 0
            now = time.monotonic()
            if self.paused.is_set():
                self.alarm.update("YOK", now)
                self._push(frame, None, IDLE)
                time.sleep(0.1)
                continue
            res = self.det.process(frame, now)
            astate = self.alarm.update(res.state, now)
            dtm = now - last
            last = now
            self.fps = 0.9 * self.fps + 0.1 * (1 / dtm if dtm > 0 else 0)
            self._push(frame, res, astate)
        cap.release()
        self.det.close()

    def _push(self, frame, res, astate):
        item = (frame, res, astate, self.alarm.events)
        try:
            self.out.get_nowait()
        except queue.Empty:
            pass
        self.out.put(item)


class Overlay:
    """Tam ekran, en üstte duran, yanıp sönen uyarı penceresi."""

    def __init__(self, root: tk.Tk, text: str, subtext: str):
        self.root = root
        self.text, self.subtext = text, subtext
        self.win = None
        self._blink = False

    def show(self):
        if self.win is not None:
            return
        w = tk.Toplevel(self.root)
        w.overrideredirect(True)
        w.geometry(f"{w.winfo_screenwidth()}x{w.winfo_screenheight()}+0+0")
        w.attributes("-topmost", True)
        w.configure(bg="#c00000")
        self.l1 = tk.Label(w, text=self.text, fg="white", bg="#c00000",
                           font=("Segoe UI", 64, "bold"), wraplength=w.winfo_screenwidth() - 100)
        self.l1.pack(expand=True)
        self.l2 = tk.Label(w, text=self.subtext, fg="white", bg="#c00000", font=("Segoe UI", 28))
        self.l2.pack(pady=(0, 120))
        self.win = w
        self._tick()

    def _tick(self):
        if self.win is None:
            return
        self._blink = not self._blink
        bg = "#000000" if self._blink else "#c00000"
        for x in (self.win, self.l1, self.l2):
            x.configure(bg=bg)
        self.win.lift()
        self.win.attributes("-topmost", True)
        self.root.after(350, self._tick)

    def hide(self):
        if self.win is not None:
            self.win.destroy()
            self.win = None


class App:
    def __init__(self, root: tk.Tk, cfg: dict, source=None):
        self.root = root
        self.cfg = cfg
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.sound = SoundPlayer(cfg.get("ozel_ses_dosyasi", ""))
        self.overlay = Overlay(root, cfg["uyari_metni"], cfg["uyari_alt_metni"])
        self.prev_astate = IDLE
        self.alarm_started = None
        self.alarm_kind = ""
        self.escalated = False
        self.test_until = 0.0
        self.today = self._today_count()

        root.title(f"Trikotillomani Kontrol {__version__}")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build_ui()
        self.worker = Worker(cfg, self.q, source)
        self.worker.start()
        self.root.after(30, self._poll)

    # ------------------------------------------------------------ arayüz
    def _build_ui(self):
        r = self.root
        top = ttk.Frame(r, padding=8)
        top.pack(fill="both", expand=True)
        self.video = ttk.Label(top)
        self.video.grid(row=0, column=0, rowspan=20, sticky="n")
        side = ttk.Frame(top, padding=(12, 0))
        side.grid(row=0, column=1, sticky="n")

        self.status = tk.Label(side, text="Başlatılıyor...", font=("Segoe UI", 16, "bold"),
                               fg="white", bg="#555", width=22, pady=8)
        self.status.pack(fill="x")
        self.info = ttk.Label(side, text="İlk birkaç saniye ellerini yüzünden uzak tut\n"
                                         "(bıyığın normal görüntüsü öğreniliyor).", justify="left")
        self.info.pack(anchor="w", pady=6)
        self.count = ttk.Label(side, text="", font=("Segoe UI", 11))
        self.count.pack(anchor="w", pady=4)
        self._update_count()

        ttk.Separator(side).pack(fill="x", pady=6)
        self.vars = {}
        for key, label in [("ses", "Siren sesi"), ("kirmizi_ekran", "Tam ekran kırmızı uyarı"),
                           ("ekrani_kapat", "Devam ederse ekranı kapat"),
                           ("ekrani_kilitle", "Devam ederse bilgisayarı kilitle"),
                           ("ortu_algilama", "Bıyık örtülmesini de algıla"),
                           ("onizleme", "Kamera önizlemesi")]:
            v = tk.BooleanVar(value=bool(self.cfg[key]))
            ttk.Checkbutton(side, text=label, variable=v, command=self._cfg_changed).pack(anchor="w")
            self.vars[key] = v

        ttk.Label(side, text="Bıyık bölgesi büyüklüğü").pack(anchor="w", pady=(10, 0))
        self.zone = tk.DoubleVar(value=self.cfg["biyik_bolgesi"])
        ttk.Scale(side, from_=1.0, to=1.8, variable=self.zone,
                  command=lambda _=None: self._cfg_changed()).pack(fill="x")
        ttk.Label(side, text="Hassasiyet (bölge çevresi toleransı)").pack(anchor="w", pady=(6, 0))
        self.sens = tk.DoubleVar(value=self.cfg["hassasiyet"])
        ttk.Scale(side, from_=0.02, to=0.15, variable=self.sens,
                  command=lambda _=None: self._cfg_changed()).pack(fill="x")
        ttk.Label(side, text="Alarm gecikmesi (sn)").pack(anchor="w", pady=(6, 0))
        self.hold = tk.DoubleVar(value=self.cfg["temas_bekleme_sn"])
        ttk.Scale(side, from_=0.0, to=2.0, variable=self.hold,
                  command=lambda _=None: self._cfg_changed()).pack(fill="x")
        self.hold_lbl = ttk.Label(side, text="")
        self.hold_lbl.pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=6)
        btns = ttk.Frame(side)
        btns.pack(fill="x")
        self.pause_btn = ttk.Button(btns, text="Duraklat", command=self.toggle_pause)
        self.pause_btn.pack(side="left", expand=True, fill="x")
        ttk.Button(btns, text="Alarmı test et", command=self.test_alarm).pack(side="left", expand=True, fill="x")
        ttk.Label(side, text="Kısayol: Ctrl+Shift+P duraklat", foreground="#777").pack(anchor="w", pady=4)
        self.fps_lbl = ttk.Label(side, text="", foreground="#777")
        self.fps_lbl.pack(anchor="w")
        r.bind_all("<Control-Shift-P>", lambda e: self.toggle_pause())
        r.bind_all("<Control-Shift-p>", lambda e: self.toggle_pause())
        self._cfg_changed(save=False)

    def _cfg_changed(self, save=True):
        for k, v in self.vars.items():
            self.cfg[k] = bool(v.get())
        self.cfg["hassasiyet"] = round(float(self.sens.get()), 3)
        self.cfg["biyik_bolgesi"] = round(float(self.zone.get()), 2)
        self.cfg["temas_bekleme_sn"] = round(float(self.hold.get()), 2)
        self.hold_lbl.configure(text=f"gecikme {self.cfg['temas_bekleme_sn']:.1f} sn | bölge x{self.cfg['biyik_bolgesi']:.2f}"
                                     f" | hassasiyet {self.cfg['hassasiyet']:.2f}")
        if hasattr(self, "worker"):
            self.worker.apply_cfg(self.cfg)
        if not self.cfg["onizleme"]:
            self.video.configure(image="", text="(önizleme kapalı)")
        if save:
            config.save(self.cfg)

    def toggle_pause(self):
        if self.worker.paused.is_set():
            self.worker.paused.clear()
            self.pause_btn.configure(text="Duraklat")
        else:
            self.worker.paused.set()
            self.pause_btn.configure(text="Devam et")

    def test_alarm(self):
        self.test_until = time.monotonic() + 3.0

    # ------------------------------------------------------------ kayıt
    def _today_count(self) -> int:
        if not os.path.exists(LOG_PATH):
            return 0
        today = dt.date.today().isoformat()
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                return sum(1 for row in csv.reader(f) if row and row[0].startswith(today))
        except Exception:
            return 0

    def _log(self, kind: str, duration: float):
        new = not os.path.exists(LOG_PATH)
        try:
            with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["zaman", "tur", "sure_sn"])
                w.writerow([dt.datetime.now().isoformat(timespec="seconds"), kind, f"{duration:.1f}"])
        except Exception as e:
            print("kayıt yazılamadı:", e)

    def _update_count(self):
        self.count.configure(text=f"Bugün engellenen girişim: {self.today}")

    # ------------------------------------------------------------ döngü
    def _poll(self):
        if self.worker.error:
            self.sound.stop()
            self.overlay.hide()
            messagebox.showerror("Hata", self.worker.error)
            self.quit()
            return
        try:
            frame, res, astate, _events = self.q.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            if time.monotonic() < self.test_until:
                astate = ALARM
            self._handle_alarm(astate, res)
            self._update_status(res, astate)
            if self.cfg["onizleme"]:
                shown = draw(frame, res, label=False) if res is not None else frame
                shown = cv2.flip(shown, 1)  # ayna görüntüsü
                if res is not None:
                    draw_label(shown, res)
                shown = cv2.resize(shown, (560, int(560 * shown.shape[0] / shown.shape[1])))
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
                self.video.configure(image=img, text="")
                self.video.image = img
            self.fps_lbl.configure(text=f"{self.worker.fps:.0f} kare/sn")
        self.root.after(30, self._poll)

    def _handle_alarm(self, astate, res):
        active = astate in (ALARM, ESCALATE)
        was_active = self.prev_astate in (ALARM, ESCALATE)
        if active and not was_active:
            self.alarm_started = time.monotonic()
            self.escalated = False
            self.alarm_kind = "YOLMA" if (res is not None and res.state == STATE_PULL) else "TEMAS"
            if self.cfg["ses"]:
                self.sound.start()
            if self.cfg["kirmizi_ekran"]:
                self.overlay.show()
        if active and res is not None and res.state == STATE_PULL:
            self.alarm_kind = "YOLMA"
        if astate == ESCALATE and not self.escalated:
            self.escalated = True
            if self.cfg["ekrani_kapat"]:
                monitor_off()
            if self.cfg["ekrani_kilitle"]:
                lock_screen()
        if not active and was_active:
            self.sound.stop()
            self.overlay.hide()
            if time.monotonic() >= self.test_until and self.alarm_started is not None:
                self.today += 1
                self._update_count()
                self._log(self.alarm_kind, time.monotonic() - self.alarm_started)
            self.alarm_started = None
        self.prev_astate = astate

    def _update_status(self, res, astate):
        if self.worker.paused.is_set():
            self.status.configure(text="DURAKLATILDI", bg="#777")
        elif astate in (ALARM, ESCALATE):
            self.status.configure(text="ELİNİ ÇEK!", bg="#c00000")
        elif res is None or res.zone is None:
            self.status.configure(text="Yüz görünmüyor", bg="#996600")
        elif res.state in (STATE_TOUCH, STATE_PULL):
            self.status.configure(text="Dikkat: temas", bg="#d35400")
        elif res.state == "YAKIN":
            self.status.configure(text="El yaklaşıyor", bg="#c9a000")
        else:
            self.status.configure(text="Koruma aktif ✓", bg="#1e8e3e")

    def quit(self):
        self.worker.stop_evt.set()
        self.sound.stop()
        self.overlay.hide()
        self.root.after(200, self.root.destroy)


def run_image(path: str, out: str):
    """Tek görüntü üzerinde algılamayı çalıştırır (test amaçlı)."""
    det = MustacheDetector(video_mode=False)
    img = cv2.imread(path)
    if img is None:
        print("Görüntü okunamadı:", path)
        return 2
    r = det.process(img)
    cv2.imwrite(out, draw(img, r))
    print(f"{os.path.basename(path)}: durum={r.state} el={r.hands} yuz={r.face_found} -> {out}")
    det.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Trikotillomani Kontrol")
    ap.add_argument("--kamera", type=int, help="kamera numarası (0, 1, ...)")
    ap.add_argument("--resim", help="kamera yerine tek bir görüntüyü test et")
    ap.add_argument("--cikti", default="sonuc.jpg", help="--resim için çıktı dosyası")
    ap.add_argument("--video", help="kamera yerine video dosyası kullan (test amaçlı)")
    args = ap.parse_args()
    if args.resim:
        sys.exit(run_image(args.resim, args.cikti))
    cfg = config.load()
    if args.kamera is not None:
        cfg["kamera_no"] = args.kamera
    config.save(cfg)
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    App(root, cfg, source=args.video)
    root.mainloop()


if __name__ == "__main__":
    main()
