"""Alarm zamanlama mantığı ve sistem aksiyonları (ses, ekranı kapatma, kilitleme)."""
from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import tempfile
import threading
import wave
from typing import Optional

from .detector import STATE_PULL, STATE_TOUCH

IDLE, ALARM, ESCALATE = "idle", "alarm", "escalate"


class AlarmLogic:
    """
    Durum makinesi:
      - Temas touch_hold_s boyunca sürerse alarm (kısa kaşımalar alarm vermez).
      - Yolma (çimdik) pull_hold_s sonra alarm (daha hızlı).
      - El release_s boyunca bıyıktan uzak kalırsa alarm kapanır.
      - Alarm escalate_s boyunca sürerse üst seviye (ekranı kapat / kilitle).
    """

    def __init__(self, touch_hold_s=0.5, pull_hold_s=0.15, release_s=1.5,
                 escalate_s=5.0, grace_s=0.3):
        self.touch_hold_s = touch_hold_s
        self.pull_hold_s = pull_hold_s
        self.release_s = release_s
        self.escalate_s = escalate_s
        self.grace_s = grace_s  # algılamadaki anlık kopmaları tolere eder
        self.active = False
        self.events = 0
        self.contact_start: Optional[float] = None
        self.last_contact: Optional[float] = None
        self.alarm_start: Optional[float] = None
        self.pull_seen = False

    def update(self, state: str, t: float) -> str:
        contact = state in (STATE_TOUCH, STATE_PULL)
        if contact:
            if self.last_contact is None or t - self.last_contact > self.grace_s:
                self.contact_start = t
                self.pull_seen = False
            self.pull_seen |= state == STATE_PULL
            self.last_contact = t
            hold = self.pull_hold_s if self.pull_seen else self.touch_hold_s
            if not self.active and t - self.contact_start >= hold:
                self.active = True
                self.alarm_start = t
                self.events += 1
        if self.active and self.last_contact is not None and t - self.last_contact >= self.release_s:
            self.active = False
            self.alarm_start = None
        if not self.active:
            return IDLE
        if self.escalate_s and t - self.alarm_start >= self.escalate_s and \
                t - self.last_contact < self.release_s:
            return ESCALATE
        return ALARM


# ----------------------------------------------------------------- ses
def make_siren_wav(path: str, seconds: float = 1.2, rate: int = 22050) -> str:
    """Dikkat çekici, iki tonlu siren sesi üretir."""
    frames = bytearray()
    n = int(seconds * rate)
    for i in range(n):
        t = i / rate
        f = 880 if int(t * 6) % 2 == 0 else 1320
        v = 0.9 * math.sin(2 * math.pi * f * t)
        v = 1.0 if v > 0.3 else (-1.0 if v < -0.3 else v / 0.3)  # sert/keskin ton
        frames += struct.pack("<h", int(v * 30000))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return path


class SoundPlayer:
    def __init__(self, custom_sound: str = ""):
        if custom_sound and os.path.exists(custom_sound):
            self.path = custom_sound
        else:
            self.path = make_siren_wav(os.path.join(tempfile.gettempdir(), "trikokontrol_alarm.wav"))
        self._playing = False
        self._proc = None
        self._thread = None

    def start(self):
        if self._playing:
            return
        self._playing = True
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(self.path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        else:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        cmds = [["afplay", self.path]] if sys.platform == "darwin" else \
            [["paplay", self.path], ["aplay", "-q", self.path]]
        while self._playing:
            for c in cmds:
                try:
                    self._proc = subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self._proc.wait()
                    break
                except FileNotFoundError:
                    continue
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
                threading.Event().wait(0.5)

    def stop(self):
        if not self._playing:
            return
        self._playing = False
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(None, 0)
        elif self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


# ----------------------------------------------------------------- sistem
def monitor_off():
    """Monitörü kapatır (fare/klavye hareketiyle tekrar açılır)."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER = 0xFFFF, 0x0112, 0xF170
            ctypes.windll.user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)
        elif sys.platform == "darwin":
            subprocess.Popen(["pmset", "displaysleepnow"])
        else:
            subprocess.Popen(["xset", "dpms", "force", "off"])
    except Exception as e:  # pragma: no cover
        print("Ekran kapatılamadı:", e)


def lock_screen():
    try:
        if sys.platform.startswith("win"):
            import ctypes
            ctypes.windll.user32.LockWorkStation()
        elif sys.platform == "darwin":
            subprocess.Popen(["pmset", "displaysleepnow"])
        else:
            subprocess.Popen(["loginctl", "lock-session"])
    except Exception as e:  # pragma: no cover
        print("Ekran kilitlenemedi:", e)
