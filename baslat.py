"""Trikotillomani Kontrol başlatıcı:  python baslat.py"""
import os
import sys

# Pencereli (.exe / pythonw) modda konsol yoktur; çıktıları dosyaya yönlendir.
if sys.stdout is None or sys.stderr is None:
    _base = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
    try:
        _log = open(os.path.join(_base, "hata_kaydi.txt"), "a", encoding="utf-8", buffering=1)
    except OSError:
        _log = open(os.devnull, "w")
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

from trikokontrol.app import main  # noqa: E402

if __name__ == "__main__":
    main()
