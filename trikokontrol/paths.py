import os
import sys


def base_dir() -> str:
    """Uygulama kök klasörü (PyInstaller ile paketlendiğinde de çalışır)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def user_dir() -> str:
    """Ayar ve kayıtların yazılacağı klasör."""
    if getattr(sys, "frozen", False):
        d = os.path.dirname(sys.executable)
    else:
        d = base_dir()
    return d


def model_path(name: str) -> str:
    return os.path.join(base_dir(), "models", name)
