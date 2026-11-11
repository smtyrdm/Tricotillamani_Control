import json
import os

from .paths import user_dir

DEFAULTS = {
    "kamera_no": 0,                 # birden fazla kamera varsa 1, 2 ... deneyin
    "kamera_genislik": 640,
    "kamera_yukseklik": 480,
    "hassasiyet": 0.07,             # bıyık bölgesi toleransı (yüz genişliği oranı). Büyük = daha hassas
    "temas_bekleme_sn": 0.4,        # parmak bu kadar süre bıyıkta kalırsa alarm
    "yolma_bekleme_sn": 0.1,        # çimdik (yolma) hareketinde alarm gecikmesi
    "birakma_sn": 1.5,              # el bu kadar süre uzak kalınca alarm kapanır
    "ses": True,
    "kirmizi_ekran": True,          # tam ekran yanıp sönen uyarı
    "ekrani_kapat": True,           # alarm devam ederse monitörü kapat
    "ekrani_kilitle": False,        # alarm devam ederse Windows'u kilitle
    "ust_seviye_sn": 5.0,           # alarm bu kadar sürerse ekranı kapat / kilitle
    "ortu_algilama": True,          # parmak ucu görünmese bile bıyığın örtülmesini algıla
    "onizleme": True,
    "ozel_ses_dosyasi": "",         # isterseniz kendi .wav dosyanızın yolu
    "uyari_metni": "ELİNİ BIYIĞINDAN ÇEK!",
    "uyari_alt_metni": "Derin bir nefes al. Ellerini masaya koy. Bu dürtü geçecek.",
}

CONFIG_PATH = os.path.join(user_dir(), "ayarlar.json")


def load() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print("ayarlar.json okunamadı, varsayılanlar kullanılıyor:", e)
    return cfg


def save(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("ayarlar kaydedilemedi:", e)
