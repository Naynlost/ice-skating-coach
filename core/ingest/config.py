"""On isleme esikleri.

Esikler koda gomulu degil: varsayilanlar burada durur, `config/ingest.yaml`
varsa uzerine yazar. Boylece bir kulubun videolari surekli reddediliyorsa
kodu degil yapilandirmayi degistirirsin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DEFAULT_CONFIG_PATH", "IngestConfig"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ingest.yaml"


class IngestConfig(BaseModel):
    """Kabul kapisi ve normalizasyon ayarlari.

    fps ile ilgili iki ayar var ve karistirilmamali:

    `min_fps` bir kabul kapisidir. Sicrama yuksekligi airtime'in KARESIYLE
    orantili oldugu icin kalkis/inis karesindeki tek karelik hata dogrudan
    yukseklige tasar. 30 fps'te bir kare 33 ms; 0.45 sn airtime'da bu %15
    yukseklik hatasi demek. 60 fps'te %7, 120 fps'te %4.

    `target_fps` normalizasyon hedefidir ve varsayilani None. None ise kaynagin
    kare hizi KORUNUR, video yalnizca degisken kare hizindan (VFR) sabit kare
    hizina (CFR) cevrilir. Bilerek bir sayi vermedikce fps dusurme: her
    dusurulen kare kalkis anindaki hassasiyetten goturur.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- kabul kapisi
    min_duration_s: float = Field(default=1.0, gt=0.0)
    max_duration_s: float = Field(default=30.0, gt=0.0)
    min_width: int = Field(default=480, gt=0)
    min_height: int = Field(default=360, gt=0)
    # Bunun altinda kalkis ve inis karesi ayirt edilemez; kabul kapisi.
    min_fps: float = Field(default=24.0, gt=0.0)
    # Ret degil uyari: bu esigin altinda olcum yapilir ama hassasiyet dusuktur.
    warn_fps_below: float = Field(default=50.0, gt=0.0)
    # 0-255 ortalama parlaklik. Buz beyaz oldugu icin normal bir pist videosu
    # 100'un uzerinde olur; 40 alti kapali salon veya pozlama hatasidir.
    min_brightness: float = Field(default=40.0, ge=0.0, le=255.0)
    # Ardisik kareler arasi global kaymanin standart sapmasi, piksel.
    # Sabit tripodda 1-2 piksel; elde cekimde hizla buyur.
    max_shake_px: float = Field(default=8.0, gt=0.0)

    # --- normalizasyon
    target_fps: float | None = Field(default=None, gt=0.0)
    max_width: int = Field(default=1280, gt=0)

    # --- olcum
    # Kalite olcumu icin klipten esit araliklarla kac kare orneklenecek.
    quality_sample_frames: int = Field(default=24, ge=2)

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """YAML'dan yukle. Dosya yoksa varsayilanlarla don."""
        source = path or DEFAULT_CONFIG_PATH
        if not source.is_file():
            return cls()
        raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            msg = f"{source}: sozluk bekleniyor, {type(raw).__name__} geldi"
            raise ValueError(msg)
        return cls.model_validate(raw)
