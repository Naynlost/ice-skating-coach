"""Kisi tespiti, takip ve sporcu secimi ayarlari.

Esikler koda gomulu degil: varsayilanlar burada, `config/pose.yaml` varsa
uzerine yazar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["DEFAULT_CONFIG_PATH", "PoseConfig"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pose.yaml"


class PoseConfig(BaseModel):
    """Tespit/takip modeli ve sporcu secim olcutleri.

    Model varyanti bilerek `m`: `n` (nano) hizli ama hareket bulaniklıginda
    kisiyi kaciriyor. Kacirilan kare, o karede keypoint yok demek; o da
    olculemeyen metrik demek. 30 videoluk bir sette hiz onemsiz, dogruluk
    onemli.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- model
    model_path: str = "yolo11m.pt"
    tracker: str = "bytetrack.yaml"
    # None ise ultralytics kendisi secer (GPU varsa GPU).
    device: str | None = None
    min_detection_confidence: float = Field(default=0.4, ge=0.0, le=1.0)

    # --- sporcu secimi
    # Bir takip kimligi, klibin en az bu kadarinda gorunmeliyse aday sayilir.
    # Pistten gecip giden bir patenci genelde klibin kucuk bir bolumunde olur.
    min_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    # "En buyuk ve en merkezi" olcutunun iki bileseninin agirligi.
    area_weight: float = Field(default=0.6, ge=0.0)
    centrality_weight: float = Field(default=0.4, ge=0.0)

    # --- kimlik degisimi uyarilari
    # Ardisik iki karede merkezin, kutu genisliginin bu kati kadar kaymasi
    # kimligin baska bir vucuda atladiginin isaretidir. Sporcu sicrarken bile
    # bir karede kendi genisliginin yarisi kadar yatay yol almaz.
    max_center_jump_ratio: float = Field(default=0.5, gt=0.0)
    # Ardisik karelerde kutu alaninin bu kattan fazla degismesi ayni seyi
    # gosterir: kutu baska bir cisme oturmus.
    max_scale_ratio: float = Field(default=1.6, gt=1.0)
    # Sporcunun kutusu baska bir kutuyla bu kadar ortusuyorsa takip
    # karismaya acik demektir.
    crowding_iou: float = Field(default=0.3, ge=0.0, le=1.0)
    # Bu kadar veya daha kisa bosluklar normal sayilir (anlik kapanma).
    max_gap_frames: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        if self.area_weight + self.centrality_weight <= 0:
            msg = "area_weight ve centrality_weight ikisi birden sifir olamaz"
            raise ValueError(msg)
        return self

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
