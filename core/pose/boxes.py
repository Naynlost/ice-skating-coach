"""Tespit kutulari ve kare basina tespit listeleri.

Koordinat sistemi: piksel, sol ust kose (0, 0), y ASAGI dogru artar.
Kutular xyxy bicimi: (x1, y1) sol ust, (x2, y2) sag alt.

Bu dosyada model yok, video yok. Yalnizca kutu geometrisi; hepsi saf.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["BBox", "Detection", "DetectionSequence", "FrameDetections"]

_BASE_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class BBox(BaseModel):
    """Eksen hizali sinirlayici kutu, xyxy."""

    model_config = _BASE_CONFIG

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.x2 < self.x1 or self.y2 < self.y1:
            msg = f"gecersiz kutu: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2})"
            raise ValueError(msg)
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def iou(self, other: BBox) -> float:
        """Kesisim / birlesim. Iki kutunun ne kadar ortustugunu 0-1 arasi verir.

        Kalabalik sahne uyarisi bunu kullanir: sporcunun kutusuyla baska bir
        kutu ortusuyorsa takip ID'sinin karsi tarafa atlama riski var.
        """
        inter_w = min(self.x2, other.x2) - max(self.x1, other.x1)
        inter_h = min(self.y2, other.y2) - max(self.y1, other.y1)
        if inter_w <= 0 or inter_h <= 0:
            return 0.0
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def center_distance(self, other: BBox) -> float:
        """Iki kutunun merkezleri arasi oklit mesafesi, piksel."""
        return float(((self.cx - other.cx) ** 2 + (self.cy - other.cy) ** 2) ** 0.5)


class Detection(BaseModel):
    """Tek karede tespit edilmis bir kisi.

    `track_id` None ise takip o karede bu kutuya kimlik atayamamis demektir.
    Kimliksiz tespitler sporcu secimine girmez; hangi kisiye ait olduklarini
    bilmedigimiz icin klip boyunca izlenemezler.
    """

    model_config = _BASE_CONFIG

    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    track_id: int | None = None


class FrameDetections(BaseModel):
    """Bir karedeki tum tespitler."""

    model_config = _BASE_CONFIG

    frame_index: int = Field(ge=0)
    detections: tuple[Detection, ...] = ()

    def by_track(self, track_id: int) -> Detection | None:
        for detection in self.detections:
            if detection.track_id == track_id:
                return detection
        return None

    def others(self, track_id: int) -> tuple[Detection, ...]:
        """Verilen kimlik disindaki tespitler."""
        return tuple(d for d in self.detections if d.track_id != track_id)


class DetectionSequence(BaseModel):
    """Bir klibin kare kare tespitleri.

    Kare boyutu tasiniyor cunku "en merkezi kutu" sorusu kare boyutu olmadan
    cevaplanamaz. Dizi bosluksuz: hic tespit olmayan kare, bos `detections`
    ile yerinde durur.
    """

    model_config = _BASE_CONFIG

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: tuple[FrameDetections, ...]

    @model_validator(mode="after")
    def _check_contiguous(self) -> Self:
        for expected, frame in enumerate(self.frames):
            if frame.frame_index != expected:
                msg = (
                    "kare indeksleri 0'dan itibaren ardisik olmali; "
                    f"{expected}. sirada {frame.frame_index} bulundu"
                )
                raise ValueError(msg)
        return self

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, frame: int) -> FrameDetections:
        return self.frames[frame]

    @property
    def frame_area(self) -> float:
        return float(self.width * self.height)

    @property
    def frame_center(self) -> tuple[float, float]:
        return (self.width / 2, self.height / 2)

    def track_ids(self) -> tuple[int, ...]:
        """Klipte gecen tum takip kimlikleri, ilk gorulme sirasina gore."""
        seen: dict[int, None] = {}
        for frame in self.frames:
            for detection in frame.detections:
                if detection.track_id is not None:
                    seen.setdefault(detection.track_id, None)
        return tuple(seen)
