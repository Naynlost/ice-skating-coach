"""Kare cikarma.

Kareler diske yazilmaz, bellekte akitilir. 5 saniyelik 120 fps bir klip 600
kare; 1280x720 BGR olarak diske yazmak 1.5 GB eder ve poz tahmini zaten
kareleri sirayla tuketir.

Kare indeksi 0 tabanli ve okunan sirayla artar; `FrameBatch.index` degeri
`PoseSequence` icindeki kare indeksiyle ayni seyi gosterir.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.ingest.probe import VideoUnreadableError

__all__ = ["Frame", "count_frames", "iter_frames", "read_frame"]


@dataclass(frozen=True, slots=True)
class Frame:
    """Tek bir kare.

    `image` OpenCV'nin verdigi BGR dizisi, sekli (yukseklik, genislik, 3).
    Pydantic degil dataclass: her karede dogrulama calistirmak 600 karelik bir
    klipte olcusuz maliyet, ve dizinin kendisi zaten sema tasimiyor.
    """

    index: int
    image: np.ndarray

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


def iter_frames(path: Path, *, start: int = 0, end: int | None = None) -> Iterator[Frame]:
    """Videoyu kare kare oku.

    `start` ve `end` yari acik aralik: [start, end). `end` None ise klip sonuna
    kadar gider.

    Kareye atlamak yerine bastan sirayla okuyup istenmeyenleri grab() ile
    geciyoruz. Bazi codec'lerde CAP_PROP_POS_FRAMES ile atlamak sessizce
    komsu bir kareyi verir; faz sinirlarinin tek kareye duyarli oldugu bir
    projede bu kabul edilemez.
    """
    if start < 0:
        msg = f"start negatif olamaz: {start}"
        raise ValueError(msg)
    if end is not None and end < start:
        msg = f"end ({end}) start'tan ({start}) kucuk olamaz"
        raise ValueError(msg)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        msg = f"video acilamadi: {path}"
        raise VideoUnreadableError(msg)

    try:
        index = 0
        while end is None or index < end:
            if index < start:
                if not capture.grab():
                    return
                index += 1
                continue
            ok, image = capture.read()
            if not ok:
                return
            yield Frame(index=index, image=image)
            index += 1
    finally:
        capture.release()


def read_frame(path: Path, index: int) -> Frame:
    """Tek bir kareyi oku. pose-debug skill'indeki kare izole etme akisi icin."""
    for frame in iter_frames(path, start=index, end=index + 1):
        return frame
    msg = f"kare {index} okunamadi: {path}"
    raise VideoUnreadableError(msg)


def count_frames(path: Path) -> int:
    """Klipteki kareleri gercekten okuyarak say.

    ffprobe'un `nb_frames` degeri bazi konteynerlerde konteyner ust verisinden
    gelir ve yanlis olabilir. Kare sayisinin dogru olmasi gereken yerlerde
    (kare indeksi dogrulamasi) bunu kullan; hizli tahmin icin probe yeterli.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        msg = f"video acilamadi: {path}"
        raise VideoUnreadableError(msg)
    try:
        total = 0
        while capture.grab():
            total += 1
    finally:
        capture.release()
    return total
