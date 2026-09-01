"""YOLO11 ile kisi tespiti, ByteTrack ile takip.

Bu dosya ultralytics'e dokunan tek yer. Model calistirmak ile takip cikitisini
yorumlamak bilerek ayrildi:

`to_frame_detections()` ham sayilari tiplere cevirir ve saftir.
`detect_persons()` modeli calistirir ve saf degildir.

Ayrimin sebebi, ultralytics'in kurulu olmadigi ya da modelin indirilmedigi bir
ortamda da secim ve kimlik analizi mantiginin test edilebilmesi. `ultralytics`
import'u bu yuzden modul seviyesinde degil, fonksiyon govdesinde.

Model agirligi ilk calistirmada indirilir (yolo11m ~40 MB) ve calisma dizinine
yazilir; .gitignore *.pt dosyalarini disarida tutuyor.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.pose.boxes import BBox, Detection, DetectionSequence, FrameDetections
from core.pose.config import PoseConfig

__all__ = ["PERSON_CLASS_ID", "detect_persons", "to_frame_detections"]

# COCO sinif indeksinde "person". YOLO'ya yalnizca bu sinifi ariyoruz diyoruz;
# pistteki bariyer, koni veya reklam panosu tespit edilmesin.
PERSON_CLASS_ID = 0


def to_frame_detections(
    frame_index: int,
    boxes_xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    track_ids: Sequence[int | None] | None = None,
) -> FrameDetections:
    """Ham takip ciktisini tiplere cevir. Saf.

    `track_ids` None olabilir: ByteTrack bir karede hicbir kutuya kimlik
    atayamadiginda ultralytics `boxes.id` alanini bos birakir. Kimliksiz
    tespitleri atmiyoruz, `track_id=None` ile tasiyoruz; kac kisi goruldugunu
    bilmek kalabalik uyarisi icin gerekli.
    """
    if len(boxes_xyxy) != len(confidences):
        msg = (
            f"kutu sayisi ({len(boxes_xyxy)}) ile "
            f"guven sayisi ({len(confidences)}) uyusmuyor"
        )
        raise ValueError(msg)
    if track_ids is not None and len(track_ids) != len(boxes_xyxy):
        msg = f"kutu sayisi ({len(boxes_xyxy)}) ile kimlik sayisi ({len(track_ids)}) uyusmuyor"
        raise ValueError(msg)

    detections: list[Detection] = []
    for index, (box, confidence) in enumerate(zip(boxes_xyxy, confidences, strict=True)):
        x1, y1, x2, y2 = (float(value) for value in box)
        track_id = track_ids[index] if track_ids is not None else None
        detections.append(
            Detection(
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(confidence),
                track_id=int(track_id) if track_id is not None else None,
            )
        )
    return FrameDetections(frame_index=frame_index, detections=tuple(detections))


def _extract(result: Any) -> tuple[list[list[float]], list[float], list[int | None] | None]:
    """Bir ultralytics sonucundan kutulari, guvenleri ve kimlikleri cikar."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], [], None

    xyxy = [[float(v) for v in row] for row in boxes.xyxy.tolist()]
    confidences = [float(v) for v in boxes.conf.tolist()]
    if boxes.id is None:
        return xyxy, confidences, None
    ids: list[int | None] = [int(v) for v in boxes.id.tolist()]
    return xyxy, confidences, ids


def detect_persons(
    video_path: Path, config: PoseConfig | None = None
) -> DetectionSequence:
    """Videodaki kisileri tespit et ve klip boyunca takip et.

    Saf degil: modeli yukler, videoyu okur, gerekiyorsa agirlik indirir.

    Takip durumu klip boyunca korunur (`persist=True`); her kare bagimsiz
    islenirse kimlik diye bir sey kalmaz. Kareler sirayla gelir ve
    `frame_index` okuma sirasina gore artar, yani `PoseSequence` ile ayni
    indeksleme.
    """
    # Agir import; yalnizca gercekten calistirirken. ultralytics __all__ tanimlamadigi
    # icin mypy'a bu satirda goz yumuyoruz; sembol calisma aninda mevcut.
    from ultralytics import YOLO  # type: ignore[attr-defined]

    cfg = config or PoseConfig()
    if not video_path.is_file():
        msg = f"video yok: {video_path}"
        raise FileNotFoundError(msg)

    model = YOLO(cfg.model_path)
    stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=cfg.tracker,
        classes=[PERSON_CLASS_ID],
        conf=cfg.min_detection_confidence,
        device=cfg.device,
        verbose=False,
    )

    frames: list[FrameDetections] = []
    width = height = 0
    for frame_index, result in enumerate(stream):
        if not width:
            height, width = (int(v) for v in result.orig_shape[:2])
        boxes_xyxy, confidences, track_ids = _extract(result)
        frames.append(to_frame_detections(frame_index, boxes_xyxy, confidences, track_ids))

    if not frames:
        msg = f"videodan hic kare okunamadi: {video_path}"
        raise ValueError(msg)

    return DetectionSequence(width=width, height=height, frames=tuple(frames))
