"""Sentetik poz uretici yardimcilar.

Kok dizinde duruyor ki hem tests/ hem elements/*/tests/ erisebilsin.

Testler videoya bagimli olmamali. Metrik testleri elle kurulmus PoseSequence
uzerinde calisir; boylece test bir GPU'ya, model agirligina veya klasordeki
bir mp4'e bagli olmadan saniyeler icinde doner.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.ingest.probe import VideoInfo
from core.ingest.quality import ClipStats
from core.pose.boxes import BBox, Detection, DetectionSequence, FrameDetections
from core.schema import (
    NUM_KEYPOINTS,
    ElementInstance,
    FrameRange,
    Keypoint,
    KeypointName,
    Pose,
    PoseSequence,
)


def make_keypoint(x: float = 0.0, y: float = 0.0, confidence: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=confidence)


def make_pose(
    frame_index: int = 0,
    *,
    confidence: float = 0.9,
    overrides: dict[KeypointName, Keypoint] | None = None,
) -> Pose:
    """17 keypoint'i olan bir Pose.

    Varsayilan olarak tum eklemler (0, 0) noktasinda. Bir metrigi test ederken
    yalnizca ilgilendigin eklemleri `overrides` ile yerine koy; geri kalani
    gurultu olarak durur ama metrigi etkilemez.
    """
    points = [make_keypoint(confidence=confidence) for _ in range(NUM_KEYPOINTS)]
    for name, keypoint in (overrides or {}).items():
        points[name.value] = keypoint
    return Pose(frame_index=frame_index, keypoints=tuple(points))


def make_sequence(
    n_frames: int = 10,
    *,
    fps: float = 120.0,
    confidence: float = 0.9,
) -> PoseSequence:
    """Bosluksuz, sabit guvenli bir poz dizisi."""
    poses = tuple(make_pose(i, confidence=confidence) for i in range(n_frames))
    return PoseSequence(fps=fps, poses=poses)


def make_instance(
    n_frames: int = 10,
    *,
    element: str = "jump",
    fps: float = 120.0,
) -> ElementInstance:
    poses = make_sequence(n_frames, fps=fps)
    return ElementInstance(
        element=element,
        frames=FrameRange(start=0, end=n_frames),
        poses=poses,
    )


@pytest.fixture
def sequence() -> PoseSequence:
    return make_sequence()


@pytest.fixture
def instance() -> ElementInstance:
    return make_instance()


# --------------------------------------------------------- ingest yardimcilari


def make_video_info(
    *,
    path: Path | None = None,
    width: int = 1280,
    height: int = 720,
    fps: float = 120.0,
    duration_s: float = 4.0,
    codec: str = "h264",
    is_vfr: bool = False,
) -> VideoInfo:
    """Saf kalite testleri icin ust veri. Diskte dosya olmasi gerekmez."""
    return VideoInfo(
        path=path or Path("data/raw/ornek.mp4"),
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        n_frames=int(duration_s * fps),
        codec=codec,
        is_vfr=is_vfr,
    )


def make_clip_stats(
    *,
    mean_brightness: float = 130.0,
    shake_px: float = 1.0,
    sampled_frames: int = 24,
) -> ClipStats:
    """Kabul edilebilir bir klibin olcumleri. Testler tek alani bozar."""
    return ClipStats(
        mean_brightness=mean_brightness,
        shake_px=shake_px,
        sampled_frames=sampled_frames,
    )


def write_test_video(
    path: Path,
    *,
    n_frames: int = 120,
    fps: float = 60.0,
    size: tuple[int, int] = (640, 480),
    brightness: int = 140,
    shake_px: float = 0.0,
    seed: int = 0,
) -> Path:
    """Kontrollu ozelliklerle sentetik bir mp4 uret.

    ffmpeg'in test kaynaklari yerine kareleri kendimiz ciziyoruz: parlaklik ve
    kamera kaymasi test icinde tam olarak bilinen degerler olsun istiyoruz.

    `shake_px` her karede uygulanan rastgele oteleme genligi. Kaydirma np.roll
    ile yapiliyor, yani icerik degismez, yalnizca yerinden oynar; bu da kamera
    titremesinin ta kendisidir.
    """
    width, height = size
    rng = np.random.default_rng(seed)

    # Sabit bir doku: duz zeminde faz korelasyonu kaymayi bulamaz.
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :] = brightness
    step_x, step_y = max(width // 16, 1), max(height // 12, 1)
    base[::step_y, :] = np.clip(brightness + 60, 0, 255)
    base[:, ::step_x] = np.clip(brightness - 60, 0, 255)
    cv2.circle(base, (width // 3, height // 2), min(width, height) // 8,
               (int(np.clip(brightness + 90, 0, 255)),) * 3, -1)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        msg = f"VideoWriter acilamadi: {path}"
        raise RuntimeError(msg)
    try:
        for _ in range(n_frames):
            if shake_px > 0:
                dx, dy = rng.normal(0.0, shake_px, size=2)
                frame = np.roll(base, (int(round(dy)), int(round(dx))), axis=(0, 1))
            else:
                frame = base
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture(scope="session")
def good_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Kabul edilmesi gereken klip: 2 sn, 60 fps, 640x480, aydinlik, sabit."""
    directory = tmp_path_factory.mktemp("videos")
    return write_test_video(directory / "good.mp4", n_frames=120, fps=60.0)


# ----------------------------------------------------------- pose yardimcilari


def make_bbox(
    cx: float = 640.0, cy: float = 360.0, width: float = 180.0, height: float = 400.0
) -> BBox:
    """Merkez ve boyuttan kutu. Testler kutuyu merkezinden konumlandirmak
    istiyor; xyxy elle yazmak okunmuyor."""
    return BBox(
        x1=cx - width / 2, y1=cy - height / 2, x2=cx + width / 2, y2=cy + height / 2
    )


def make_detection(
    bbox: BBox | None = None, *, confidence: float = 0.9, track_id: int | None = 1
) -> Detection:
    return Detection(bbox=bbox or make_bbox(), confidence=confidence, track_id=track_id)


def straight_track(
    n_frames: int,
    *,
    start: tuple[float, float] = (400.0, 360.0),
    end: tuple[float, float] | None = None,
    size: tuple[float, float] = (180.0, 400.0),
) -> list[BBox | None]:
    """Kare kare duzgun ilerleyen bir kutu dizisi.

    Gercek bir sporcunun kutusu kareler arasinda kucuk adimlarla kayar;
    kimlik degisimi testleri bu duzgunlugu bilerek bozar.
    """
    finish = end or start
    width, height = size
    boxes: list[BBox | None] = []
    for index in range(n_frames):
        ratio = index / (n_frames - 1) if n_frames > 1 else 0.0
        cx = start[0] + (finish[0] - start[0]) * ratio
        cy = start[1] + (finish[1] - start[1]) * ratio
        boxes.append(make_bbox(cx, cy, width, height))
    return boxes


def make_detection_sequence(
    tracks: dict[int, Sequence[BBox | None]],
    *,
    width: int = 1280,
    height: int = 720,
    confidence: float = 0.9,
) -> DetectionSequence:
    """Takip kimligi basina kare kare kutulardan bir tespit dizisi kur.

    `None` o karede o takibin gorunmedigi anlamina gelir. Tum listeler ayni
    uzunlukta olmali; kare indeksi ile liste indeksi birebir eslesiyor.
    """
    lengths = {len(boxes) for boxes in tracks.values()}
    if len(lengths) > 1:
        msg = f"tum takipler ayni uzunlukta olmali, gelen: {sorted(lengths)}"
        raise ValueError(msg)
    n_frames = lengths.pop() if lengths else 0

    frames: list[FrameDetections] = []
    for index in range(n_frames):
        detections = tuple(
            Detection(bbox=boxes[index], confidence=confidence, track_id=track_id)
            for track_id, boxes in tracks.items()
            if boxes[index] is not None
        )
        frames.append(FrameDetections(frame_index=index, detections=detections))
    return DetectionSequence(width=width, height=height, frames=tuple(frames))
