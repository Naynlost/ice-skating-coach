"""Kisi tespiti (YOLO11), takip (ByteTrack) ve hedef sporcu secimi.

Tipik akis:

    sequence = detect_persons(video_path)
    selection = select_athlete(sequence)
    if not selection.ok:
        # sporcu secilemedi; sebep selection.warnings icinde
        ...
    for warning in selection.warnings:
        # kimlik degisimi belirtileri; overlay ile gozle dogrula
        ...

Katmanlar bilerek ayri: `detect_persons()` modeli calistirir, geri kalan her
sey saf. Sporcu secimi ve kimlik analizi ultralytics olmadan da calisir ve
test edilir.

2D poz tahmini bu modulde degil; gorev 1.5'te eklenecek.
"""

from core.pose.athlete import (
    AthleteSelection,
    TrackIssue,
    TrackSummary,
    TrackWarning,
    analyze_track,
    select_athlete,
    summarize_tracks,
)
from core.pose.boxes import BBox, Detection, DetectionSequence, FrameDetections
from core.pose.config import DEFAULT_CONFIG_PATH, PoseConfig
from core.pose.detect import PERSON_CLASS_ID, detect_persons, to_frame_detections

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PERSON_CLASS_ID",
    "AthleteSelection",
    "BBox",
    "Detection",
    "DetectionSequence",
    "FrameDetections",
    "PoseConfig",
    "TrackIssue",
    "TrackSummary",
    "TrackWarning",
    "analyze_track",
    "detect_persons",
    "select_athlete",
    "summarize_tracks",
    "to_frame_detections",
]
