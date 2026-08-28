"""Video on isleme: ust veri okuma, kabul kapisi, normalizasyon, kare cikarma.

Tipik akis:

    report = check_quality(path)
    if not report.accepted:
        # sebebi kullaniciya soyle, isleme devam etme
        ...
    result = normalize(path, output=processed_path, info=report.info)
    for frame in iter_frames(processed_path):
        ...

Kabul kapisi bilerek islemenin onunde: kotu girdiden iyi metrik cikmaz ve bunu
sonraki katmanlarda duzeltemezsin. Reddedilen video icin sebep dondururuz,
tahmin uretmeyiz.
"""

from core.ingest.config import DEFAULT_CONFIG_PATH, IngestConfig
from core.ingest.frames import Frame, count_frames, iter_frames, read_frame
from core.ingest.normalize import FFmpegError, NormalizeResult, normalize
from core.ingest.probe import (
    FFprobeMissingError,
    VideoInfo,
    VideoUnreadableError,
    probe,
)
from core.ingest.quality import (
    ClipStats,
    QualityIssue,
    QualityReport,
    RejectionReason,
    WarningReason,
    check_quality,
    evaluate_quality,
    measure_clip,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ClipStats",
    "FFmpegError",
    "FFprobeMissingError",
    "Frame",
    "IngestConfig",
    "NormalizeResult",
    "QualityIssue",
    "QualityReport",
    "RejectionReason",
    "VideoInfo",
    "VideoUnreadableError",
    "WarningReason",
    "check_quality",
    "count_frames",
    "evaluate_quality",
    "iter_frames",
    "measure_clip",
    "normalize",
    "probe",
    "read_frame",
]
