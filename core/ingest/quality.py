"""Kabul kapisi: klibi olc, olcumlere gore kabul veya ret kararini ver.

Iki asama bilerek ayrilmis:

`measure_clip()` dosyayi acar ve sayilari cikarir (I/O).
`evaluate_quality()` yalnizca sayilara bakip karar verir (saf).

Ayrimin sebebi test edilebilirlik: karar mantigini bir mp4 dosyasi olmadan,
elle kurulmus olcumlerle sinayabiliyoruz. Kabul kapisinin kendisi de bir
olcumdur; "kotu girdiden iyi metrik cikmaz" kuralinin kodda karsiligi.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.ingest.config import IngestConfig
from core.ingest.probe import VideoInfo, VideoUnreadableError, probe

__all__ = [
    "ClipStats",
    "QualityIssue",
    "QualityReport",
    "RejectionReason",
    "WarningReason",
    "check_quality",
    "evaluate_quality",
    "measure_clip",
]

# Titreklik ve parlaklik olcumu icin kareler bu genislige kucultulur.
# Amac hiz degil gurultu azaltma: tam cozunurlukte sensor gurultusu global
# kayma tahminini bozar.
_ANALYSIS_WIDTH = 320


class RejectionReason(StrEnum):
    """Videonun reddedilme sebebi. Kullaniciya bu sebep gosterilir."""

    TOO_SHORT = "cok_kisa"
    TOO_LONG = "cok_uzun"
    RESOLUTION_TOO_LOW = "cozunurluk_dusuk"
    TOO_DARK = "cok_karanlik"
    TOO_SHAKY = "asiri_titrek"
    FPS_TOO_LOW = "kare_hizi_dusuk"
    UNREADABLE = "okunamadi"


class WarningReason(StrEnum):
    """Reddi gerektirmeyen ama olcum dogrulugunu dusuren durumlar."""

    LOW_FPS_PRECISION = "kare_hizi_hassasiyeti_dusuk"
    VARIABLE_FRAME_RATE = "degisken_kare_hizi"
    MILD_SHAKE = "hafif_titreme"


class QualityIssue(BaseModel):
    """Tek bir bulgu: ne, ne olcduk, esik neydi.

    Olculen deger ve esigi birlikte tasiyoruz ki kullaniciya "videonuz kotu"
    yerine "18 saniye, ust sinir 30 saniye" diyebilelim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: RejectionReason | WarningReason
    message: str
    measured: float | None = None
    threshold: float | None = None


class ClipStats(BaseModel):
    """Kareler uzerinden olculen degerler. Yorum yok, sadece sayi."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Orneklenen karelerin ortalama parlakligi, 0-255.
    mean_brightness: float = Field(ge=0.0, le=255.0)
    # Ardisik kare ciftlerindeki global kaymanin standart sapmasi, piksel.
    # Sabit tripodda 1-2 piksel; elde cekimde hizla buyur.
    shake_px: float = Field(ge=0.0)
    sampled_frames: int = Field(ge=0)


class QualityReport(BaseModel):
    """Kabul karari ve gerekcesi."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    # Ust veri okunamadiysa None. Uydurma deger koymuyoruz.
    info: VideoInfo | None = None
    stats: ClipStats | None = None
    rejections: tuple[QualityIssue, ...] = ()
    warnings: tuple[QualityIssue, ...] = ()

    @property
    def reasons(self) -> tuple[RejectionReason | WarningReason, ...]:
        return tuple(issue.reason for issue in self.rejections)

    def summary(self) -> str:
        if self.accepted:
            head = "kabul"
            if self.warnings:
                notes = ", ".join(issue.message for issue in self.warnings)
                return f"{head} (uyari: {notes})"
            return head
        notes = ", ".join(issue.message for issue in self.rejections)
        return f"ret: {notes}"


def _global_shift(previous: np.ndarray, current: np.ndarray) -> float:
    """Iki kare arasindaki global kaymanin buyuklugu, piksel.

    Faz korelasyonu goruntunun tamamina bakar, bu yuzden karedeki tek bir
    hareketli nesne (sporcu) sonucu az etkiler; arka plan baskindir. Yani bu
    olcum sporcunun hareketini degil kameranin hareketini yakalar.
    """
    window = cv2.createHanningWindow(
        (previous.shape[1], previous.shape[0]), cv2.CV_32F
    )
    (dx, dy), _response = cv2.phaseCorrelate(previous, current, window)
    return float(np.hypot(dx, dy))


def _to_analysis_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width > _ANALYSIS_WIDTH:
        scale = _ANALYSIS_WIDTH / width
        gray = cv2.resize(
            gray, (_ANALYSIS_WIDTH, max(int(height * scale), 1)), interpolation=cv2.INTER_AREA
        )
    return gray.astype(np.float32)


def _sample_indices(n_frames: int, wanted: int) -> list[int]:
    """Klip boyunca esit araliklarla ornek noktalari.

    Her nokta icin (i, i+1) cifti okunacagi icin son kare disarida birakilir.
    """
    usable = max(n_frames - 1, 1)
    count = min(wanted, usable)
    if count <= 1:
        return [0]
    step = usable / count
    return sorted({min(int(k * step), usable - 1) for k in range(count)})


def measure_clip(path: Path, config: IngestConfig | None = None) -> ClipStats:
    """Klipten parlaklik ve titreklik olc.

    Dosyayi acar; saf degil. Kareler sirayla okunur ve yalnizca ornek
    noktalarinda cozulur (grab ucuz, retrieve pahali). Rastgele erisim yerine
    sirali okuma tercih edildi cunku bazi codec'lerde kareye atlama sessizce
    yanlis kareyi verir.
    """
    cfg = config or IngestConfig()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        msg = f"video acilamadi: {path}"
        raise VideoUnreadableError(msg)

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = cfg.quality_sample_frames * 2
        targets = set(_sample_indices(total, cfg.quality_sample_frames))

        brightness: list[float] = []
        shifts: list[float] = []
        index = 0
        pending: np.ndarray | None = None

        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            if index in targets or pending is not None:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                analysis = _to_analysis_frame(frame)
                if pending is None:
                    brightness.append(float(analysis.mean()))
                    pending = analysis
                else:
                    shifts.append(_global_shift(pending, analysis))
                    pending = None
            index += 1
    finally:
        capture.release()

    if not brightness:
        msg = f"videodan hic kare okunamadi: {path}"
        raise VideoUnreadableError(msg)

    return ClipStats(
        mean_brightness=float(np.mean(brightness)),
        shake_px=float(np.std(shifts)) if shifts else 0.0,
        sampled_frames=len(brightness),
    )


def evaluate_quality(
    info: VideoInfo, stats: ClipStats, config: IngestConfig | None = None
) -> QualityReport:
    """Olcumlere bakip kabul karari ver. Saf: dosya okumaz, aga cikmaz."""
    cfg = config or IngestConfig()
    rejections: list[QualityIssue] = []
    warnings: list[QualityIssue] = []

    if info.duration_s < cfg.min_duration_s:
        rejections.append(
            QualityIssue(
                reason=RejectionReason.TOO_SHORT,
                message=(
                    f"video {info.duration_s:.1f} sn, "
                    f"en az {cfg.min_duration_s:.1f} sn gerekli"
                ),
                measured=info.duration_s,
                threshold=cfg.min_duration_s,
            )
        )
    elif info.duration_s > cfg.max_duration_s:
        rejections.append(
            QualityIssue(
                reason=RejectionReason.TOO_LONG,
                message=(
                    f"video {info.duration_s:.1f} sn, en fazla {cfg.max_duration_s:.1f} sn "
                    "olmali; klibi tek sicramaya kirp"
                ),
                measured=info.duration_s,
                threshold=cfg.max_duration_s,
            )
        )

    if info.width < cfg.min_width or info.height < cfg.min_height:
        rejections.append(
            QualityIssue(
                reason=RejectionReason.RESOLUTION_TOO_LOW,
                message=(
                    f"cozunurluk {info.width}x{info.height}, en az "
                    f"{cfg.min_width}x{cfg.min_height} gerekli"
                ),
                measured=float(min(info.width, info.height)),
                threshold=float(min(cfg.min_width, cfg.min_height)),
            )
        )

    if info.fps < cfg.min_fps:
        rejections.append(
            QualityIssue(
                reason=RejectionReason.FPS_TOO_LOW,
                message=(
                    f"{info.fps:.0f} fps ile kalkis ve inis karesi ayirt edilemiyor; "
                    f"en az {cfg.min_fps:.0f} fps gerekli"
                ),
                measured=info.fps,
                threshold=cfg.min_fps,
            )
        )
    elif info.fps < cfg.warn_fps_below:
        # Airtime'in karesi yuksekligi verdigi icin dusuk fps hatayi buyutur.
        warnings.append(
            QualityIssue(
                reason=WarningReason.LOW_FPS_PRECISION,
                message=(
                    f"{info.fps:.0f} fps: tek karelik hata sicrama yuksekliginde "
                    f"belirgin sapma yaratir, {cfg.warn_fps_below:.0f} fps ve uzeri onerilir"
                ),
                measured=info.fps,
                threshold=cfg.warn_fps_below,
            )
        )

    if stats.mean_brightness < cfg.min_brightness:
        rejections.append(
            QualityIssue(
                reason=RejectionReason.TOO_DARK,
                message=(
                    f"ortalama parlaklik {stats.mean_brightness:.0f}, "
                    f"en az {cfg.min_brightness:.0f} gerekli"
                ),
                measured=stats.mean_brightness,
                threshold=cfg.min_brightness,
            )
        )

    if stats.shake_px > cfg.max_shake_px:
        rejections.append(
            QualityIssue(
                reason=RejectionReason.TOO_SHAKY,
                message=(
                    f"kamera hareketi {stats.shake_px:.1f} piksel, ust sinir "
                    f"{cfg.max_shake_px:.1f}; sabit tripod gerekli"
                ),
                measured=stats.shake_px,
                threshold=cfg.max_shake_px,
            )
        )
    elif stats.shake_px > cfg.max_shake_px / 2:
        warnings.append(
            QualityIssue(
                reason=WarningReason.MILD_SHAKE,
                message=f"hafif kamera hareketi ({stats.shake_px:.1f} piksel)",
                measured=stats.shake_px,
                threshold=cfg.max_shake_px,
            )
        )

    if info.is_vfr:
        warnings.append(
            QualityIssue(
                reason=WarningReason.VARIABLE_FRAME_RATE,
                message="degisken kare hizi; normalizasyonda sabit kare hizina cevrilecek",
            )
        )

    return QualityReport(
        accepted=not rejections,
        info=info,
        stats=stats,
        rejections=tuple(rejections),
        warnings=tuple(warnings),
    )


def check_quality(path: Path, config: IngestConfig | None = None) -> QualityReport:
    """Ust veriyi oku, klibi olc, karari dondur.

    Dosya hic okunamiyorsa istisna firlatmak yerine UNREADABLE gerekcesiyle
    reddedilmis bir rapor doner: cagiran taraf 30 videoyu dongude islerken
    tek bozuk dosya yuzunden durmasin.
    """
    cfg = config or IngestConfig()
    try:
        info = probe(path)
    except VideoUnreadableError as exc:
        return QualityReport(
            accepted=False,
            rejections=(
                QualityIssue(reason=RejectionReason.UNREADABLE, message=str(exc)),
            ),
        )

    try:
        stats = measure_clip(path, cfg)
    except VideoUnreadableError as exc:
        return QualityReport(
            accepted=False,
            info=info,
            rejections=(
                QualityIssue(reason=RejectionReason.UNREADABLE, message=str(exc)),
            ),
        )

    return evaluate_quality(info, stats, cfg)
