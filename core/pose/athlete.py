"""Hedef sporcuyu sec ve takibin guvenilirligini denetle.

Bu dosyanin tamami saf: model calistirmaz, video acmaz, dosya okumaz. Girdi
tespit dizisi, cikti secim ve uyarilar. Bu sayede secim mantigini ve kimlik
degisimi tespitini bir mp4 dosyasi olmadan sinayabiliyoruz.

pose-debug skill'inin 2. katmani burasi: "overlay'de kutunun klip boyunca ayni
kiside kaldigini dogrula". Buradaki uyarilar o gozle bakisi otomatiklestirir,
yerine gecmez.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from core.pose.boxes import BBox, Detection, DetectionSequence
from core.pose.config import PoseConfig
from core.schema import FrameRange

__all__ = [
    "AthleteSelection",
    "TrackIssue",
    "TrackSummary",
    "TrackWarning",
    "analyze_track",
    "select_athlete",
    "summarize_tracks",
]

_BASE_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class TrackIssue(StrEnum):
    """Takipte guven dusuren durumlar. Hicbiri hata degil, hepsi uyari."""

    NO_PERSON = "kisi_bulunamadi"
    NO_TRACK_ID = "takip_kimligi_yok"
    LOW_COVERAGE = "dusuk_kapsama"
    TRACKING_GAP = "takip_boslugu"
    POSITION_JUMP = "ani_konum_sicramasi"
    SCALE_JUMP = "ani_olcek_degisimi"
    CROWDED_SCENE = "kalabalik_sahne"
    AMBIGUOUS_CHOICE = "belirsiz_sporcu_secimi"


class TrackWarning(BaseModel):
    """Tek bir uyari: ne oldu, nerede, ne kadar.

    `frames` uyarinin gectigi kare araligi. Overlay'de bu araligi isaretleyip
    gozle dogrulayabilmek icin tasiyoruz.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue: TrackIssue
    message: str
    frames: FrameRange | None = None
    measured: float | None = None
    threshold: float | None = None


class TrackSummary(BaseModel):
    """Bir takip kimliginin klip genelindeki ozeti.

    `area_score` ve `centrality` 0-1 arasi normalize edilmis degerler;
    `score` bu ikisinin agirlikli toplami. Ham piksel degerleri yerine
    normalize deger tutuyoruz cunku "en buyuk" gorelidir: karedeki en buyuk
    kutuya gore buyuk demek.
    """

    model_config = _BASE_CONFIG

    track_id: int
    frame_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    mean_area_ratio: float = Field(ge=0.0)
    area_score: float = Field(ge=0.0, le=1.0)
    centrality: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0)
    span: FrameRange


class AthleteSelection(BaseModel):
    """Secilen sporcu ve secimin gerekcesi.

    `track_id` None ise sporcu secilememis demektir; `warnings` sebebi soyler.
    Boyle bir durumda tahmin uretmiyoruz: yanlis kisiyi olcmek, olcmemekten
    kotudur.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: int | None = None
    detections: dict[int, Detection] = Field(default_factory=dict)
    summary: TrackSummary | None = None
    candidates: tuple[TrackSummary, ...] = ()
    warnings: tuple[TrackWarning, ...] = ()

    @property
    def ok(self) -> bool:
        return self.track_id is not None

    @property
    def issues(self) -> tuple[TrackIssue, ...]:
        return tuple(warning.issue for warning in self.warnings)

    def bbox_at(self, frame: int) -> BBox | None:
        detection = self.detections.get(frame)
        return detection.bbox if detection else None


def _centrality(
    bbox: BBox, frame_center: tuple[float, float], width: int, height: int
) -> float:
    """Kutunun kare merkezine yakinligi, 1 = tam merkez, 0 = kose.

    Yatay ve dikey sapmalar kendi yari boyutlarina bolunerek olceklenir,
    boylece 16:9 bir karede yatay uzaklik haksiz agirlik kazanmaz.
    """
    cx, cy = frame_center
    dx = abs(bbox.cx - cx) / (width / 2)
    dy = abs(bbox.cy - cy) / (height / 2)
    distance = min(float((dx**2 + dy**2) ** 0.5) / (2**0.5), 1.0)
    return 1.0 - distance


def summarize_tracks(
    sequence: DetectionSequence, config: PoseConfig | None = None
) -> tuple[TrackSummary, ...]:
    """Her takip kimligi icin klip genelinde ozet cikar. Skora gore sirali.

    Guveni esigin altinda kalan tespitler hesaba katilmaz: dusuk guvenli bir
    kutu genelde yanlis yerde olur ve ortalamayi bozar.
    """
    cfg = config or PoseConfig()
    total_frames = len(sequence)
    if total_frames == 0:
        return ()

    frame_center = sequence.frame_center
    areas: dict[int, list[float]] = {}
    centralities: dict[int, list[float]] = {}
    frames_seen: dict[int, list[int]] = {}

    for frame in sequence.frames:
        for detection in frame.detections:
            if detection.track_id is None:
                continue
            if detection.confidence < cfg.min_detection_confidence:
                continue
            track_id = detection.track_id
            areas.setdefault(track_id, []).append(detection.bbox.area / sequence.frame_area)
            centralities.setdefault(track_id, []).append(
                _centrality(detection.bbox, frame_center, sequence.width, sequence.height)
            )
            frames_seen.setdefault(track_id, []).append(frame.frame_index)

    if not areas:
        return ()

    max_area = max(sum(values) / len(values) for values in areas.values())
    summaries: list[TrackSummary] = []

    for track_id, area_values in areas.items():
        mean_area = sum(area_values) / len(area_values)
        mean_centrality = sum(centralities[track_id]) / len(centralities[track_id])
        # Alan skoru en buyuk takibe gore normalize: "en buyuk kutu" gorelidir.
        area_score = mean_area / max_area if max_area > 0 else 0.0
        weight_sum = cfg.area_weight + cfg.centrality_weight
        score = (cfg.area_weight * area_score + cfg.centrality_weight * mean_centrality)
        seen = frames_seen[track_id]
        summaries.append(
            TrackSummary(
                track_id=track_id,
                frame_count=len(seen),
                coverage=len(seen) / total_frames,
                mean_area_ratio=mean_area,
                area_score=area_score,
                centrality=mean_centrality,
                score=score / weight_sum,
                span=FrameRange(start=min(seen), end=max(seen) + 1),
            )
        )

    return tuple(sorted(summaries, key=lambda s: s.score, reverse=True))


def select_athlete(
    sequence: DetectionSequence, config: PoseConfig | None = None
) -> AthleteSelection:
    """Klipteki hedef sporcuyu sec: en buyuk ve en merkezi takip.

    Once kapsama kapisi uygulanir. Pistten gecip giden bir patenci birkac
    karede cok buyuk gorunebilir; kapsama kapisi olmadan skoru gercek
    sporcuyu gecebilir.
    """
    cfg = config or PoseConfig()
    candidates = summarize_tracks(sequence, cfg)
    warnings: list[TrackWarning] = []

    if len(sequence) == 0 or not candidates:
        warnings.append(
            TrackWarning(
                issue=TrackIssue.NO_PERSON,
                message="klipte takip kimligi atanmis kisi bulunamadi",
            )
        )
        return AthleteSelection(warnings=tuple(warnings))

    eligible = [c for c in candidates if c.coverage >= cfg.min_coverage]
    if not eligible:
        # Kimse kapsama kapisini gecemedi: en cok gorunen takibi aliyoruz ama
        # bunun guvenilir olmadigini acikca soyluyoruz.
        best_coverage = max(candidates, key=lambda c: c.coverage)
        warnings.append(
            TrackWarning(
                issue=TrackIssue.LOW_COVERAGE,
                message=(
                    f"hicbir takip klibin {cfg.min_coverage:.0%} kadarini kapsamiyor; "
                    f"en iyisi {best_coverage.coverage:.0%}"
                ),
                measured=best_coverage.coverage,
                threshold=cfg.min_coverage,
            )
        )
        eligible = [best_coverage]

    chosen = max(eligible, key=lambda c: c.score)

    runners_up = [c for c in eligible if c.track_id != chosen.track_id]
    if runners_up:
        second = max(runners_up, key=lambda c: c.score)
        if chosen.score - second.score < 0.1:
            # Iki aday neredeyse esit: yanlis kisiyi olcuyor olabiliriz.
            warnings.append(
                TrackWarning(
                    issue=TrackIssue.AMBIGUOUS_CHOICE,
                    message=(
                        f"takip {chosen.track_id} ve {second.track_id} skorlari yakin "
                        f"({chosen.score:.2f} / {second.score:.2f}); overlay ile dogrula"
                    ),
                    measured=chosen.score - second.score,
                    threshold=0.1,
                )
            )

    detections = {
        frame.frame_index: detection
        for frame in sequence.frames
        if (detection := frame.by_track(chosen.track_id)) is not None
    }

    selection = AthleteSelection(
        track_id=chosen.track_id,
        detections=detections,
        summary=chosen,
        candidates=candidates,
        warnings=tuple(warnings),
    )
    return selection.model_copy(
        update={"warnings": (*warnings, *analyze_track(selection, sequence, cfg))}
    )


def _gap_ranges(present: list[int], span: FrameRange) -> list[FrameRange]:
    """Takibin gorunmedigi kare araliklari, secilen aralik icinde."""
    seen = set(present)
    gaps: list[FrameRange] = []
    start: int | None = None
    for frame in range(span.start, span.end):
        if frame in seen:
            if start is not None:
                gaps.append(FrameRange(start=start, end=frame))
                start = None
        elif start is None:
            start = frame
    if start is not None:
        gaps.append(FrameRange(start=start, end=span.end))
    return gaps


def analyze_track(
    selection: AthleteSelection,
    sequence: DetectionSequence,
    config: PoseConfig | None = None,
) -> tuple[TrackWarning, ...]:
    """Secilen takipte kimlik degisimi belirtilerini ara.

    Uc belirti ariyoruz ve ucu de ayni seyi gosteriyor: kutu artik ayni
    vucutta degil.

    1. Bosluk: takip birkac kare kayboluyor, sonra geri geliyor. Geri geldigi
       vucut ayni olmayabilir.
    2. Ani konum sicramasi: merkez bir karede kutu genisliginin buyuk bir
       kismi kadar kayiyor. Sporcu sicrarken bile bunu yapmaz.
    3. Ani olcek degisimi: kutu alani bir karede katlaniyor ya da yariya
       iniyor. Kutu baska bir cisme oturmus demektir.

    Ayrica kalabalik sahne uyarisi: sporcunun kutusu baska bir kutuyla
    ortusuyorsa, takip henuz atlamamis olsa bile atlamaya acik demektir.
    """
    cfg = config or PoseConfig()
    if selection.track_id is None or selection.summary is None:
        return ()

    warnings: list[TrackWarning] = []
    frames = sorted(selection.detections)
    span = selection.summary.span

    for gap in _gap_ranges(frames, span):
        if len(gap) > cfg.max_gap_frames:
            warnings.append(
                TrackWarning(
                    issue=TrackIssue.TRACKING_GAP,
                    message=(
                        f"takip {gap.start}-{gap.end - 1}. karelerde kayip "
                        f"({len(gap)} kare); geri donduğunde ayni kisi olmayabilir"
                    ),
                    frames=gap,
                    measured=float(len(gap)),
                    threshold=float(cfg.max_gap_frames),
                )
            )

    for previous_frame, current_frame in zip(frames, frames[1:], strict=False):
        if current_frame - previous_frame != 1:
            continue  # bosluk zaten ayrica raporlandi
        previous = selection.detections[previous_frame].bbox
        current = selection.detections[current_frame].bbox

        reference_width = max(previous.width, 1e-6)
        jump_ratio = previous.center_distance(current) / reference_width
        if jump_ratio > cfg.max_center_jump_ratio:
            warnings.append(
                TrackWarning(
                    issue=TrackIssue.POSITION_JUMP,
                    message=(
                        f"{current_frame}. karede kutu merkezi kendi genisliginin "
                        f"{jump_ratio:.1f} kati kadar kaydi; kimlik atlamis olabilir"
                    ),
                    frames=FrameRange(start=previous_frame, end=current_frame + 1),
                    measured=jump_ratio,
                    threshold=cfg.max_center_jump_ratio,
                )
            )

        if previous.area > 0:
            scale = current.area / previous.area
            ratio = max(scale, 1 / scale) if scale > 0 else float("inf")
            if ratio > cfg.max_scale_ratio:
                warnings.append(
                    TrackWarning(
                        issue=TrackIssue.SCALE_JUMP,
                        message=(
                            f"{current_frame}. karede kutu alani {ratio:.1f} kat degisti; "
                            "kutu baska bir cisme oturmus olabilir"
                        ),
                        frames=FrameRange(start=previous_frame, end=current_frame + 1),
                        measured=ratio,
                        threshold=cfg.max_scale_ratio,
                    )
                )

    crowded = _crowded_ranges(selection, sequence, cfg)
    warnings.extend(crowded)
    return tuple(warnings)


def _crowded_ranges(
    selection: AthleteSelection, sequence: DetectionSequence, config: PoseConfig
) -> list[TrackWarning]:
    """Sporcunun kutusunun baska kutularla ortustugu araliklar.

    Kare kare uyari uretmek yerine ardisik kareleri tek araliga topluyoruz;
    yoksa iki patencinin yan yana gectigi bir klipte yuzlerce uyari cikar.
    """
    if selection.track_id is None:
        return []

    flagged: list[int] = []
    peak = 0.0
    for frame_index, detection in selection.detections.items():
        frame = sequence[frame_index]
        for other in frame.others(selection.track_id):
            overlap = detection.bbox.iou(other.bbox)
            if overlap > config.crowding_iou:
                flagged.append(frame_index)
                peak = max(peak, overlap)
                break

    if not flagged:
        return []

    warnings: list[TrackWarning] = []
    flagged.sort()
    start = previous = flagged[0]
    for frame_index in flagged[1:]:
        if frame_index != previous + 1:
            warnings.append(_crowding_warning(start, previous, peak, config))
            start = frame_index
        previous = frame_index
    warnings.append(_crowding_warning(start, previous, peak, config))
    return warnings


def _crowding_warning(start: int, end: int, peak: float, config: PoseConfig) -> TrackWarning:
    return TrackWarning(
        issue=TrackIssue.CROWDED_SCENE,
        message=(
            f"{start}-{end}. karelerde sporcunun kutusu baska bir kisiyle ortusuyor "
            f"(en yuksek ortusme {peak:.2f}); takip karisabilir"
        ),
        frames=FrameRange(start=start, end=end + 1),
        measured=peak,
        threshold=config.crowding_iou,
    )
