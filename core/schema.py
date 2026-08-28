"""Tum modullerin paylastigi veri tipleri.

Koordinat sistemi: piksel, sol ust kose (0, 0), y ASAGI dogru artar.
Kare indeksleri 0 tabanli. Zaman saniye cinsinden float.

Tipler dondurulmus (frozen). Bir asamayi degistirmek yeni bir nesne uretir;
metrik fonksiyonlari girdilerini yerinde degistiremez, bu saflik kuralinin
tip seviyesindeki karsiligi.

NaN ve sonsuz degerler her alanda reddedilir. Sessizce yayilan bir NaN,
uydurma sayidan daha kotudur: rapora "olculemedi" olarak degil, gecerli bir
sonuc gibi girer.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import IntEnum, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "NUM_KEYPOINTS",
    "ElementInstance",
    "FrameRange",
    "Keypoint",
    "KeypointName",
    "MetricResult",
    "Pose",
    "PoseSequence",
    "SkillLevel",
    "Unmeasurable",
]

# Bu esigin altindaki keypoint ile hesaplanan metrik guvenilmez sayilir.
# Metrik fonksiyonlari bunu parametre olarak alir; sabit yalnizca varsayilandir.
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

_BASE_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class KeypointName(IntEnum):
    """COCO-17 keypoint sirasi. Deger, poz modelinin dondurdugu dizi indeksidir.

    Sag/sol, sporcunun kendi sagi ve solu; goruntudeki degil. Yan aci
    cekimlerde poz modelleri bu ikisini sik karistirir, bkz. pose-debug
    skill 4. katman.
    """

    NOSE = 0
    EYE_L = 1
    EYE_R = 2
    EAR_L = 3
    EAR_R = 4
    SHOULDER_L = 5
    SHOULDER_R = 6
    ELBOW_L = 7
    ELBOW_R = 8
    WRIST_L = 9
    WRIST_R = 10
    HIP_L = 11
    HIP_R = 12
    KNEE_L = 13
    KNEE_R = 14
    ANKLE_L = 15
    ANKLE_R = 16


NUM_KEYPOINTS = len(KeypointName)


class SkillLevel(StrEnum):
    """Sporcu seviyesi. rules.yaml icindeki levels anahtarlariyla birebir ayni."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Unmeasurable(StrEnum):
    """MetricResult.reason icin standart sebepler.

    Serbest metin yerine sabit sebep kullanmak "olculemedi" satirlarini
    gruplanabilir yapar: 30 videonun kacinda ayni sebeple olcum dustu?
    """

    LOW_KEYPOINT_CONFIDENCE = "dusuk_keypoint_guveni"
    CLIP_TOO_SHORT = "klip_cok_kisa"
    PHASE_NOT_FOUND = "faz_bulunamadi"
    MISSING_KEYPOINT = "keypoint_eksik"
    ICE_LINE_NOT_FOUND = "buz_cizgisi_bulunamadi"


class Keypoint(BaseModel):
    """Tek bir eklemin bir karedeki konumu.

    x, y piksel cinsinden; y asagi dogru artar. Kare disina tasan tahminler
    olabilecegi icin negatif koordinat serbest.

    confidence 0.0, poz modelinin o eklemi hic bulamadigi anlamina gelir.
    """

    model_config = _BASE_CONFIG

    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)

    def is_reliable(self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
        return self.confidence >= threshold


class Pose(BaseModel):
    """Tek karedeki 17 keypoint.

    Erisim iki sekilde: isimle (pose[KeypointName.HIP_R]) ya da sik kullanilan
    eklemler icin kisayol ozelligiyle (pose.hip_r). Metrik kodu okunur kalsin
    diye kisayollar var; ham dizi indeksiyle erisme.
    """

    model_config = _BASE_CONFIG

    frame_index: int = Field(ge=0)
    keypoints: tuple[Keypoint, ...]

    @model_validator(mode="after")
    def _check_keypoint_count(self) -> Self:
        if len(self.keypoints) != NUM_KEYPOINTS:
            msg = f"{NUM_KEYPOINTS} keypoint bekleniyor, {len(self.keypoints)} geldi"
            raise ValueError(msg)
        return self

    def __getitem__(self, name: KeypointName) -> Keypoint:
        return self.keypoints[name.value]

    def min_confidence(self, *names: KeypointName) -> float:
        """Verilen eklemlerin en dusuk guveni. Metrik fonksiyonlarinin giris kapisi."""
        if not names:
            msg = "en az bir keypoint adi gerekli"
            raise ValueError(msg)
        return min(self[name].confidence for name in names)

    def all_reliable(
        self, *names: KeypointName, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> bool:
        return self.min_confidence(*names) >= threshold

    # Kisayollar. Bacak ve govde metrikleri bunlarin uzerinden yazilir.
    @property
    def nose(self) -> Keypoint:
        return self[KeypointName.NOSE]

    @property
    def shoulder_l(self) -> Keypoint:
        return self[KeypointName.SHOULDER_L]

    @property
    def shoulder_r(self) -> Keypoint:
        return self[KeypointName.SHOULDER_R]

    @property
    def elbow_l(self) -> Keypoint:
        return self[KeypointName.ELBOW_L]

    @property
    def elbow_r(self) -> Keypoint:
        return self[KeypointName.ELBOW_R]

    @property
    def wrist_l(self) -> Keypoint:
        return self[KeypointName.WRIST_L]

    @property
    def wrist_r(self) -> Keypoint:
        return self[KeypointName.WRIST_R]

    @property
    def hip_l(self) -> Keypoint:
        return self[KeypointName.HIP_L]

    @property
    def hip_r(self) -> Keypoint:
        return self[KeypointName.HIP_R]

    @property
    def knee_l(self) -> Keypoint:
        return self[KeypointName.KNEE_L]

    @property
    def knee_r(self) -> Keypoint:
        return self[KeypointName.KNEE_R]

    @property
    def ankle_l(self) -> Keypoint:
        return self[KeypointName.ANKLE_L]

    @property
    def ankle_r(self) -> Keypoint:
        return self[KeypointName.ANKLE_R]


class FrameRange(BaseModel):
    """Yari acik kare araligi: [start, end). Python dilimlemesiyle ayni davranir.

    Bos aralik (start == end) gecerlidir ve "bu faz bu klipte yok" demektir.
    """

    model_config = _BASE_CONFIG

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end < self.start:
            msg = f"end ({self.end}) start'tan ({self.start}) kucuk olamaz"
            raise ValueError(msg)
        return self

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        return iter(range(self.start, self.end))

    def __contains__(self, frame: int) -> bool:
        return self.start <= frame < self.end

    @property
    def is_empty(self) -> bool:
        return self.end == self.start

    def duration_s(self, fps: float) -> float:
        """Aralik suresi, saniye."""
        return len(self) / fps


class PoseSequence(BaseModel):
    """Bir klibin kare kare poz verisi.

    Dizi bosluksuz: poses[i] her zaman i numarali karedir. Poz modelinin bir
    eklemi bulamadigi kareler atlanmaz, o keypoint confidence=0 ile yerinde
    durur. Atlansaydi kare indeksi ile liste indeksi ayrisir ve faz sinirlari
    sessizce kayardi.
    """

    model_config = _BASE_CONFIG

    fps: float = Field(gt=0.0)
    poses: tuple[Pose, ...]

    @model_validator(mode="after")
    def _check_contiguous(self) -> Self:
        for expected, pose in enumerate(self.poses):
            if pose.frame_index != expected:
                msg = (
                    "kare indeksleri 0'dan itibaren ardisik olmali; "
                    f"{expected}. sirada {pose.frame_index} bulundu"
                )
                raise ValueError(msg)
        return self

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, frame: int) -> Pose:
        return self.poses[frame]

    def __iter__(self) -> Iterator[Pose]:  # type: ignore[override]
        return iter(self.poses)

    @property
    def duration_s(self) -> float:
        return len(self.poses) / self.fps

    def time_of(self, frame: int) -> float:
        """Kare indeksinin klip basindan gectigi sure, saniye."""
        return frame / self.fps

    def frames_for(self, seconds: float) -> int:
        """Saniyeyi kare sayisina cevir. Asagi yuvarlar."""
        return int(seconds * self.fps)

    def window(self, span: FrameRange) -> tuple[Pose, ...]:
        """Aralik icindeki pozlar. Aralik klip disina tasarsa kirpilir."""
        return self.poses[span.start : span.end]


class ElementInstance(BaseModel):
    """Klipte bir elemanin gectigi aralik ve o araligin faz bolunmesi.

    phases bos baslar; segment() ciktisi with_phases() ile eklenir. Faz
    anahtarlari rules.yaml icindeki phase alanlariyla birebir ayni olmali,
    yoksa kural motoru o metrigi sessizce atlar.
    """

    model_config = _BASE_CONFIG

    element: str
    frames: FrameRange
    poses: PoseSequence
    level: SkillLevel = SkillLevel.BEGINNER
    phases: dict[str, FrameRange] = Field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.poses.fps

    def with_phases(self, phases: dict[str, FrameRange]) -> ElementInstance:
        """Faz bolunmesi eklenmis yeni bir kopya dondur. Nesneler dondurulmus."""
        return self.model_copy(update={"phases": dict(phases)})

    def phase(self, name: str) -> FrameRange | None:
        """Faz araligi, tanimli degilse None.

        None gelirse metrik fonksiyonu
        MetricResult.unmeasured(Unmeasurable.PHASE_NOT_FOUND) dondurur.
        """
        return self.phases.get(name)


class MetricResult(BaseModel):
    """Tek bir metrigin sonucu.

    Olculemeyen metrik value=None ve bir reason ile doner. Uydurma sayi
    uretmek bu projedeki en kotu sonuctur: sporcuya yanlis teknik tavsiyesi
    sakatlik riski demek.

    Dogrulayici iki yonlu bir kurali zorunlu tutuyor: value None ise reason
    dolu olmali, value varsa reason bos olmali. Boylece ne "sebepsiz bos
    sonuc" ne de "sebep yazilmis ama yine de sayi verilmis" gibi belirsiz bir
    sonuc uretilebilir.
    """

    model_config = _BASE_CONFIG

    value: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: Unmeasurable | None = None

    @model_validator(mode="after")
    def _check_value_reason_pair(self) -> Self:
        if self.value is None and self.reason is None:
            msg = "olculemeyen metrik icin reason zorunlu"
            raise ValueError(msg)
        if self.value is not None and self.reason is not None:
            msg = (
                f"olculmus metrik reason tasiyamaz "
                f"(value={self.value}, reason={self.reason})"
            )
            raise ValueError(msg)
        return self

    @property
    def ok(self) -> bool:
        return self.value is not None

    @classmethod
    def unmeasured(cls, reason: Unmeasurable, confidence: float = 0.0) -> MetricResult:
        """Olculemedi sonucu. Metrik fonksiyonlarinin erken cikis yolu."""
        return cls(value=None, confidence=confidence, reason=reason)

    @classmethod
    def measured(cls, value: float, confidence: float) -> MetricResult:
        return cls(value=value, confidence=confidence, reason=None)
