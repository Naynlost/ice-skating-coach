"""core/schema.py tipleri.

Her tipin bir ornegi var (gorev 1.2 kriteri) ve tiplerin sessiz hataya izin
vermedigini gosteren dogrulama testleri var. Asil deger ikincisinde: sema
gecersiz veriyi kabul ederse hata metrik katmaninda degil, raporda ortaya cikar.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from conftest import make_keypoint, make_pose, make_sequence
from core.schema import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    NUM_KEYPOINTS,
    ElementInstance,
    FrameRange,
    Keypoint,
    KeypointName,
    MetricResult,
    Pose,
    PoseSequence,
    SkillLevel,
    Unmeasurable,
)

# --------------------------------------------------------------- Keypoint


def test_keypoint_instance() -> None:
    kp = Keypoint(x=120.5, y=340.0, confidence=0.87)
    assert (kp.x, kp.y, kp.confidence) == (120.5, 340.0, 0.87)


def test_keypoint_reliability_uses_default_threshold() -> None:
    assert make_keypoint(confidence=0.31).is_reliable()
    assert not make_keypoint(confidence=0.29).is_reliable()
    assert make_keypoint(confidence=0.29).is_reliable(threshold=0.2)


def test_keypoint_may_be_outside_the_frame() -> None:
    # Poz modeli kare disina tasan tahmin uretebilir; kisitlamiyoruz.
    assert Keypoint(x=-12.0, y=-3.0, confidence=0.4).x == -12.0


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_keypoint_confidence_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        Keypoint(x=0.0, y=0.0, confidence=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_keypoint_rejects_nan_and_inf(bad: float) -> None:
    # Sessizce yayilan bir NaN rapora gecerli sonuc gibi girer.
    with pytest.raises(ValidationError):
        Keypoint(x=bad, y=0.0, confidence=0.5)


def test_model_is_frozen() -> None:
    kp = make_keypoint()
    with pytest.raises(ValidationError):
        kp.x = 5.0


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Keypoint(x=0.0, y=0.0, confidence=0.5, z=1.0)


# ------------------------------------------------------------------- Pose


def test_pose_instance_has_seventeen_keypoints() -> None:
    pose = make_pose(frame_index=3)
    assert pose.frame_index == 3
    assert len(pose.keypoints) == NUM_KEYPOINTS == 17


def test_pose_rejects_wrong_keypoint_count() -> None:
    with pytest.raises(ValidationError, match="17 keypoint bekleniyor"):
        Pose(frame_index=0, keypoints=tuple(make_keypoint() for _ in range(16)))


def test_pose_named_access_matches_index_access() -> None:
    hip = make_keypoint(x=200.0, y=400.0, confidence=0.8)
    pose = make_pose(overrides={KeypointName.HIP_R: hip})
    assert pose.hip_r is hip
    assert pose[KeypointName.HIP_R] is hip
    assert pose.keypoints[KeypointName.HIP_R.value] is hip


def test_pose_min_confidence_picks_the_weakest_joint() -> None:
    pose = make_pose(
        confidence=0.9,
        overrides={KeypointName.ANKLE_R: make_keypoint(confidence=0.21)},
    )
    assert pose.min_confidence(KeypointName.HIP_R, KeypointName.ANKLE_R) == 0.21
    assert not pose.all_reliable(KeypointName.HIP_R, KeypointName.ANKLE_R)
    assert pose.all_reliable(KeypointName.HIP_R, KeypointName.KNEE_R)


def test_pose_min_confidence_needs_at_least_one_joint() -> None:
    with pytest.raises(ValueError, match="en az bir keypoint"):
        make_pose().min_confidence()


# ------------------------------------------------------------- FrameRange


def test_frame_range_instance_is_half_open() -> None:
    span = FrameRange(start=10, end=14)
    assert len(span) == 4
    assert list(span) == [10, 11, 12, 13]
    assert 13 in span
    assert 14 not in span


def test_frame_range_empty_means_phase_absent() -> None:
    span = FrameRange(start=7, end=7)
    assert span.is_empty
    assert len(span) == 0
    assert list(span) == []


def test_frame_range_duration_uses_fps() -> None:
    assert FrameRange(start=0, end=60).duration_s(120.0) == 0.5


def test_frame_range_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError, match="kucuk olamaz"):
        FrameRange(start=9, end=4)


def test_frame_range_rejects_negative_start() -> None:
    with pytest.raises(ValidationError):
        FrameRange(start=-1, end=4)


# ----------------------------------------------------------- PoseSequence


def test_pose_sequence_instance() -> None:
    seq = make_sequence(n_frames=24, fps=120.0)
    assert len(seq) == 24
    assert seq[5].frame_index == 5
    assert seq.duration_s == pytest.approx(0.2)


def test_pose_sequence_index_equals_frame_index() -> None:
    seq = make_sequence(n_frames=5)
    assert [pose.frame_index for pose in seq] == [0, 1, 2, 3, 4]


def test_pose_sequence_rejects_gaps() -> None:
    # Kare atlanirsa liste indeksi ile kare indeksi ayrisir, faz sinirlari kayar.
    poses = (make_pose(0), make_pose(1), make_pose(3))
    with pytest.raises(ValidationError, match="ardisik olmali"):
        PoseSequence(fps=120.0, poses=poses)


def test_pose_sequence_rejects_non_positive_fps() -> None:
    with pytest.raises(ValidationError):
        PoseSequence(fps=0.0, poses=())


def test_pose_sequence_time_and_frame_conversion() -> None:
    seq = make_sequence(n_frames=30, fps=120.0)
    assert seq.time_of(60) == 0.5
    assert seq.frames_for(0.1) == 12


def test_pose_sequence_window_clips_to_available_frames() -> None:
    seq = make_sequence(n_frames=10)
    assert len(seq.window(FrameRange(start=2, end=5))) == 3
    assert len(seq.window(FrameRange(start=8, end=40))) == 2


# -------------------------------------------------------- ElementInstance


def test_element_instance_starts_without_phases() -> None:
    seq = make_sequence(n_frames=40, fps=60.0)
    inst = ElementInstance(
        element="jump",
        frames=FrameRange(start=0, end=40),
        poses=seq,
        level=SkillLevel.INTERMEDIATE,
    )
    assert inst.element == "jump"
    assert inst.level is SkillLevel.INTERMEDIATE
    assert inst.fps == 60.0
    assert inst.phases == {}
    assert inst.phase("kalkis") is None


def test_with_phases_returns_a_new_instance(instance: ElementInstance) -> None:
    phases = {"kalkis": FrameRange(start=2, end=5)}
    updated = instance.with_phases(phases)
    assert updated is not instance
    assert instance.phases == {}
    assert updated.phase("kalkis") == FrameRange(start=2, end=5)


def test_with_phases_copies_the_mapping(instance: ElementInstance) -> None:
    phases = {"kalkis": FrameRange(start=2, end=5)}
    updated = instance.with_phases(phases)
    phases["kalkis"] = FrameRange(start=99, end=100)
    assert updated.phase("kalkis") == FrameRange(start=2, end=5)


def test_skill_level_values_match_rules_yaml_keys() -> None:
    assert [level.value for level in SkillLevel] == [
        "beginner",
        "intermediate",
        "advanced",
    ]


# ----------------------------------------------------------- MetricResult


def test_metric_result_measured() -> None:
    result = MetricResult.measured(value=138.4, confidence=0.72)
    assert result.ok
    assert result.value == 138.4
    assert result.reason is None


def test_metric_result_unmeasured() -> None:
    result = MetricResult.unmeasured(Unmeasurable.LOW_KEYPOINT_CONFIDENCE, confidence=0.18)
    assert not result.ok
    assert result.value is None
    assert result.reason is Unmeasurable.LOW_KEYPOINT_CONFIDENCE


def test_metric_result_without_value_requires_a_reason() -> None:
    # Sebepsiz bos sonuc raporda "olculemedi" satirini aciklamasiz birakir.
    with pytest.raises(ValidationError, match="reason zorunlu"):
        MetricResult(value=None, confidence=0.0)


def test_metric_result_with_value_cannot_carry_a_reason() -> None:
    # Hem sayi hem sebep tasiyan sonuc, kural motorunun hangisine
    # bakacagini belirsiz birakir.
    with pytest.raises(ValidationError, match="reason tasiyamaz"):
        MetricResult(
            value=120.0,
            confidence=0.5,
            reason=Unmeasurable.LOW_KEYPOINT_CONFIDENCE,
        )


def test_metric_result_rejects_nan_value() -> None:
    with pytest.raises(ValidationError):
        MetricResult.measured(value=math.nan, confidence=0.5)


def test_unmeasurable_reasons_are_ascii_snake_case() -> None:
    for reason in Unmeasurable:
        assert reason.value.isascii()
        assert reason.value.islower()
        assert " " not in reason.value


def test_default_confidence_threshold_is_documented_value() -> None:
    # CLAUDE.md ve pose-debug skill 0.3 diyor; degistirirsen ikisini de guncelle.
    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.3
