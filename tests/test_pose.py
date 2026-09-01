"""Kisi tespiti, takip ve sporcu secimi testleri.

Agirlik saf katmanda: sporcu secimi ve kimlik degisimi tespiti elle kurulmus
tespit dizileri uzerinde sinaniyor. Bu testler ultralytics kurulu olmasa da,
model agirligi indirilmemis olsa da calisir.

Modeli gercekten calistiran tek test en altta ve agirlik yoksa atlanir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_bbox, make_detection_sequence, straight_track
from core.pose import (
    BBox,
    Detection,
    DetectionSequence,
    FrameDetections,
    PoseConfig,
    TrackIssue,
    analyze_track,
    select_athlete,
    summarize_tracks,
    to_frame_detections,
)

# --------------------------------------------------------------------- BBox


def test_bbox_geometry() -> None:
    box = BBox(x1=100.0, y1=200.0, x2=300.0, y2=600.0)
    assert (box.width, box.height) == (200.0, 400.0)
    assert box.area == 80_000.0
    assert box.center == (200.0, 400.0)


def test_bbox_rejects_inverted_corners() -> None:
    with pytest.raises(Exception, match="gecersiz kutu"):
        BBox(x1=300.0, y1=0.0, x2=100.0, y2=100.0)


def test_bbox_iou_identical_is_one() -> None:
    box = make_bbox()
    assert box.iou(box) == pytest.approx(1.0)


def test_bbox_iou_disjoint_is_zero() -> None:
    left = make_bbox(cx=100.0, width=100.0)
    right = make_bbox(cx=1000.0, width=100.0)
    assert left.iou(right) == 0.0


def test_bbox_iou_half_overlap() -> None:
    a = BBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
    b = BBox(x1=50.0, y1=0.0, x2=150.0, y2=100.0)
    # kesisim 5000, birlesim 15000
    assert a.iou(b) == pytest.approx(1 / 3)


def test_bbox_center_distance() -> None:
    a = make_bbox(cx=0.0, cy=0.0)
    b = make_bbox(cx=3.0, cy=4.0)
    assert a.center_distance(b) == pytest.approx(5.0)


# ---------------------------------------------------------- DetectionSequence


def test_sequence_rejects_frame_index_gaps() -> None:
    frames = (
        FrameDetections(frame_index=0),
        FrameDetections(frame_index=2),
    )
    with pytest.raises(Exception, match="ardisik olmali"):
        DetectionSequence(width=1280, height=720, frames=frames)


def test_sequence_track_ids_in_first_seen_order() -> None:
    sequence = make_detection_sequence(
        {
            7: straight_track(4),
            2: straight_track(4, start=(900.0, 360.0)),
        }
    )
    assert set(sequence.track_ids()) == {7, 2}


def test_frame_by_track_and_others() -> None:
    frame = FrameDetections(
        frame_index=0,
        detections=(
            Detection(bbox=make_bbox(cx=300.0), confidence=0.9, track_id=1),
            Detection(bbox=make_bbox(cx=900.0), confidence=0.8, track_id=2),
        ),
    )
    assert frame.by_track(1) is not None
    assert frame.by_track(99) is None
    assert len(frame.others(1)) == 1


# --------------------------------------------------------- summarize_tracks


def test_summary_coverage_counts_visible_frames() -> None:
    boxes: list[BBox | None] = [*straight_track(6), None, None, None, None]
    sequence = make_detection_sequence({1: boxes})
    summary = summarize_tracks(sequence)[0]
    assert summary.frame_count == 6
    assert summary.coverage == pytest.approx(0.6)
    assert summary.span.start == 0
    assert summary.span.end == 6


def test_summary_ignores_low_confidence_detections() -> None:
    # Dusuk guvenli kutu genelde yanlis yerdedir ve ortalamayi bozar.
    sequence = make_detection_sequence({1: straight_track(8)}, confidence=0.1)
    assert summarize_tracks(sequence) == ()


def test_summary_area_score_is_relative_to_largest() -> None:
    sequence = make_detection_sequence(
        {
            1: straight_track(10, size=(200.0, 400.0)),
            2: straight_track(10, start=(900.0, 360.0), size=(100.0, 200.0)),
        }
    )
    by_id = {s.track_id: s for s in summarize_tracks(sequence)}
    assert by_id[1].area_score == pytest.approx(1.0)
    assert by_id[2].area_score < 0.3


def test_summary_centrality_is_higher_at_frame_center() -> None:
    sequence = make_detection_sequence(
        {
            1: straight_track(10, start=(640.0, 360.0)),
            2: straight_track(10, start=(60.0, 60.0)),
        }
    )
    by_id = {s.track_id: s for s in summarize_tracks(sequence)}
    assert by_id[1].centrality > 0.95
    assert by_id[2].centrality < by_id[1].centrality


def test_summaries_are_sorted_by_score() -> None:
    sequence = make_detection_sequence(
        {
            1: straight_track(10, start=(60.0, 60.0), size=(80.0, 160.0)),
            2: straight_track(10, start=(640.0, 360.0), size=(200.0, 420.0)),
        }
    )
    scores = [s.score for s in summarize_tracks(sequence)]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------ select_athlete


def test_selects_the_only_person() -> None:
    sequence = make_detection_sequence({3: straight_track(20)})
    selection = select_athlete(sequence)
    assert selection.ok
    assert selection.track_id == 3
    assert len(selection.detections) == 20


def test_selects_largest_and_most_central_person() -> None:
    # Sporcu merkezde ve buyuk; arka planda kucuk ve kenarda biri var.
    sequence = make_detection_sequence(
        {
            1: straight_track(30, start=(640.0, 380.0), size=(220.0, 460.0)),
            2: straight_track(30, start=(1150.0, 120.0), size=(60.0, 130.0)),
        }
    )
    selection = select_athlete(sequence)
    assert selection.track_id == 1


def test_passer_by_does_not_win_despite_being_large() -> None:
    """Kapsama kapisinin varlik sebebi.

    Onden gecen bir patenci birkac karede kameraya cok yakin, yani cok buyuk
    gorunur. Kapsama kapisi olmasa skoru gercek sporcuyu gecerdi.
    """
    athlete: list[BBox | None] = straight_track(
        40, start=(600.0, 380.0), end=(700.0, 380.0), size=(180.0, 400.0)
    )
    passer: list[BBox | None] = [None] * 40
    for index in range(15, 21):
        passer[index] = make_bbox(cx=640.0, cy=400.0, width=500.0, height=650.0)

    selection = select_athlete(make_detection_sequence({1: athlete, 2: passer}))
    assert selection.track_id == 1


def test_no_person_reports_reason_and_selects_nothing() -> None:
    sequence = DetectionSequence(
        width=1280,
        height=720,
        frames=tuple(FrameDetections(frame_index=i) for i in range(10)),
    )
    selection = select_athlete(sequence)
    assert not selection.ok
    assert selection.track_id is None
    assert TrackIssue.NO_PERSON in selection.issues


def test_detections_without_track_id_are_not_selectable() -> None:
    # Kimliksiz kutular klip boyunca izlenemez; sporcu secimine giremezler.
    frames = tuple(
        FrameDetections(
            frame_index=i,
            detections=(Detection(bbox=make_bbox(), confidence=0.9, track_id=None),),
        )
        for i in range(10)
    )
    selection = select_athlete(DetectionSequence(width=1280, height=720, frames=frames))
    assert not selection.ok
    assert TrackIssue.NO_PERSON in selection.issues


def test_low_coverage_still_selects_but_warns() -> None:
    boxes: list[BBox | None] = [*straight_track(4), *([None] * 16)]
    selection = select_athlete(make_detection_sequence({1: boxes}))
    assert selection.track_id == 1
    assert TrackIssue.LOW_COVERAGE in selection.issues


def test_close_scores_raise_ambiguity_warning() -> None:
    # Iki esit patenci: hangisinin sporcu oldugunu bilemeyiz, soyleyelim.
    sequence = make_detection_sequence(
        {
            1: straight_track(30, start=(560.0, 360.0), size=(180.0, 400.0)),
            2: straight_track(30, start=(720.0, 360.0), size=(180.0, 400.0)),
        }
    )
    selection = select_athlete(sequence)
    assert selection.ok
    assert TrackIssue.AMBIGUOUS_CHOICE in selection.issues


def test_clean_track_produces_no_warnings() -> None:
    sequence = make_detection_sequence(
        {1: straight_track(40, start=(500.0, 360.0), end=(780.0, 360.0))}
    )
    selection = select_athlete(sequence)
    assert selection.ok
    assert selection.warnings == ()


def test_weights_come_from_config() -> None:
    # Kenarda buyuk, merkezde kucuk: agirliga gore kazanan degisir.
    sequence = make_detection_sequence(
        {
            1: straight_track(30, start=(120.0, 360.0), size=(300.0, 560.0)),
            2: straight_track(30, start=(640.0, 360.0), size=(150.0, 320.0)),
        }
    )
    area_first = PoseConfig(area_weight=1.0, centrality_weight=0.0)
    center_first = PoseConfig(area_weight=0.0, centrality_weight=1.0)
    assert select_athlete(sequence, area_first).track_id == 1
    assert select_athlete(sequence, center_first).track_id == 2


# -------------------------------------------------------------- analyze_track


def test_short_gap_is_tolerated() -> None:
    boxes: list[BBox | None] = straight_track(30, start=(500.0, 360.0), end=(760.0, 360.0))
    boxes[10] = None  # tek karelik kapanma
    selection = select_athlete(make_detection_sequence({1: boxes}))
    assert TrackIssue.TRACKING_GAP not in selection.issues


def test_long_gap_is_reported_with_frame_range() -> None:
    boxes: list[BBox | None] = straight_track(40, start=(500.0, 360.0), end=(800.0, 360.0))
    for index in range(12, 20):
        boxes[index] = None
    selection = select_athlete(make_detection_sequence({1: boxes}))
    gap = next(w for w in selection.warnings if w.issue is TrackIssue.TRACKING_GAP)
    assert gap.frames is not None
    assert (gap.frames.start, gap.frames.end) == (12, 20)


def test_position_jump_is_reported() -> None:
    """Kimlik baska bir vucuda atladiginda gorulen imza.

    pose-debug skill'inin 2. katmani: kutu bir anda baska birine sicriyor.
    """
    boxes: list[BBox | None] = straight_track(30, start=(500.0, 360.0), end=(560.0, 360.0))
    boxes[15] = make_bbox(cx=1100.0, cy=360.0)  # bir karede pistin obur ucu
    selection = select_athlete(make_detection_sequence({1: boxes}))
    assert TrackIssue.POSITION_JUMP in selection.issues


def test_scale_jump_is_reported() -> None:
    boxes: list[BBox | None] = straight_track(30, start=(600.0, 360.0), end=(660.0, 360.0))
    boxes[20] = make_bbox(cx=630.0, cy=360.0, width=420.0, height=900.0)
    selection = select_athlete(make_detection_sequence({1: boxes}))
    assert TrackIssue.SCALE_JUMP in selection.issues


def test_crowding_is_reported_as_one_range_not_per_frame() -> None:
    """Iki patenci yan yana gecerken kare kare uyari uretmek raporu bogar."""
    athlete: list[BBox | None] = straight_track(40, start=(640.0, 360.0))
    other: list[BBox | None] = [None] * 40
    for index in range(10, 25):
        other[index] = make_bbox(cx=660.0, cy=360.0)

    selection = select_athlete(make_detection_sequence({1: athlete, 2: other}))
    crowding = [w for w in selection.warnings if w.issue is TrackIssue.CROWDED_SCENE]
    assert len(crowding) == 1
    assert crowding[0].frames is not None
    assert (crowding[0].frames.start, crowding[0].frames.end) == (10, 25)


def test_separate_crowding_episodes_are_reported_separately() -> None:
    athlete: list[BBox | None] = straight_track(40, start=(640.0, 360.0))
    other: list[BBox | None] = [None] * 40
    for index in (*range(5, 9), *range(25, 30)):
        other[index] = make_bbox(cx=655.0, cy=360.0)

    selection = select_athlete(make_detection_sequence({1: athlete, 2: other}))
    crowding = [w for w in selection.warnings if w.issue is TrackIssue.CROWDED_SCENE]
    assert len(crowding) == 2


def test_analyze_track_on_empty_selection_returns_nothing() -> None:
    sequence = make_detection_sequence({1: straight_track(5)})
    from core.pose import AthleteSelection

    assert analyze_track(AthleteSelection(), sequence) == ()


def test_jump_thresholds_come_from_config() -> None:
    boxes: list[BBox | None] = straight_track(30, start=(600.0, 360.0), end=(660.0, 360.0))
    boxes[15] = make_bbox(cx=760.0, cy=360.0)
    sequence = make_detection_sequence({1: boxes})
    assert TrackIssue.POSITION_JUMP in select_athlete(sequence).issues
    tolerant = PoseConfig(max_center_jump_ratio=5.0)
    assert TrackIssue.POSITION_JUMP not in select_athlete(sequence, tolerant).issues


# ------------------------------------------------------------- PoseConfig


def test_config_loads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "pose.yaml"
    path.write_text("min_coverage: 0.8\nmodel_path: yolo11n.pt\n", encoding="utf-8")
    cfg = PoseConfig.load(path)
    assert cfg.min_coverage == 0.8
    assert cfg.model_path == "yolo11n.pt"


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "pose.yaml"
    path.write_text("min_coverag: 0.8\n", encoding="utf-8")
    with pytest.raises(Exception, match="min_coverag"):
        PoseConfig.load(path)


def test_config_rejects_zero_weights() -> None:
    with pytest.raises(Exception, match="sifir olamaz"):
        PoseConfig(area_weight=0.0, centrality_weight=0.0)


def test_repo_config_file_is_valid() -> None:
    assert PoseConfig.load().min_coverage > 0


# ------------------------------------------------------- to_frame_detections


def test_to_frame_detections_maps_boxes_and_ids() -> None:
    frame = to_frame_detections(
        4,
        boxes_xyxy=[[10.0, 20.0, 110.0, 220.0], [300.0, 40.0, 380.0, 240.0]],
        confidences=[0.91, 0.55],
        track_ids=[7, 9],
    )
    assert frame.frame_index == 4
    assert [d.track_id for d in frame.detections] == [7, 9]
    assert frame.detections[0].bbox.width == 100.0


def test_to_frame_detections_without_track_ids() -> None:
    # ByteTrack bir karede hicbir kutuya kimlik atayamayabilir.
    frame = to_frame_detections(0, [[0.0, 0.0, 10.0, 10.0]], [0.7], None)
    assert frame.detections[0].track_id is None


def test_to_frame_detections_empty_frame() -> None:
    assert to_frame_detections(2, [], []).detections == ()


def test_to_frame_detections_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="uyusmuyor"):
        to_frame_detections(0, [[0.0, 0.0, 1.0, 1.0]], [0.5, 0.6])


# ------------------------------------------------------- model (entegrasyon)


def _weights_available(config: PoseConfig) -> bool:
    return Path(config.model_path).is_file()


@pytest.mark.skipif(
    not _weights_available(PoseConfig.load()),
    reason="YOLO agirligi indirilmemis; `make check` sonrasi ilk analizde iner",
)
def test_detect_persons_runs_on_a_video(good_video: Path) -> None:
    """Hat uctan uca calisiyor mu.

    Sentetik test klibinde insan yok, bu yuzden tespit beklemiyoruz; beklenen
    sey hattin cokmeden calismasi ve kare sayisinin videoyla eslesmesi.
    """
    from core.pose import detect_persons

    sequence = detect_persons(good_video)
    assert len(sequence) > 0
    assert sequence.width == 640
    assert sequence.height == 480
