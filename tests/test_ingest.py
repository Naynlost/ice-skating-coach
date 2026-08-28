"""Video on isleme testleri.

Agirlik saf `evaluate_quality()` uzerinde: kabul kapisinin karar mantigini bir
mp4 dosyasi olmadan, elle kurulmus olcumlerle siniyoruz. Video gerektiren
testler entegrasyon niteliginde ve sayica az; sentetik klipler conftest icinde
uretiliyor, depoya video girmiyor.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import make_clip_stats, make_video_info, write_test_video
from core.ingest import (
    ClipStats,
    IngestConfig,
    RejectionReason,
    VideoUnreadableError,
    WarningReason,
    check_quality,
    count_frames,
    evaluate_quality,
    iter_frames,
    measure_clip,
    normalize,
    probe,
    read_frame,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg PATH'te yok")


# ------------------------------------------------------------ IngestConfig


def test_config_defaults_are_usable() -> None:
    cfg = IngestConfig()
    assert cfg.min_fps < cfg.warn_fps_below
    assert cfg.min_duration_s < cfg.max_duration_s


def test_config_target_fps_defaults_to_none() -> None:
    # Varsayilan davranis kare hizini korumak. Buraya sayi yazmak fps dusurur
    # ve kalkis anindaki hassasiyetten goturur.
    assert IngestConfig().target_fps is None


def test_config_loads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "ingest.yaml"
    path.write_text("min_brightness: 55.0\nmax_width: 960\n", encoding="utf-8")
    cfg = IngestConfig.load(path)
    assert cfg.min_brightness == 55.0
    assert cfg.max_width == 960
    assert cfg.min_duration_s == IngestConfig().min_duration_s  # dokunulmayan alan varsayilan


def test_config_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    assert IngestConfig.load(tmp_path / "yok.yaml") == IngestConfig()


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    # Yanlis yazilmis bir anahtar sessizce yok sayilirsa esik degismemis olur
    # ama degistigini sanirsin.
    path = tmp_path / "ingest.yaml"
    path.write_text("min_brightnes: 55.0\n", encoding="utf-8")
    with pytest.raises(Exception, match="min_brightnes"):
        IngestConfig.load(path)


def test_repo_config_file_is_valid() -> None:
    # config/ingest.yaml bozulursa butun hat durur; testte yakalansin.
    assert IngestConfig.load().min_fps > 0


# -------------------------------------------------- evaluate_quality (saf)


def test_good_clip_is_accepted() -> None:
    report = evaluate_quality(make_video_info(), make_clip_stats())
    assert report.accepted
    assert report.rejections == ()
    assert report.summary() == "kabul"


def test_short_clip_rejected_with_reason() -> None:
    report = evaluate_quality(make_video_info(duration_s=0.4), make_clip_stats())
    assert not report.accepted
    assert RejectionReason.TOO_SHORT in report.reasons


def test_long_clip_rejected() -> None:
    report = evaluate_quality(make_video_info(duration_s=45.0), make_clip_stats())
    assert RejectionReason.TOO_LONG in report.reasons


def test_low_resolution_rejected() -> None:
    report = evaluate_quality(make_video_info(width=320, height=240), make_clip_stats())
    assert RejectionReason.RESOLUTION_TOO_LOW in report.reasons


def test_dark_clip_rejected() -> None:
    report = evaluate_quality(make_video_info(), make_clip_stats(mean_brightness=12.0))
    assert RejectionReason.TOO_DARK in report.reasons


def test_shaky_clip_rejected() -> None:
    report = evaluate_quality(make_video_info(), make_clip_stats(shake_px=19.0))
    assert RejectionReason.TOO_SHAKY in report.reasons


def test_very_low_fps_rejected() -> None:
    report = evaluate_quality(make_video_info(fps=12.0), make_clip_stats())
    assert RejectionReason.FPS_TOO_LOW in report.reasons


def test_moderate_fps_warns_but_is_accepted() -> None:
    # 30 fps ile olcum yapilabilir, ama airtime'in karesi yuksekligi verdigi
    # icin tek karelik hata rapora tasar. Reddetmiyoruz, uyariyoruz.
    report = evaluate_quality(make_video_info(fps=30.0), make_clip_stats())
    assert report.accepted
    assert WarningReason.LOW_FPS_PRECISION in {issue.reason for issue in report.warnings}


def test_variable_frame_rate_warns_but_is_accepted() -> None:
    report = evaluate_quality(make_video_info(is_vfr=True), make_clip_stats())
    assert report.accepted
    assert WarningReason.VARIABLE_FRAME_RATE in {issue.reason for issue in report.warnings}


def test_mild_shake_warns_but_is_accepted() -> None:
    cfg = IngestConfig()
    report = evaluate_quality(
        make_video_info(), make_clip_stats(shake_px=cfg.max_shake_px * 0.75), cfg
    )
    assert report.accepted
    assert WarningReason.MILD_SHAKE in {issue.reason for issue in report.warnings}


def test_multiple_faults_all_reported() -> None:
    # Kullaniciya tek tek soylemek yerine hepsini bir defada verelim.
    report = evaluate_quality(
        make_video_info(duration_s=0.3, width=320, height=240),
        make_clip_stats(mean_brightness=5.0, shake_px=30.0),
    )
    assert set(report.reasons) == {
        RejectionReason.TOO_SHORT,
        RejectionReason.RESOLUTION_TOO_LOW,
        RejectionReason.TOO_DARK,
        RejectionReason.TOO_SHAKY,
    }


def test_issue_carries_measured_value_and_threshold() -> None:
    # Kullaniciya "videonuz kotu" degil "0.4 sn, en az 1.0 sn gerekli" diyoruz.
    report = evaluate_quality(make_video_info(duration_s=0.4), make_clip_stats())
    issue = next(i for i in report.rejections if i.reason is RejectionReason.TOO_SHORT)
    assert issue.measured == pytest.approx(0.4)
    assert issue.threshold == pytest.approx(IngestConfig().min_duration_s)
    assert "0.4" in issue.message


def test_thresholds_come_from_config_not_code() -> None:
    strict = IngestConfig(min_brightness=200.0)
    stats = make_clip_stats(mean_brightness=130.0)
    assert evaluate_quality(make_video_info(), stats).accepted
    assert not evaluate_quality(make_video_info(), stats, strict).accepted


def test_summary_lists_rejection_reasons() -> None:
    report = evaluate_quality(make_video_info(duration_s=0.2), make_clip_stats())
    assert report.summary().startswith("ret:")


# --------------------------------------------------------------- probe


@needs_ffmpeg
def test_probe_reads_metadata(good_video: Path) -> None:
    info = probe(good_video)
    assert info.resolution == (640, 480)
    assert info.fps == pytest.approx(60.0, abs=0.1)
    assert info.duration_s == pytest.approx(2.0, abs=0.1)
    assert info.n_frames > 0


def test_probe_missing_file_raises() -> None:
    with pytest.raises(VideoUnreadableError, match="dosya yok"):
        probe(Path("olmayan-dosya.mp4"))


@needs_ffmpeg
def test_probe_rejects_non_video_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("bu bir video degil", encoding="utf-8")
    with pytest.raises(VideoUnreadableError):
        probe(path)


# ---------------------------------------------------------- measure_clip


def test_measure_clip_on_still_video(good_video: Path) -> None:
    stats = measure_clip(good_video)
    assert stats.sampled_frames > 1
    assert stats.mean_brightness > 100.0
    # Sabit tripod: kareler arasi global kayma yok denecek kadar az.
    assert stats.shake_px < 1.0


def test_measure_clip_detects_camera_shake(tmp_path: Path) -> None:
    path = write_test_video(tmp_path / "shaky.mp4", shake_px=25.0)
    assert measure_clip(path).shake_px > IngestConfig().max_shake_px


def test_measure_clip_detects_darkness(tmp_path: Path) -> None:
    path = write_test_video(tmp_path / "dark.mp4", brightness=8)
    assert measure_clip(path).mean_brightness < IngestConfig().min_brightness


def test_measure_clip_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoUnreadableError):
        measure_clip(tmp_path / "yok.mp4")


# ---------------------------------------------------- check_quality (uctan uca)


@needs_ffmpeg
def test_check_quality_accepts_good_video(good_video: Path) -> None:
    report = check_quality(good_video)
    assert report.accepted, report.summary()
    assert report.info is not None
    assert report.stats is not None


@needs_ffmpeg
@pytest.mark.parametrize(
    ("name", "kwargs", "expected"),
    [
        ("short.mp4", {"n_frames": 24, "fps": 60.0}, RejectionReason.TOO_SHORT),
        ("small.mp4", {"size": (320, 240)}, RejectionReason.RESOLUTION_TOO_LOW),
        ("dark.mp4", {"brightness": 8}, RejectionReason.TOO_DARK),
        ("shaky.mp4", {"shake_px": 25.0}, RejectionReason.TOO_SHAKY),
    ],
)
def test_check_quality_rejects_broken_video_with_right_reason(
    tmp_path: Path, name: str, kwargs: dict[str, object], expected: RejectionReason
) -> None:
    """Kasitli bozulmus klipler dogru gerekceyle reddedilmeli.

    Gorev 1.3'un kriteri bunu 3 bozuk video icin istiyor; dordunu de siniyoruz.
    """
    path = write_test_video(tmp_path / name, **kwargs)  # type: ignore[arg-type]
    report = check_quality(path)
    assert not report.accepted
    assert expected in report.reasons


def test_check_quality_returns_report_for_unreadable_file(tmp_path: Path) -> None:
    # 30 videoyu dongude islerken tek bozuk dosya yuzunden durmamali.
    path = tmp_path / "bozuk.mp4"
    path.write_bytes(b"bu bir mp4 degil")
    report = check_quality(path)
    assert not report.accepted
    assert RejectionReason.UNREADABLE in report.reasons
    assert report.info is None


# ------------------------------------------------------------- normalize


@needs_ffmpeg
def test_normalize_keeps_source_fps_by_default(good_video: Path, tmp_path: Path) -> None:
    result = normalize(good_video, tmp_path / "out.mp4")
    assert result.output.fps == pytest.approx(result.source.fps, abs=0.1)
    assert not result.fps_changed
    assert not result.resized


@needs_ffmpeg
def test_normalize_downscales_when_above_max_width(tmp_path: Path) -> None:
    source = write_test_video(tmp_path / "big.mp4", size=(1920, 1080), n_frames=30)
    result = normalize(source, tmp_path / "small.mp4", IngestConfig(max_width=640))
    assert result.resized
    assert result.output.width == 640
    # En boy orani korunmali: 1920x1080 -> 640x360
    assert result.output.height == 360


@needs_ffmpeg
def test_normalize_applies_explicit_target_fps(good_video: Path, tmp_path: Path) -> None:
    result = normalize(good_video, tmp_path / "30fps.mp4", IngestConfig(target_fps=30.0))
    assert result.output.fps == pytest.approx(30.0, abs=0.1)
    assert result.fps_changed


@needs_ffmpeg
def test_normalize_output_is_readable(good_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    normalize(good_video, output)
    assert check_quality(output).accepted


# ---------------------------------------------------------------- frames


def test_iter_frames_yields_zero_based_contiguous_indices(good_video: Path) -> None:
    indices = [frame.index for frame in iter_frames(good_video)]
    assert indices[:5] == [0, 1, 2, 3, 4]
    assert indices == list(range(len(indices)))


def test_iter_frames_respects_half_open_range(good_video: Path) -> None:
    frames = list(iter_frames(good_video, start=10, end=15))
    assert [frame.index for frame in frames] == [10, 11, 12, 13, 14]


def test_iter_frames_frame_has_expected_shape(good_video: Path) -> None:
    frame = next(iter(iter_frames(good_video)))
    assert (frame.width, frame.height) == (640, 480)
    assert frame.image.shape == (480, 640, 3)


def test_iter_frames_rejects_reversed_range(good_video: Path) -> None:
    with pytest.raises(ValueError, match="kucuk olamaz"):
        list(iter_frames(good_video, start=10, end=5))


def test_read_frame_matches_iteration(good_video: Path) -> None:
    from_iter = list(iter_frames(good_video, start=7, end=8))[0]
    direct = read_frame(good_video, 7)
    assert direct.index == 7
    assert (direct.image == from_iter.image).all()


def test_read_frame_beyond_end_raises(good_video: Path) -> None:
    with pytest.raises(VideoUnreadableError):
        read_frame(good_video, 10_000)


def test_count_frames_matches_iteration(good_video: Path) -> None:
    assert count_frames(good_video) == len(list(iter_frames(good_video)))


# ------------------------------------------------------------ ClipStats


def test_clip_stats_rejects_out_of_range_brightness() -> None:
    with pytest.raises(Exception, match="less than or equal to 255"):
        ClipStats(mean_brightness=300.0, shake_px=1.0, sampled_frames=5)
