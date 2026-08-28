"""ffprobe ile video ust verisi.

Kare hizini `avg_frame_rate` yerine kesirli alanlardan okuyoruz cunku ffprobe
bunu "60000/1001" gibi bir rasyonel olarak verir; float'a cevirmeyi kendimiz
yaparsak 59.94 ile 60 arasindaki farki kaybetmeyiz.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["FFprobeMissingError", "VideoInfo", "VideoUnreadableError", "probe"]

_PROBE_TIMEOUT_S = 60


class FFprobeMissingError(RuntimeError):
    """ffprobe PATH'te yok. Kurulum eksigi, video hatasi degil."""


class VideoUnreadableError(RuntimeError):
    """Dosya acilamadi ya da icinde video akisi yok."""


class VideoInfo(BaseModel):
    """Bir klibin ham ust verisi. Yorum yok, olcum de yok; sadece ne oldugu."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)
    n_frames: int = Field(ge=0)
    codec: str
    is_vfr: bool

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


def _require_tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        msg = f"{name} PATH'te bulunamadi. ffmpeg kurulu mu?"
        raise FFprobeMissingError(msg)
    return found


def _rate(value: str | None) -> float | None:
    """ffprobe'un "60000/1001" bicimindeki rasyonelini float'a cevir."""
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _stream_of(payload: dict[str, Any]) -> dict[str, Any]:
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video":
            return dict(stream)
    msg = "dosyada video akisi yok"
    raise VideoUnreadableError(msg)


def probe(path: Path) -> VideoInfo:
    """Videonun ust verisini oku.

    Dosya sistemine dokunur; saf degil. Kalite karari veren kod bunun ciktisini
    alir, dosyayi kendisi acmaz.
    """
    if not path.is_file():
        msg = f"dosya yok: {path}"
        raise VideoUnreadableError(msg)

    binary = _require_tool("ffprobe")
    command = [
        binary,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"ffprobe zaman asimi: {path}"
        raise VideoUnreadableError(msg) from exc

    if completed.returncode != 0:
        msg = f"ffprobe okuyamadi: {path} ({completed.stderr.strip()})"
        raise VideoUnreadableError(msg)

    try:
        payload: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        msg = f"ffprobe ciktisi cozulemedi: {path}"
        raise VideoUnreadableError(msg) from exc

    stream = _stream_of(payload)

    avg_fps = _rate(stream.get("avg_frame_rate"))
    nominal_fps = _rate(stream.get("r_frame_rate"))
    fps = avg_fps or nominal_fps
    if fps is None:
        msg = f"kare hizi okunamadi: {path}"
        raise VideoUnreadableError(msg)

    duration = _first_float(stream.get("duration"), payload.get("format", {}).get("duration"))
    n_frames = _first_int(stream.get("nb_frames"))
    if duration is None and n_frames is not None:
        duration = n_frames / fps
    if duration is None or duration <= 0:
        msg = f"sure okunamadi: {path}"
        raise VideoUnreadableError(msg)
    if n_frames is None:
        n_frames = int(round(duration * fps))

    width = _first_int(stream.get("width"))
    height = _first_int(stream.get("height"))
    if width is None or height is None:
        msg = f"cozunurluk okunamadi: {path}"
        raise VideoUnreadableError(msg)

    # avg_frame_rate gercek ortalamadir, r_frame_rate nominal tabandir.
    # Ikisi belirgin ayrisiyorsa kare hizi degiskendir (VFR).
    is_vfr = (
        avg_fps is not None
        and nominal_fps is not None
        and abs(avg_fps - nominal_fps) / nominal_fps > 0.01
    )

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration,
        n_frames=max(n_frames, 0),
        codec=str(stream.get("codec_name", "bilinmiyor")),
        is_vfr=is_vfr,
    )


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, "N/A"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value in (None, "N/A"):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
