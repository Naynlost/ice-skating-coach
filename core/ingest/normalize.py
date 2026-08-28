"""ffmpeg ile normalizasyon: sabit kare hizi ve cozunurluk siniri.

Normalizasyonun asil isi kare hizini SABITLEMEK, dusurmek degil. Telefonlar
sik sik degisken kare hizli (VFR) kayit yapar; boyle bir dosyada n numarali
karenin zamani n/fps degildir. Airtime kare sayisindan hesaplandigi icin VFR
bir klipte sicrama yuksekligi sessizce yanlis cikar.

Bu yuzden `target_fps` varsayilan olarak None: kaynagin hizi korunur, video
yalnizca CFR'ye cevrilir. Bilerek bir sayi vermedikce fps dusurulmez.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from core.ingest.config import IngestConfig
from core.ingest.probe import VideoInfo, probe

__all__ = ["FFmpegError", "NormalizeResult", "normalize"]

_ENCODE_TIMEOUT_S = 900


class FFmpegError(RuntimeError):
    """ffmpeg cagirisi basarisiz oldu."""


class NormalizeResult(BaseModel):
    """Normalizasyon ciktisi ve ne degistiginin kaydi.

    `source` ve `output` ust verileri birlikte duruyor ki sonraki katman hangi
    donusumun uygulandigini rapora yazabilsin. Olcumun hangi kare hizi
    uzerinden yapildigini bilmek zorundayiz.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: VideoInfo
    output: VideoInfo
    resized: bool
    fps_changed: bool


def _require_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found is None:
        msg = "ffmpeg PATH'te bulunamadi"
        raise FFmpegError(msg)
    return found


def _scale_filter(info: VideoInfo, max_width: int) -> str | None:
    """Genislik sinirini asan videoyu en-boy oranini koruyarak kucult.

    Yukseklik -2 ile veriliyor: ffmpeg oranı korur ve sonucu 2'nin katina
    yuvarlar, cunku H.264 tek sayili boyut kabul etmez.
    """
    if info.width <= max_width:
        return None
    return f"scale={max_width}:-2:flags=lanczos"


def normalize(
    path: Path,
    output: Path,
    config: IngestConfig | None = None,
    *,
    info: VideoInfo | None = None,
) -> NormalizeResult:
    """Videoyu sabit kare hizina cevir ve gerekiyorsa kucult.

    Ses akisi atilir: olcum katmani sesle ilgilenmez ve atmak dosyayi kucultur.

    Cikti dosyasi varsa uzerine yazilir. Cagiran taraf ciktinin nereye
    gidecegini bilir; burada gecici dizin secmiyoruz.
    """
    cfg = config or IngestConfig()
    source = info or probe(path)
    binary = _require_ffmpeg()

    output.parent.mkdir(parents=True, exist_ok=True)

    target_fps = cfg.target_fps or source.fps
    scale = _scale_filter(source, cfg.max_width)

    command = [binary, "-y", "-v", "error", "-i", str(path)]
    if scale is not None:
        command += ["-vf", scale]
    command += [
        # -fps_mode cfr kareleri kopyalayarak/atarak sabit hiza oturtur;
        # VFR kaynakta kare indeksi ile zaman iliskisini bu kurar.
        "-fps_mode", "cfr",
        "-r", f"{target_fps:.6f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output),
    ]

    try:
        completed = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=_ENCODE_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"ffmpeg zaman asimi: {path}"
        raise FFmpegError(msg) from exc

    if completed.returncode != 0 or not output.is_file():
        msg = f"ffmpeg basarisiz: {path} ({completed.stderr.strip()})"
        raise FFmpegError(msg)

    result_info = probe(output)
    return NormalizeResult(
        source=source,
        output=result_info,
        resized=scale is not None,
        fps_changed=abs(result_info.fps - source.fps) > 0.01,
    )
