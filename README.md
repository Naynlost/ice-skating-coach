# Buz Pateni Video Kocluk Sistemi

Kurallar ve mimari `../CLAUDE.md`, gorev sirasi `../TASKS.md`,
nerede kalindigi `docs/status.md` icinde.

## Kurulum

Sistem gereksinimleri: **Python 3.11+** ve **ffmpeg** (PATH'te olmali;
`ffprobe` ayni pakette gelir). Windows'ta `winget install Gyan.FFmpeg`.

```bash
git clone <repo-url>
cd Ice/ice-skating-coach

python -m venv .venv
source .venv/Scripts/activate    # PowerShell: .venv\Scripts\Activate.ps1

pip install -e ".[core,dev]"     # sema, video isleme, testler
pip install -e ".[pose]"         # YOLO11 + ByteTrack (torch getirir, ~2 GB)
```

Bagimlilik gruplari bilerek ayri: `core` (numpy/scipy/opencv/pydantic),
`pose` (ultralytics/lap/rtmlib/onnxruntime), `serve` (anthropic/jinja2).
Her grup ilgili gorevde kurulur, iskelet asamasinda agir paket gelmez.

Kurulumu dogrula:

```bash
python -m pytest         # 142 test gecmeli
python -m ruff check .
python -m mypy
```

### Repoya girmeyen, yeni makinede gereken seyler

| Ne | Nasil gelir |
|---|---|
| YOLO agirligi (`yolo11m.pt`, ~39 MB) | Ilk analizde ultralytics kendisi indirir |
| RTMPose agirligi | Gorev 1.5'te rtmlib kendisi indirir |
| Test videolari (`data/raw/`) | Elle konacak, sartlar `docs/status.md` icinde |
| `.venv/` | Yukaridaki komutlarla kurulur |

`data/` ve `*.pt` bilerek `.gitignore`'da: sporcu videolari ve model
agirliklari depoya girmez.

### Claude Code ile calisiyorsan

Iki skill deponun kokunde duruyor (`pose-debug-SKILL.md`,
`element-module-SKILL.md`). Otomatik tetiklenmeleri icin kopyala:

```powershell
$s = "$env:USERPROFILE\.claude\skills"
New-Item -ItemType Directory -Force "$s\pose-debug"
New-Item -ItemType Directory -Force "$s\element-module"
Copy-Item "..\pose-debug-SKILL.md"     "$s\pose-debug\SKILL.md"
Copy-Item "..\element-module-SKILL.md" "$s\element-module\SKILL.md"
```

## Komutlar

`make` Windows'ta kurulu degilse `make.ps1` ayni hedefleri calistirir.

| Amac | make | Windows |
|---|---|---|
| birim testler | `make test` | `.\make.ps1 test` |
| ruff + mypy | `make lint` | `.\make.ps1 lint` |
| otomatik duzeltme | `make fmt` | `.\make.ps1 fmt` |
| kalite kontrolu | `make check D=data/raw` | `.\make.ps1 check -D data\raw` |
| tek video isle | `make analyze V=data/raw/x.mp4 E=jump L=beginner` | `.\make.ps1 analyze -V data\raw\x.mp4` |
| poz overlay | `make overlay V=data/raw/x.mp4` | `.\make.ps1 overlay -V data\raw\x.mp4` |

`analyze` ve `overlay` henuz uygulanmadi (Gorev 3.4 ve 1.6).

## Durum

Faz 1.1 (iskelet), 1.2 (veri tipleri), 1.3 (video on isleme) ve
1.4 (kisi tespiti + takip) kod tarafinda tamam; 142 test geciyor.

Kalan moduller bilerek bos: her biri kendi gorevinde doldurulur, dosyalarin
docstring'i hangi gorev oldugunu yaziyor.

Ayrinti icin `docs/status.md`.
