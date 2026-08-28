# Buz Pateni Video Kocluk Sistemi

Kurallar ve mimari `../CLAUDE.md`, gorev sirasi `../TASKS.md` icinde.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[core,dev]"
```

Bagimlilik gruplari bilerek ayri: `core` (numpy/scipy/opencv/pydantic),
`pose` (ultralytics/rtmlib/onnxruntime), `serve` (fastapi/anthropic).
Her grup ilgili gorevde kurulur, iskelet asamasinda agir paket gelmez.

## Komutlar

`make` Windows'ta kurulu degilse `make.ps1` ayni hedefleri calistirir.

| Amac | make | Windows |
|---|---|---|
| birim testler | `make test` | `.\make.ps1 test` |
| ruff + mypy | `make lint` | `.\make.ps1 lint` |
| otomatik duzeltme | `make fmt` | `.\make.ps1 fmt` |
| tek video isle | `make analyze V=data/raw/x.mp4 E=jump L=beginner` | `.\make.ps1 analyze -V data\raw\x.mp4` |
| poz overlay | `make overlay V=data/raw/x.mp4` | `.\make.ps1 overlay -V data\raw\x.mp4` |
| kalite kontrolu | `make check D=data/raw` | `.\make.ps1 check -D data\raw` |

## Durum

Faz 1.1 (iskelet), 1.2 (veri tipleri) ve 1.3 (video on isleme) tamam.
Kalan moduller bilerek bos:
her biri kendi gorevinde doldurulur, dosyalarin docstring'i hangi gorev
oldugunu yaziyor.
