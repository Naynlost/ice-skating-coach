.PHONY: help install dev test lint fmt analyze overlay check clean

PY ?= python
V  ?=
E  ?= jump
L  ?= beginner
D  ?= data/raw

help:
	@echo "install  - gelistirme bagimliliklarini kur"
	@echo "dev      - API gelistirme modunda (Faz 4)"
	@echo "test     - birim testler"
	@echo "lint     - ruff + mypy"
	@echo "fmt      - ruff format + otomatik duzeltme"
	@echo "analyze  - make analyze V=data/raw/x.mp4 E=jump L=beginner"
	@echo "overlay  - make overlay V=data/raw/x.mp4"
	@echo "check    - make check D=data/raw (kalite kontrolu)"

install:
	$(PY) -m pip install -e ".[core,dev]"

dev:
	$(PY) -m uvicorn api.main:app --reload

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m mypy

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

analyze:
	$(PY) -m core.cli analyze --video "$(V)" --element "$(E)" --level "$(L)"

overlay:
	$(PY) -m core.cli overlay --video "$(V)"

check:
	$(PY) -m core.cli check --dir "$(D)"

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path(\x27.\x27).rglob(\x27__pycache__\x27)]"
