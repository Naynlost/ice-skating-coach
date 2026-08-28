"""Iskelet dogrulamasi: paketler import ediliyor mu, dizin semasi CLAUDE.md'ye uyuyor mu."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CORE_MODULES = [
    "core",
    "core.cli",
    "core.schema",
    "core.ingest",
    "core.pose",
    "core.geometry",
    "core.keyframe",
    "core.report",
    "core.rules",
]

ELEMENT_MODULES = [
    "elements",
    "elements.jump",
    "elements.jump.segment",
    "elements.jump.metrics",
    "elements.jump.keyframes",
]


@pytest.mark.parametrize("name", CORE_MODULES + ELEMENT_MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_jump_rules_file_exists() -> None:
    assert (ROOT / "elements" / "jump" / "rules.yaml").is_file()


def test_cli_parser_has_expected_commands() -> None:
    from core.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert "analyze" in help_text
    assert "overlay" in help_text
