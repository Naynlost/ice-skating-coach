"""Faz isimleri ile rules.yaml esleşmesi.

segment() yazildiktan sonra (gorev 1.9) buraya gercek faz testleri eklenecek.
Simdilik rules.yaml'daki faz listesinin beklenen ada sahip oldugunu dogruluyoruz;
faz adi ile rules.yaml uyusmazsa kural motoru metrigi SESSIZCE atlar.
"""

from __future__ import annotations

import re
from pathlib import Path

RULES = Path(__file__).resolve().parent.parent / "rules.yaml"
EXPECTED_PHASES = ["hazirlik", "kalkis", "ucus", "inis", "cikis"]


def _phases_from_rules() -> list[str]:
    # pyyaml iskelet asamasinda kurulu degil; tek satirlik liste icin regex yeterli.
    text = RULES.read_text(encoding="utf-8")
    match = re.search(r"^phases:\s*\[(.+)\]\s*$", text, re.MULTILINE)
    assert match is not None, "rules.yaml icinde 'phases' satiri yok"
    return [p.strip() for p in match.group(1).split(",")]


def test_phase_names_match_expected() -> None:
    assert _phases_from_rules() == EXPECTED_PHASES


def test_phase_names_are_ascii_lowercase() -> None:
    for phase in _phases_from_rules():
        assert re.fullmatch(r"[a-z_]+", phase), f"gecersiz faz adi: {phase}"


def test_every_metric_phase_is_declared() -> None:
    text = RULES.read_text(encoding="utf-8")
    used = set(re.findall(r"^\s{4}phase:\s*(\S+)\s*$", text, re.MULTILINE))
    assert used, "rules.yaml icinde hic metrik fazi yok"
    assert used <= set(EXPECTED_PHASES), f"bilinmeyen faz: {used - set(EXPECTED_PHASES)}"
