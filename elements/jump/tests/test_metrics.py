"""Sicrama metrik testleri.

Gorev 1.10'da her metrik icin sentetik PoseSequence ile bir test eklenecek.
Testler videoya bagimli olmayacak; elle kurulmus poz verisi yeterli.
"""

from __future__ import annotations

import elements.jump.metrics as metrics


def test_metrics_module_importable() -> None:
    assert metrics is not None
