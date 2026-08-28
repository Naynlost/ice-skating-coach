"""Sicramayi fazlara bol.

Gorev 1.9'da doldurulacak.

Yontem: ayak bileginin buz cizgisinden ayrildigi kare = takeoff, geri dondugu
kare = landing. Kalan fazlar bu iki kareden turetilir.

Faz isimleri rules.yaml'daki `phase` alanlariyla BIREBIR ayni olmali. Eslesmezse
kural motoru o metrigi sessizce atlar; bunu yakalayan bir dogrulama testi var
(tests/test_segment.py).
"""
