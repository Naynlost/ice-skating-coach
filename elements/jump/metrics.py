"""Sicrama metrikleri.

Gorev 1.10'da doldurulacak.

Kurallar:
- Fonksiyonlar saf. Dosya okumaz, aga cikmaz, log yazmaz, LLM cagirmaz.
- Acilar derece doner.
- Gereken keypoint'lerin guveni 0.3 altindaysa
  MetricResult(value=None, confidence=conf, reason="dusuk_keypoint_guveni") doner.
- Her metrigin sentetik poz verisiyle calisan birim testi olur.

Sicrama yuksekligi havada kalma suresinden cikar: h = 9.81 * airtime**2 / 8.
Kamera kalibrasyonu gerekmez, tek gereken dogru takeoff ve landing karesi.
"""
