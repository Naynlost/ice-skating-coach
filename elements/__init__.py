"""Eleman modulleri.

Her eleman ayri bir alt paket ve ayni arayuzu uygular:

    detect(poses)  -> list[ElementInstance]
    segment(inst)  -> dict[str, FrameRange]
    measure(inst)  -> dict[str, MetricResult]

Yeni eleman eklemek burada yeni bir klasor acmak demektir. core/ altinda bir
dosyayi degistirmen gerekiyorsa dur: tasarim yanlis, once arayuzu tartis.
"""
