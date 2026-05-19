"""Sionna RT sahne kurulumu ve malzeme tanımlama.

Modül 1 — Adım 1 (README.md):
    - sionna.rt.Scene → 10 m × 5 m × 3 m oda.
    - load_scene → .obj veya .mitsuba modeli (6 bölmeli raf).
    - Malzemeler: concrete (Lambertian), metal (Specular/PEC),
      ahşap/plastik/karton (rastgele ε_r).

TODO:
    * configs/scene.yaml'ı oku.
    * Sionna Scene objesi inşa et.
    * BSDF parametrelerini malzeme başına ayarla.
    * Raf slot'larına stochastic offset uygula (mm hassasiyet, açı).
"""
