"""LEGACY STUB — yerini generate_hybrid_sionna.py aldı.

Bu dosya REVİZE 2026-05-14'te kullanım dışı bırakıldı. Yeni Teacher-Student
KD pipeline için entry point:

    python -m data_synthesis.generate_hybrid_sionna --help

Eski PDF'in `[N, 6, 32, 2]` tek tensör mantığı, yerini paired Teacher-Student
şemasına bıraktı. Detay: README.md → "Tasarım Kararları" bölümü.

Geriye dönük import çağrılarını yakalamak için boş bırakıldı.
"""

import warnings


def main(*args, **kwargs):
    warnings.warn(
        "generate_hybrid.py legacy oldu; generate_hybrid_sionna.py kullanın.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(
        "Bu script artık kullanılmıyor. Çalıştırın:\n"
        "  python -m data_synthesis.generate_hybrid_sionna --help"
    )


if __name__ == "__main__":
    main()
