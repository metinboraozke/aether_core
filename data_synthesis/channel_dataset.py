"""Hibrit kanal veri seti — paired Teacher-Student loader (PyTorch).

REVİZE 2026-05-14: Veri artık tek `.npy` değil, generate_hybrid_sionna.py'nin
ürettiği 6 ayrı dosya. Bu Dataset onları paired sample olarak okur.

Beklenen dosyalar (data_dir altında):
    uwb_cir_oracle.npy    [N, 6, 32, 2]   (Teacher input — KD hedefi)
    uwb_cir_anchor.npy    [N, 4, 32, 2]   (Student input — model girdisi)
    slot_labels.npy       [N, 6]
    material_labels.npy   [N, 6]
    codeword_labels.npy   [N, 6, 7]
    data_id_labels.npy    [N, 6]

TODO:
    * class HybridChannelDataset(torch.utils.data.Dataset):
        - __init__(data_dir, return_oracle=True, transform=None)
        - __getitem__(i):
            return {
                'cir_anchor':  tensor [4, 32, 2]   # student input
                'cir_oracle':  tensor [6, 32, 2]   # KD hedefi (varsa)
                'slot':        tensor [6]
                'material':    tensor [6]
                'codeword':    tensor [6, 7]
                'data_id':     tensor [6]
            }
    * data_dir'da uwb_cir_oracle.npy yoksa eğitim modunda hata, inference
      modunda sadece anchor döndür.
    * (opsiyonel) tensor_format.py preprocessing pipeline'ı uygula (transform).
"""
