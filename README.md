# Aether Core

> **2.4 GHz WiFi CSI + UWB CIR tabanlı çok modlu RF-vizyon sistemi.**
> 50 m² (10 × 5 × 3 m) depo / sınıf ortamında 6 bölmeli rafın **3B dijital ikizini** çıkarır; her bölmedeki nesnenin **doluluğunu**, **malzeme türünü** ve **spektral barkod ID'sini** gerçek zamanlı (10–20 Hz, < 200 ms gecikme) tespit eder.
>
> Bu README, projenin tek başvuru kaynağıdır — `DATASHEETTTTTTT.pdf`'in (13 sayfa) tüm teknik içeriği birebir korunarak buraya gömülmüştür.

---

## İçindekiler

1. [Mimari Genel Bakış](#mimari-genel-bakış)
2. [Modül 1 — Veri Sentezi ve Fiziksel Gürültü Modelleme (Sionna 2.0)](#modül-1--veri-sentezi-ve-fiziksel-gürültü-modelleme-sionna-20)
3. [Modül 2 — Çok Kademeli Ön İşleme (Pre-processing) Pipeline](#modül-2--çok-kademeli-ön-işleme-pre-processing-pipeline)
4. [Modül 3 — Dual-Stream Encoder Mimarisi (The Eyes)](#modül-3--dual-stream-encoder-mimarisi-the-eyes)
5. [Modül 4 — Hiyerarşik Cascade DRI ve Voxel Tahmini (The Brain)](#modül-4--hiyerarşik-cascade-dri-ve-voxel-tahmini-the-brain)
6. [Modül 5 — Sim-to-Real Adaptasyonu ve Real-Time Dashboard (The Bridge)](#modül-5--sim-to-real-adaptasyonu-ve-real-time-dashboard-the-bridge)
7. [Sionna 2.0 Mimari ve Uygulama Datasheet (Aether Core Özel)](#sionna-20-mimari-ve-uygulama-datasheet-aether-core-özel)
8. [Veri Formatları](#veri-formatları)
9. [Dinamiklik (Stochasticity) Stratejisi](#dinamiklik-stochasticity-stratejisi)
10. [Çalışma İlkeleri (Claude Code'a Notlar)](#çalışma-i̇lkeleri-claude-codea-notlar)
11. [Klasör Yapısı](#klasör-yapısı)
12. [Kurulum](#kurulum)
13. [Geliştirme Yol Haritası](#geliştirme-yol-haritası)

---

## Mimari Genel Bakış

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                AETHER CORE                                   │
│                                                                              │
│   ┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────────┐   │
│   │  Modül 1    │    │  Modül 2     │    │  Modül 3   │    │  Modül 4    │   │
│   │  Sionna 2.0 │───▶│  Pre-process │───▶│  Dual      │───▶│  Cascade    │   │
│   │  Veri       │    │  Pipeline    │    │  Encoder   │    │  DRI +      │   │
│   │  Sentezi    │    │  (6 adım)    │    │  (Eyes)    │    │  Voxel      │   │
│   │             │    │              │    │            │    │  (Brain)    │   │
│   └─────────────┘    └──────────────┘    └────────────┘    └──────┬──────┘   │
│                                                                   │          │
│                                                                   ▼          │
│                                                        ┌────────────────┐    │
│                                                        │   Modül 5      │    │
│                                                        │   Sim-to-Real  │    │
│                                                        │   + Dashboard  │    │
│                                                        │   (Bridge)     │    │
│                                                        └────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **Sensör donanımı:** 8 adet ESP32 düğümü → C(8,2) = **28 benzersiz link** (WiFi 2.4 GHz CSI, 108 alt taşıyıcı, HT40 bant) + **UWB CIR** (32 tap).
- **Hedef bölge:** 50 m² (10 × 5 × 3 m) sınıf/depo ortamı, 6 bölmeli metal raf.
- **Çıktılar:** Voxel doluluk (μ + σ²), malzeme sınıfı (Metal/Plastik/Ahşap/Karton), spektral barkod ID (64-d L2-norm), hareket takibi (presence/class/velocity).

---

## Modül 1 — Veri Sentezi ve Fiziksel Gürültü Modelleme (Sionna 2.0)

Bu aşama, 50 m²'lik sınıf ortamını radyo frekansı (RF) gözüyle dijital bir ikize dönüştürme ve üzerine 2.4 GHz'in kaosunu ekleme sürecidir.

### Adım 1: Sahne ve Malzeme Tanımlama (Sionna RT)

Claude Code'a ilk görevimiz, odanın geometrisini ve malzemelerin dielektrik özelliklerini tanımlatmak olacak.

- **Sahne Kurulumu:** 50 m² sınıf; duvarlar, tavan ve taban için standart beton/alçıpan modelleri.
- **Nesne Tanımlama:** 6 bölmeli raf (metal yüzeyler) ve rafa yerleştirilecek farklı dielektrik sabitine ($\epsilon_r$) sahip nesneler (ahşap, plastik, metal).
- **BSDF (Bidirectional Scattering Distribution Function):** Duvarlar için **Lambertian** (dağınık yansıma), metal raf için **Specular** (aynasal yansıma) modelleri kullanılmalı.

### Adım 2: Diferansiyellenebilir Ray Tracing (D-RT)

Sionna 2.0'ın en büyük gücü olan gradyan tabanlı yol hesaplamasını devreye alıyoruz.

- **RadioMapSolver Entegrasyonu:** Büyük ölçekli veri üretimi için `RadioMapSolver` kullanılarak her koordinat için kanal tepkisi ($H$) üretilmeli.
- **Propagation Path:** Sinyalin sadece doğrudan (LoS) değil, yansıma ve kırınımlarını (NLoS) içeren yolların hesaplanması.
- **Diferansiyel Mantık:** Her simülasyon için iki tarama yapılmalı: bir adet **"boş oda" (background)** ve bir adet **"nesneli oda"**. Çıktı olarak bu ikisinin farkı olan $CSI_{\Delta}$ üretilmeli.

### Adım 3: Sentetik Rezonatör ID ($S(f)$) Enjeksiyonu

Ürettiğimiz CFR (Channel Frequency Response) verisine, ürün kimliğini belirleyen spektral barkodu matematiksel olarak işliyoruz.

**Rezonans Formülü** — Lorentzian-style, simülasyon çıktısındaki ilgili frekans binlerine uygulanır:

$$
S(f) = \prod_{i=1}^{N_{\text{res}}} \left[\, 1 - A_i \cdot \frac{1}{1 + Q_i^{2}\,(f/f_i - f_i/f)^{2}} \,\right]
$$

- **ID Kodlama (REVİZE — 2026-05-14):** Orijinal PDF 10-bit önermişti; bitirme MVP'si için **4-bit data + 3-bit Hamming(7,4) parity = 7-bit codeword** kullanılır. Spektrum **7 parçaya** bölünür, her parçanın sönümleme durumu **0 veya 1**.
  - **Unique ID sayısı:** $2^4 = 16$ (jüri demosu için 6 slot × 6 ürün için fazlasıyla yeterli).
  - **Hata düzeltme:** Hamming(7,4) tek-bit hata düzeltir.
  - **Çentik bandı:** 500 MHz / 7 ≈ **71 MHz**, gereken **kalite faktörü Q ≥ 91**.
  - **Rezonatör tipi:** **3D-printed karbon-katkılı PLA dielektrik blok** (Q ~100-150) ana tercih; basit pasif metal şerit (Q ~50) yedek plan, sınırdadır.
  - Üretim ölçeğine geçince hierarchical encoding (raf konumu + lokal ID) ile binlerce unique placement elde edilebilir.

### Adım 4: 2.4 GHz Spesifik Gürültü Modelleme (Augmentation)

Sionna'nın mükemmel dünyasını "kirletmek" için [`noise_models.py`](data_synthesis/noise_models.py) içindeki fonksiyonlar [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) scriptine entegre ediliyor — sadece Student (anchor) CIR'a uygulanır, Teacher (oracle) temiz kalır.

- **WiFi Interference:** Rastgele dar bantlı sinüzoid tonların (interferans) eklenmesi.
- **Bluetooth Bursts:** Ardışık alt taşıyıcılar üzerinde rastgele sönümlemeler (dropout) yapılması.
- **CFO (Carrier Frequency Offset):** Donanımsal faz kaymalarını simüle etmek için lineer faz rampalarının uygulanması.
- **AGC (Automatic Gain Control):** Link bazlı rastgele genlik normalizasyonu ekleyerek donanım kazancını modelleme.

### Adım 5: Veri Paketleme ve Çıktı Formatı

Eğitim için Claude Code'dan şu dosya yapısını üretmesini isteyeceğiz:

| Dosya | Şekil | Açıklama |
|---|---|---|
| `csivec_delta.npy` | `[N, 28, 108, 2]` | Fark CSI (WiFi 2.4 GHz, 28 link, 108 subcarrier, diferansiyel) |
| `uwb_cir_oracle.npy` | `[N, 6, 32, 2]` | Teacher — slot merkezli, gürültüsüz ideal CIR (KD hedefi) |
| `uwb_cir_anchor.npy` | `[N, 4, 32, 2]` | Student — 4 anchor'dan, gürültülü CIR (model girdisi) |
| `path_params_oracle.npy` | `[N, 6, 20, 4]` | Sionna ground-truth path (DML-AP teacher hedefi); kanal: real(a), imag(a), tau, validity |
| `path_params_anchor.npy` | `[N, 4, 20, 4]` | Anchor perspektifli ground-truth path (auxiliary supervision) |
| `slot_labels.npy` | `[N, 6]` | Slot doluluk (0/1) |
| `material_labels.npy` | `[N, 6]` | 0=boş, 1=metal, 2=plastik, 3=ahşap, 4=karton |
| `codeword_labels.npy` | `[N, 6, 7]` | 7-bit Hamming(7,4) codeword (eğitimde BCE hedefi) |
| `data_id_labels.npy` | `[N, 6]` | Ham 4-bit ID (0-15), validation/lookup için |

> Kral, ham verinin "yarak kürek" gürültüsünden arınıp pırlanta gibi bir özniteliğe (feature) dönüştüğü o kritik mutfağa, yani **Çok Kademeli Ön İşleme (Pre-processing) Pipeline** aşamasına giriyoruz. Bu aşama, modelin 2.4 GHz bandındaki WiFi ve UWB sinyallerini "duymasını" değil, **"anlamasını"** sağlar.

---

## Modül 2 — Çok Kademeli Ön İşleme (Pre-processing) Pipeline

Bu boru hattı, donanımsal bozulmaları (hardware impairments) ve ortam gürültüsünü eleyerek, modelin sadece "nesne imzasına" odaklanmasını sağlar.

### Adım 1: Ham Faz Sanitizasyonu (Phase Sanitization)

2.4 GHz bandında ESP32'den gelen faz verisi, donanımsal saat kaymaları (CFO) yüzünden tamamen "çöp" haldedir.

- **Phase Unwrapping:** $-\pi$ ile $+\pi$ arasında sıçrayan faz değerlerini sürekli bir çizgiye dönüştür.
- **Linear Phase Removal:** Taşıyıcı Frekans Kayması (CFO) kaynaklı lineer faz eğimini ($2\pi \Delta f\, t$) hesapla ve veriden çıkar.
- **Static Phase Offset Correction:** Linkler arası sabit faz farklarını (anten kalibrasyonu) `csivec_ref` (boş oda referansı) üzerinden normalize et.

### Adım 2: Genlik Yönetimi ve AGC Telafisi

Donanımın Otomatik Kazanç Kontrolü (AGC), sinyalin gerçek sönümleme (attenuation) bilgisini bozar.

- **AGC Normalization:** Her linkin (28 link) genliğini kendi ortalama enerjisine bölerek linkler arası güç dengesizliğini gider.
- **Outlier Clipping:** Sinyaldeki ani ve yapay sıçramaları (spikes) $\mu \pm 3\sigma$ (ortalama ve 3 standart sapma) kuralına göre tıraşla. Bu, gradyan patlamalarını önler.
- **Log-Scale Transformation:** Veriyi dB ölçeğine çekerek küçük dielektrik değişimlerini modelin daha kolay fark etmesini sağla.

### Adım 3: Diferansiyel Kanal Çıkarımı ($CSI_{\Delta}$)

Bu, hiyerarşinin **"Aether Core" ruhudur**; statik duvarları değil, sadece raftaki "yeni" nesneyi görmemizi sağlar.

- **Subtraction:** Mevcut ölçümden boş oda referansını çıkar:

$$ CSI_{\Delta} = CSI_{\text{Anlık}} - CSI_{\text{Ref}} $$

- **Baseline Correction:** Boş oda referansının zamanla kaymasını (drift) önlemek için periyodik olarak `csivec_ref` güncelleme mantığını kur.

### Adım 4: DWT (Ayrık Dalgacık Dönüşümü) Denoising

İşlem yükünü uçurmadan gürültüyü frekans-zaman uzayında temizliyoruz.

- **Decomposition:** Sinyali `db4` (Daubechies) ana dalgacığı ile **3 veya 4 seviyeli** parçalara ayır.
- **Soft-Thresholding:** Yüksek frekanslı (gürültü) katsayıları yumuşak eşikleme ile temizle, düşük frekanslı "nesne karakteristiklerini" koru.
- **Reconstruction:** Temizlenmiş sinyali geri inşa et veya katsayıları doğrudan modele öznitelik olarak ver.

### Adım 5: UWB CIR Sinc-Resampling ve Peak Alignment

UWB verisinin (CIR) her örnekte aynı zaman ekseninde olması hayati önem taşır.

- **Sinc-Resampling:** Ham CIR verisini **2 ns'lik** standart bir zaman grid'ine (sinc interpolation ile) oturt.
- **First Path Alignment:** Sinyalin ilk geldiği anı (Leading Edge) tespit et ve tüm CIR sinyalini bu tepe noktasına göre sola/sağa kaydırarak hizala.
- **Windowing:** Multipath etkilerini sınırlamak için ilk **64 ns'lik** pencereyi (32 tap) kırp ve odaklan.

### Adım 6: Tensor Formatting (Modele Hazırlık)

Veriyi [`csi_encoder.py`](models/encoders/csi_encoder.py) ve [`fused_model.py`](models/fused_model.py)'ın beklediği mükemmel matematiksel forma sokuyoruz.

- **Shape:** Veriyi `[Batch, 28, 2, 108]` (Link Sayısı, Real/Imag, Subcarrier Sayısı) formatına getir.
- **Normalizasyon:** Tüm tensörü $[-1, 1]$ aralığına sıkıştırarak `CSITransformerEncoder`'ın gradyan akışını optimize et.

---

## Modül 3 — Dual-Stream Encoder Mimarisi (The Eyes)

Bu mimari, her iki sensörün fiziksel üstünlüklerini birleştirerek tek bir sensörün asla göremeyeceği bir **"radyo vizyonu"** oluşturur.

### A. Stream 1: CSI Transformer Encoder (Mekânsal Gözlemci)

WiFi sinyali (CSI), odadaki nesnelerin hacmini ve konumunu anlamak için kullanılır.

- **CSILinkEncoder (Per-Link CNN):** 28 linkin her biri için 108 alt taşıyıcı (subcarrier) bağımsız olarak işlenir. 1D-CNN katmanları, frekans uzayındaki gürültüleri filtreleyip her link için **128 boyutlu** bir öznitelik vektörü üretir.
- **LinkGeometryEmbed:** Antenlerin $[x, y, z]$ koordinatları (`link_geo`) **64 boyutlu** bir vektöre gömülür. Sinyal verisiyle bu geometrik veri birleştirilerek modelin "hangi veri hangi linkten geliyor" sorusunu uzaysal olarak anlaması sağlanır.
- **CSI Transformer Block:** **4 katmanlı, 8 kafalı** (`n_heads=8`) bir Transformer yapısı kullanılır. Linkler arasındaki etkileşimi (self-attention) analiz ederek, bir nesnenin birden fazla linki aynı anda nasıl etkilediğini (spatial correlation) çözer.

  **Scale Dot-Product Attention:**

  $$
  \text{Attn}(Q, K, V) = \text{softmax}\!\left( \frac{QK^{T}}{\sqrt{d_k}} \right) V
  $$

  Burada $Q$ (Query), bir linkin durumunu; $K$ (Key) ve $V$ (Value) ise diğer linklerin o bölgedeki etkisini temsil eder.

### B. Stream 2: 3-Tier UWB Encoder (Fiziksel Parmak İzi)

UWB sinyali, nesnenin hem mutlak mesafesini hem de spektral kimliğini (ID) çıkarmak için kullanılır.

- **Tier 1 — Summary MLP:** Mesafe (`range`), sinyal gücü (RSSI) ve NLOS (görüş hattı dışı) bayrağı gibi **5 temel istatistiği 32 boyutlu** bir vektöre sıkıştırır.
- **Tier 2 — CIR Time-Domain Branch:** **32 örnekli (taps)** ham kanal tepkisi işlenir. Geniş bantlı yansımaları yakalamak için **7 birimlik geniş çekirdekli** (`kernel_size=7`) 1D-CNN katmanları kullanılır. Bu dal, nesnenin dielektrik "imzasını" korur.
- **Tier 3 — CFR Frequency-Domain Branch:** CIR'ın FFT'si alınmış hali olan **32 binlik spektral veri** işlenir. Ürün üzerindeki rezonatörün bıraktığı spektral barkod (çentikler) burada analiz edilir.
- **Global Fusion:** Bu üç katmandan gelen veriler ($32 + 64 + 64$) birleştirilip **128 boyutlu nihai bir UWB latent vektörüne** indirgenir.

### C. Cross-Stream Alignment (Füzyon ve Hizalama)

İki gözün (WiFi ve UWB) verilerini tek bir beyinde birleştiriyoruz.

- **UWB-CSI Cross-Attention:** UWB'den gelen net mesafe bilgisi **"Sorgu" (Query)** olarak kullanılır. WiFi linklerinden gelen veriler ise **"Anahtar" (Key)** ve **"Değer" (Value)** olarak modele beslenir. Bu, modelin arka plan gürültüsünü eleyip sadece nesnenin olduğu mesafe aralığındaki WiFi değişimlerine odaklanmasını sağlar.
- **Slot Embedding:** 6 bölmeli raf için her bölmeye (slot) özel öğrenilebilir sorgu vektörleri (`slot_emb`) tanımlanır. Bu vektörler, her iki sensörden gelen verileri o bölgeye özel olarak "harmanlar" (**FiLM-style MLP**).

---

## Modül 4 — Hiyerarşik Cascade DRI ve Voxel Tahmini (The Brain)

Bu aşama, encoder'lardan gelen ham özniteliklerin (latent features) anlam kazandığı, odanın 3 boyutlu haritasının çıkarıldığı ve nesnelerin kimliklendirildiği bölümdür.

### Adım 1: Sparse ROI ve Detection (UWB ToA Tabanlı)

Tüm odayı tarayıp işlem yükünü uçurmamak için sistem önce "Nerede bir şey var?" sorusuna yanıt arar.

- **Mekanizma:** UWB global latent vektörü kullanılarak 6 bölmeli (`n_slots=6`) raf için doluluk olasılıkları hesaplanır.
- **Formül:** `Detection_Prob = Sigmoid(Detection_Logits)`.
- **ROI Maskesi:** Eğer `Detection_Prob > Threshold` (Örn: 0.5) ise o slot **"Dolu" (ROI)** kabul edilir.
- **Fayda:** Sadece "Dolu" olan bölgelerde Recognition ve Identification işlemleri yapılır, bu da CPU/GPU yükünü **%80** azaltır.

### Adım 2: Bayesian TSDF Decoder (Hacimsel Yontma)

Odadaki nesneleri sadece bir nokta olarak değil, bir hacim (voksel) olarak hayal ediyoruz.

- **Olasılıksal Tahmin:** Her voksel için kesin bir "1" veya "0" yerine, bir Bayes çıkarımı yapılır.
- **Formül:**

$$
P(\text{3B\_Ortam} \mid \text{Veri}) = \frac{P(\text{Veri} \mid \text{3B\_Ortam}) \cdot P(\text{Prior})}{P(\text{Veri})}
$$

- **Varyans Hesabı:** Model her voksel için bir **Ortalama (μ — doluluk tahmini)** ve bir **Varyans (σ² — belirsizlik)** üretir.
- **Realist Etki:** Gürültülü sinyal gelen bölgelerde σ değeri artar, bu da dashboard'da o bölgenin "belirsiz/bulanık" görünmesini sağlayarak hatalı çıkarımları engeller.

### Adım 3: Recognition (CSI Tabanlı Malzeme Sınıflandırma)

ROI içinde kalan voksellerin malzeme türü tahmin edilir.

- **Girdi:** WiFi CSI Transformer'dan gelen ve slot embedding'leri ile harmanlanmış **256 boyutlu** latent vektörler.
- **Sınıflar:** Metal, Plastik, Ahşap, Karton (`n_mat=4`).
- **Kayıp Fonksiyonu:** Sadece ROI içindeki slotlar için **Cross-Entropy** kaybı hesaplanır.

### Adım 4: Identification (UWB CIR Tabanlı Spektral Barkod)

Sistemin en üst seviyesi; ürünün üzerindeki rezonatörü okuyarak ID belirleme.

- **Mekanizma:** Her slot için UWB CIR (Channel Impulse Response) verisi cross-attention ile taranır.
- **Çift Çıkış (REVİZE — 2026-05-14):**
  1. **64 boyutlu L2-normalize spektral imza** (latent) — `SpectralBarcodeDB` lookup için.
  2. **7-bit binary codeword tahmini** — Hamming(7,4) decode → 4-bit data → 16 ID arasından sınıflandırma.
- **Eşleştirme:** Latent vektör `SpectralBarcodeDB` ile **Cosine Similarity**, codeword tahmini Hamming decoder ile **tek-bit hata düzeltir**. İki yol birbirini doğrular.
- **Kayıp Fonksiyonu (Multi-task):**
  - **TripletMarginLoss** (latent için): aynı ürünün farklı açıdaki imzaları yakın, farklı ürünler uzak.
  - **BCE** (7-bit codeword için): her bit'in bağımsız doğru tahmini.

### Adım 5: Dynamic Tracking (Mikro-Doppler Analizi)

Ortamdaki hareketli nesnelerin takibi.

- **Girdi:** 1 saniyelik pencerede toplanan **T = 100** adet CSI ölçümü.
- **İşlem:** DC bileşen çıkarılır ve FFT (Fast Fourier Transform) ile **Doppler spektrogramı** oluşturulur.
- **Çıktı:**
  - **Presence:** Hareket var mı? (BCE Loss).
  - **Class:** İnsan mı, Forklift mi? (Cross-Entropy).
  - **Velocity:** $v_x, v_y, v_z$ hız vektörleri (MSE Loss).

---

## Modül 5 — Sim-to-Real Adaptasyonu ve Real-Time Dashboard (The Bridge)

Bu katman, modelin **"Sionna pre-trained"** halinden **"Gerçek Dünya Adapted"** haline geçişini sağlar ve elde edilen 3B veriyi anlık olarak görselleştirir.

### Adım 1: Adversarial Domain Adaptation (DANN)

Sionna verisi ile gerçek ESP32 verisi arasındaki istatistiksel farkı (domain gap) kapatmak için kullanılır.

- **Gradient Reversal Layer (GRL):** Eğitim sırasında encoder'ın çıkışına bir GRL katmanı eklenir. Bu katman, ileri geçişte (forward) etkisizdir ancak geri yayılımda (backward) **gradyanı ters çevirir**.
- **GRL Schedule Formülü:**

$$
\lambda = \frac{2}{1 + e^{-10 p}} - 1
$$

  (Burada $p$, eğitimin ilerleme oranını $[0,1]$ temsil eder.)
- **Domain Discriminator:** Modelin yanına küçük bir sınıflandırıcı eklenir. Görevi, gelen latent vektörün **Sionna'dan mı (Label: 0)** yoksa **gerçek ESP32'den mi (Label: 1)** geldiğini tahmin etmektir.
- **Mekanizma:** GRL sayesinde ana encoder, discriminator'ı "yanıltacak" (yani her iki veriyi de birbirine benzer şekilde temsil edecek) öznitelikler üretmeye zorlanır. Böylece model, simülasyona aşırı güvenmek (overfitting) yerine her iki dünyaya da uyumlu hale gelir.

### Adım 2: Residual Adapter ve İki Fazlı Fine-Tune

Gerçek dünya gürültüsünü (WiFi interferansı, nem, metal yansımaları) emmek için kullanılan esnek yapıdır.

- **Domain Adapter Katmanı:** Fine-tune sırasında encoder'lar dondurulur ve araya eklenen hafif bir adapter katmanı aktif edilir.
- **Adapter Formülü:**

$$
\text{Çıktı} = x + \text{scale} \cdot \text{adapter}(x)
$$

  Başlangıçta `scale` değeri **0.1** gibi küçük bir değerde tutularak modelin bilgisi korunur.
- **Faz 1 (Etiketsiz Ön Eğitim):** Gerçek odadan alınan ama etiketi (içeride ne olduğu) bilinmeyen verilerle, sadece **adversarial** ve **contrastive (UWB-CSI hizalama)** kayıpları kullanılarak encoder'lar eğitilir.
- **Faz 2 (Etiketli İnce Ayar):** Küçük bir etiketli gerçek veri setiyle sadece **malzeme (recognition)** ve **doluluk (detection)** başlıkları eğitilerek domain farkı absorbe edilir.

### Adım 3: Real-Time Veri İletişim Hattı (Python → Web)

Modelden çıkan 3B Tensör verisinin tarayıcıya (dashboard) en hızlı şekilde ulaştırılmasıdır.

- **FastAPI Backend:** Modelin çıkarım (inference) sonuçlarını saniyede en az **10–20 kez (10–20 Hz)** üretecek bir asenkron API katmanı kurulur.
- **WebSockets (UDP-style):** Geleneksel HTTP istekleri yerine, çift yönlü ve düşük gecikmeli **WebSockets tüneli** kullanılır.
- **Veri Paketi:** Her paket; doluluk (`detection_mask`), malzeme tipi (`materials`), ID (`barcode_sparse`) ve güven skoru (`uncertainty`) içeren bir **JSON veya Protobuf** dizisidir.

### Adım 4: Three.js Voxel Dashboard (Digital Twin)

Verinin son kullanıcıya 3 boyutlu bir "Dijital İkiz" olarak sunulmasıdır.

- **InstancedMesh Teknolojisi:** Tarayıcıda 1000'lerce voksali (küpü) ayrı ayrı çizmek yerine, tek bir çizim çağrısıyla (Draw Call) tüm grid'i GPU üzerinden render ederek hızı artırır.
- **Voxel Rendering:**
  - **Opacity (Şeffaflık):** Bayesian tahminindeki μ (doluluk) değerine göre vokseller belirir veya kaybolur.
  - **Coloring (Renklendirme):** Malzeme tahmini (Metal: Gümüş, Ahşap: Kahverengi vb.) ve ID bilgilerine göre vokseller boyanır.
  - **Uncertainty Overlay:** Varyans (σ) yüksekse, vokselin etrafında bulanık (glow) bir efekt oluşturularak kullanıcının o bölgedeki veriye şüpheyle bakması sağlanır.

### Adım 5: Gecikme (Latency) Öldürücü — Knowledge Distillation

Sistemi gerçek zamanlı ortamlarda (real-time) takılmadan koşturmak için modelin "damıtılmasıdır".

- **Teacher-Student Mimari:** Eğitim bittikten sonra, tüm bu ağır Transformer ve 3D CNN yapısının bilgisi, daha küçük ve hızlı bir **"Student" (Çırak)** modele aktarılır.
- **Fayda:** Dashboard üzerindeki gecikme **200 ms'nin altına** düşürülür, böylece forklift veya insan hareketleri ekranda **"anlık"** takip edilebilir.

---

## Sionna 2.0 Mimari ve Uygulama Datasheet (Aether Core Özel)

> Sionna 2.0'ın karmaşık dökümantasyonunu Aether Core için bir "kullanım kılavuzuna" çevirdim. Aşağıdaki döküm, hangi Sionna fonksiyonunun projedeki hangi fiziksel gerçeğe karşılık geldiğini adım adım açıklar.

Sionna 2.0, radyo dalgalarını ışık hüzmesi gibi takip eden (Ray Tracing) ve bu esnada malzemenin dielektrik özelliklerini hesaba katan bir motordur.

### 1. Sahne (Scene) ve Geometri Yapılandırması

Bu kısım, 50 m² depo/sınıf ortamının fiziksel kabuğunu tanımlar.

- **`sionna.rt.Scene`:** Odanın tamamını temsil eden ana objedir. Claude'a bu objeyi **10 m × 5 m × 3 m** boyutlarında kurdurmalısın.
- **`load_scene`:** `.obj` veya `.mitsuba` formatındaki depo modelini içeri aktarır. Rafın 6 bölmeli (slots) yapısı burada tanımlanır.
- **Malzeme (Material) Tanımlama:**
  - `concrete`: Duvarlar için kullanılır (Dielektrik sabiti $\epsilon_r \approx 5{-}7$).
  - `metal`: Raf yüzeyleri için **"Perfect Electrical Conductor" (PEC)** olarak tanımlanır.
  - **Önemli:** Rafın metal olması "Specular" (aynasal) yansımaları uçurur; Claude'a `scattering_model` olarak Lambertian yerine **mutlaka Specular** ekletmelisin.

### 2. Anten ve Link Kurulumu (WiFi & UWB)

Bu kısım, 8 adet ESP32 ve 28 linkin radyo parametrelerini belirler.

- **Transmitter / Receiver:** Her ESP32 düğümü hem alıcı hem verici olabilir. 8 düğüm toplam **28 benzersiz link (C(8,2))** oluşturur.
- **Frequency:** WiFi için **2.412 GHz** olarak set edilir.
- **Bandwidth:** **40 MHz (HT40)** genişliği tanımlanır.
- **Antenna:** Standart 2.4 GHz **monopol veya dipol** anten paterni seçilir.

### 3. Ray Tracing ve Kanal Çözümleme (H-Matrix)

En kritik adım burası; sinyalin nasıl yol alacağını belirler.

- **`compute_paths`:** Sinyalin duvarlardan ve raflardan kaç kez sekerek (rebound) alıcıya ulaşacağını hesaplar. 2.4 GHz için en az **3–5 yansıma (`max_depth=5`)** hesaplanmalıdır.
- **`RadioMapSolver`:** Odanın her noktasındaki kanal durumunu (CSI) hızlıca üretmek için kullanılan motordur.
- **`ChannelResponse (H)`:** Claude'a bu fonksiyondan **108 alt taşıyıcılı (subcarriers)** karmaşık sayı matrisini almasını söylemelisin.

### 4. Diferansiyel CSI ve Gürültü Katmanı

Sionna'nın "mükemmel" verisini gerçek dünya "çöplüğüne" dönüştürme aşaması.

| Sionna Çıktısı | Projedeki Karşılığı | Uygulanacak İşlem |
|---|---|---|
| `h_raw` | Saf Kanal Tepkisi | Boş oda referansıyla farkını al: $CSI_{\Delta}$. |
| `Phase` | Donanım Fazı | `add_interference_noise` ile CFO (faz kayması) ekle. |
| `Amplitude` | Sinyal Gücü | AGC simülasyonu için rastgele normalizasyon uygula. |
| `CIR` | UWB İmzası | $S(f)$ Lorentzian formülüyle rezonans çentikleri (notches) enjekte et. |

### 5. Veri Paketi (Dataset) Hazırlığı

Modelin (`FusedCSIUWBNet`) okuyacağı nihai format.

- **`CSI_Sequence`:** Zaman serisi eğitimi için (Dynamic Tracker) ardışık **100 ölçümlük (T = 100)** veri paketleri oluşturulur.
- **`Slot_Labels`:** Rafın hangi bölmesinin dolu olduğuna dair 0/1 etiketleri.
- **`Material_Labels`:** Nesnelerin ahşap mı metal mi olduğuna dair malzeme etiketleri.
- **`Barcode_Labels` (REVİZE):** Her dolu slot için **7-bit Hamming(7,4) codeword**; eğitimde BCE hedefi, decode sonrası 4-bit ID.

---

## Veri Formatları

| Tensör | Şekil | Kaynak | Tüketici |
|---|---|---|---|
| `csivec_delta.npy` | `[N, 28, 108, 2]` | [`generate_wifi_csi_sionna.py`](data_synthesis/generate_wifi_csi_sionna.py) | [`csi_encoder.py`](models/encoders/csi_encoder.py) |
| `uwb_cir_oracle.npy` | `[N, 6, 32, 2]` | [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) | KD teacher hedefi |
| `uwb_cir_anchor.npy` | `[N, 4, 32, 2]` | [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) | [`uwb_encoder.py`](models/encoders/uwb_encoder.py) (student input) |
| `path_params_oracle.npy` | `[N, 6, 20, 4]` | [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) | path_aux_head (DML-AP auxiliary loss) |
| `path_params_anchor.npy` | `[N, 4, 20, 4]` | [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) | path_aux_head (auxiliary supervision) |
| `slot_labels.npy` | `[N, 6]` | [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) | [`detection_head.py`](models/heads/detection_head.py) |
| `material_labels.npy` | `[N, 6]` | [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) | [`recognition_head.py`](models/heads/recognition_head.py) |
| `codeword_labels.npy` | `[N, 6, 7]` | [`generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) | [`identification_head.py`](models/heads/identification_head.py) |
| Pre-process çıktısı | `[B, 28, 2, 108]` | [`tensor_format.py`](preprocessing/tensor_format.py) | [`fused_model.py`](models/fused_model.py) |
| WS paketi | `{detection_mask, materials, barcode_sparse, uncertainty}` | [`ws_stream.py`](backend/ws_stream.py) | [`voxel_renderer.js`](dashboard/voxel_renderer.js) |

---

## Dinamiklik (Stochasticity) Stratejisi

### 1. Neleri "Değişken" (Dynamic) Yapacağız?

Kodun her `for` döngüsünde şu değerleri rastgele (stochastic) atamasını isteyeceğiz:

- **Konumsal Değişkenlik:** Ürünü rafın içine her seferinde milimetrik olarak farklı yerlere, farklı açılarla (rotation) koyacağız.
- **Malzeme Değişkenliği:** Dielektrik sabiti ($\epsilon_r$) için sabit bir değer yerine, örneğin "Ahşap" için $2.0$ ile $3.0$ arasında rastgele değerler atayacağız.
- **Gürültü Değişkenliği (Interference):** WiFi gürültüsünün şiddeti, frekansı ve süresi her örnekte farklı olacak.
- **Faz Kayması (CFO):** ESP32'nin o anki ısısına göre değişen faz sapmalarını her link için bağımsız rastgele değerlerle ($\pm \pi$ aralığında) simüle edeceğiz.

### 2. Model Bunu Nasıl "Anlar"?

Model aslında verinin "değişken" olduğunu **anlamaz**; model **"değişmezleri" (invariants)** bulmaya çalışır.

- **Örnek:** Sen 1000 farklı gürültü ve 1000 farklı konumda metal bir kutuyu modele gösterdiğinde, model şunu fark eder: *"Gürültü değişiyor, konum değişiyor ama sinyaldeki şu spesifik sönümlenme karakteri hep orada kalıyor."*
- **Sonuç:** Model gürültüyü "yok saymayı" ve sadece nesnenin gerçek **radyo imzasını (signature)** yakalamayı bu değişkenlik sayesinde öğrenir.

---

## Tasarım Kararları (PDF üzerine yapılan revizyonlar)

PDF orijinal şartnameyi bitirme MVP'sine uyarlarken alınan kararlar. Her karar için **tarih**, **gerekçe** ve **etki** kayıt edilir.

### 2026-05-14 — ID barkod uzunluğu: 10-bit → 4-bit + Hamming(7,4) = 7-bit

**Karar:** Spektral barkod 10 bit yerine **4-bit data + 3-bit Hamming(7,4) parity = 7-bit codeword** olarak kodlanır.

**Gerekçe:**
- 10-bit için gereken Q ≥ 130 → pasif rezonatör için fiziksel olarak zor.
- 4-bit data 16 unique ID verir, jüri demosu (6 slot × ~6 ürün) için yeterli.
- Hamming(7,4) tek-bit hata düzeltir, demo zenginliği sağlar.

**Etki:**
- `labels.npy` şekli `[N, 6, 10]` → `[N, 6, 7]`.
- `data_synthesis/resonator_inject.py` 7 çentik üretir (Hamming codeword).
- `models/heads/identification_head.py` çift çıkışlı: 64-d latent + 7-bit binary head (multi-task).
- Identification kaybı: `TripletMargin (latent) + BCE (7-bit)`.

### 2026-05-14 — Rezonatör tipi: 3D-printed dielektrik blok (ana), pasif metal şerit (yedek)

**Karar:** Ürünlere **karbon-katkılı PLA filament ile 3D basılmış dielektrik blok** (Q ~100-150) yapıştırılır. Pasif metal şerit (Q ~50) yedek seçenek olarak kalır.

**Gerekçe:**
- 7-bit codeword için 71 MHz çentik bandı → Q ≥ 91 gerek.
- Metal şerit Q ~50 ile sınırda, çentikler birbirine girer.
- 3D-printed dielektrik blok: ucuz, tekrarlanabilir, üretim toleransı kabul edilebilir.
- SRR (PCB) alternatif: daha yüksek Q ama maliyet ve üretim karmaşıklığı yüksek; ölçek aşamasında düşünülecek.

**Etki:**
- `configs/resonator_db.yaml` rezonatör tipi alanı eklendi.
- Sionna sahne kurulumunda her ürünün üstünde `epsilon_r ~ 4.0-6.0` (karbon-katkılı PLA dielektrik) bloklar tanımlanır.
- Jüri sunumunda fiziksel rezonatör tasarımı slaydı zorunlu — örnek 3D model + Q ölçümü gösterilmelidir.

### 2026-05-15 — Hibrit Topoloji + 4 Real-Time Eksik Kapatma

**Karar paketi (4 ayrı revizyon):**

#### A) Hibrit topoloji — 8 fiziksel düğüm
- **4 master:** ESP32 + UWB yan yana (M1..M4)
- **4 satellite:** sadece ESP32 (S1..S4)
- WiFi: 8 ESP32 → **C(8,2)=28 link** (mevcut)
- UWB: 4 master → **C(4,2)=6 bistatic link** (yeni)
- TDMA cycle: 40 ms (25 Hz), her master 10 ms TX

**Tensor revizyon:** `uwb_cir_anchor.npy [N, 4, 32, 2] → [N, 6, 32, 2]` ve `path_params_anchor.npy [N, 4, 20, 4] → [N, 6, 20, 4]`.

**Why:** Önceki "4 anchor + 1 TX" varsayımı 5 UWB cihaz gerektiriyordu — donanım envanteri 4. Round-robin TDMA ile her cihaz hem TX hem RX, fiziksel envanterle %100 uyumlu.

#### B) Sensör Senkronizasyonu (Timestamp Alignment)
- **YENİ:** [`backend/sync_buffer.py`](backend/sync_buffer.py) — `MultiSensorSyncBuffer` (28 WiFi + 6 UWB ring buffer + ±5ms tolerance window)
- Master tick: 50 ms (20 Hz), eksik link varsa tick atlanır
- Sionna sentetik veriye etkisi YOK (anlık snapshot zaten ideal)

**Why:** Gerçek ESP32 + DW1000 farklı saatlerde paket atar; Cross-Attention senkronize tensör bekler.

#### C) Adaptive Baseline (Drift Düzeltmesi)
- [`preprocessing/differential_csi.py`](preprocessing/differential_csi.py) — `AdaptiveBaseline` class (B + A combo)
- Strategy A: sürekli yumuşak EMA (α=1e-4), drift bekçisi
- Strategy B: motion-gated agresif EMA (α=1e-2), quiet_frames=100 sonra aktif
- Disk persistance: server reboot sonrası reference korunur

**Why:** Sabit `csivec_ref` gerçek dünyada sıcaklık/nem ile drift, model her şeyi nesne sanar.

#### D) Sliding Window Tracker (Real-Time)
- [`models/heads/tracker.py`](models/heads/tracker.py) — `DopplerRingBuffer` + `SlidingTracker` + `TrackingEngine`
- T=100 örnekli ring buffer, her 50 ms inference tick
- COLD START: ilk 100 örnek "WARMING_UP", sonra her tick tahmin
- NET LATENCY: 50 ms (PDF 200 ms hedefi içinde rahat)

**Why:** PDF "T=100 (1 sn)" ile "<200 ms gecikme" çelişkisini sliding window çözer.

#### Etki gören dosyalar:
- `configs/scene.yaml` + 4 preset (5 dosya, hibrit nodes bloğu)
- `data_synthesis/generate_hybrid_sionna.py` (Round-Robin TDMA, 6 bistatic)
- `data_synthesis/generate_wifi_csi_sionna.py` (nodes'tan ESP32 oku)
- `models/encoders/uwb_encoder.py` (docstring 4→6)
- `preprocessing/differential_csi.py` (AdaptiveBaseline gerçek kod)
- `models/heads/tracker.py` (SlidingTracker + RingBuffer + Engine)
- `backend/sync_buffer.py` (YENİ dosya)
- `backend/ws_stream.py`, `backend/inference_server.py` (entegrasyon notları)
- `configs/training.yaml` (runtime bloğu)
- `COLAB_RUN.txt` (assert'ler 4→6)

---

### 2026-05-15 — Path-aware auxiliary supervision (DML-AP teacher signal) — UWB + CSI

**Karar:** Hem UWB hem WiFi CSI tarafında Sionna RT'den gelen ground-truth path parametrelerini (`a` kompleks amplitude + `tau` delay) kaydet. Her iki encoder'a da auxiliary head ekle, raw CIR/CFR'dan path parametrelerini tahmin etmeyi öğrensin.

**Why:**
- **CSI tarafı sim2real gap'in en kritik noktası.** Gerçek ESP32'den ham CSI gelir, path-by-path bilgi YOK; Fourier zaman çözünürlüğü 25 ns ≈ 7.5 m — slot ayrımı için kabul edilemez.
- Klasik DML-AP super-resolution **sub-Fourier 30-50 cm** seviyesine indirir (slot spacing 50 cm — TAM HEDEF), ama **iteratif AP per-sample 100-500 ms**, real-time inference (10-20 Hz) için imkansız.
- Çözüm: sentetik tarafta Sionna ground-truth path bedava → CSI Transformer Encoder'a auxiliary loss olarak supervise et → model **raw CFR'dan path-like temsil** öğrenir → gerçek dünyada **DML-AP koşturmaya gerek kalmaz**.
- UWB tarafında da aynı mantık (rezonatör Q analizine bonus path-aware feature).

**Loss tasarımı:**
```
L_total = L_main_tasks + λ_aux · MSE(predicted_paths, gt_paths)
```
- `L_main_tasks`: detection + recognition + identification + tracker
- `λ_aux`: 0.1-0.5 (eğitim sırasında ramp up, inference'ta path_aux_head devre dışı)
- Path tensörünün validity_mask kanalı sayesinde **geçersiz padding path'lerine loss uygulanmaz**

**Şema (her tensör):**
- En güçlü 20 path (amplitude'a göre) seçilir, padding ile sabit shape
- 4 kanal: `real(a)`, `imag(a)`, `tau (saniye)`, `validity_mask (0/1)`

**Storage:**
- UWB: `path_params_oracle.npy` `[N, 6, 20, 4]` (~19 MB / 10k) + `path_params_anchor.npy` `[N, 4, 20, 4]` (~13 MB)
- CSI: `path_params_csi.npy` `[N, 28, 20, 4]` (~90 MB / 10k) — full pass'in path'leri (boş oda referansı path'leri kayıt edilmez)
- Toplam ek: ~120 MB / 10k sample → **5 preset × 100k = ~6 GB** (Drive 1 TB içinde rahat)

**How to apply:**
- `data_synthesis/generate_hybrid_sionna.py` → `compute_cir_for_current_rxs` artık tuple döndürür ✓
- `data_synthesis/generate_wifi_csi_sionna.py` → `compute_all_links_cfr_batched` artık tuple döndürür ✓
- `build_one_*` dict'lerine `path_params_*` alanları eklendi ✓
- `models/encoders/uwb_encoder.py` → ek `UWBPathAuxHead` (yazılacak)
- `models/encoders/csi_encoder.py` → ek `CSIPathAuxHead` (yazılacak)
- Inference'ta path_aux_head atlanır, sadece encoder forward → gecikme < 200 ms PDF hedefi korunur

**Real-time savunması:**
- Inference path: `Raw CFR/CIR → Encoder → Heads → Output` (DML-AP YOK)
- Eğitim path: `Raw CFR/CIR → Encoder → Heads + PathAuxHead → Loss`
- Real-time'da PathAuxHead detached/silinmiş → Student model bilgiyi enkoder'a "soğurmuş" olarak kalır

**Etki gören dosyalar:** `data_synthesis/generate_hybrid_sionna.py`, `data_synthesis/generate_wifi_csi_sionna.py`, `COLAB_RUN.txt` (H6 + H9c + H10 + H19 doğrulamaları).

---

### 2026-05-15 — Multi-Domain Training: 5 sahne preset + per-sample bağımsız RNG

**Karar:** Domain Randomization + Multi-Environment Training hibrit. 5 farklı sahne preset, her birinde aynı pipeline (UWB Teacher-Student + WiFi diferansiyel CSI), bağımsız master_seed ile.

| Preset | Boyut (m) | master_seed | Tipoloji |
|---|---|---|---|
| classroom_default | 10×5×3 | 42 | mevcut baseline |
| warehouse_large | 15×10×5 | 142 | büyük depo, uzun mesafe |
| office_small | 6×4×2.5 | 242 | küçük ofis, alçıpan duvar |
| lab_medium | 8×6×3 | 342 | metal-yoğun lab |
| room_low_ceiling | 10×5×2.2 | 442 | düşük tavan |

**Üretim akışı:** Her preset için ayrı XML ([scenes/aether_<preset>.xml](scenes/)) + scene config ([configs/scene_<preset>.yaml](configs/)) + COLAB_RUN.txt H14-H18 hücresi. Çıktı: `/MyDrive/aether_core/data/<preset_name>/` her preset ayrı klasör. Toplam 5 × 100k = 500k veri için ~45 saat compute (4-5 Colab session).

**Critical bugfix — Per-sample bağımsız RNG:**

Önceki üretimde tek `rng` objesi sample'lar arasında state biriktiriyordu; UWB ve WiFi script'lerinde RNG çağrı sayıları farklı olduğu için Sample 0 paired ✓, Sample 1+ drift → slot/material etiketleri eşleşmiyordu.

**Fix:** `master_rng = np.random.default_rng(args.seed)` + `sample_seeds = master_rng.integers(...)`. Her sample için bağımsız `sample_rng = np.random.default_rng(sample_seeds[i])`. UWB ve WiFi aynı master seed → garantili paired.

**Why:** 5 preset × 100k = 500k sample'da drift bug 5 farklı yerde patlardı; şimdi her preset'in kendi master_seed'i var, içinde per-sample bağımsız RNG. **2026-05-15 T1-T9 doğrulama testleri tamamı geçti** (detay: `memory/project_validation_report.md`).

**Etki gören dosyalar:** `data_synthesis/generate_hybrid_sionna.py`, `data_synthesis/generate_wifi_csi_sionna.py`, `configs/scene*.yaml` (5 dosya), `scenes/aether_*.xml` (5 dosya), `COLAB_RUN.txt`.

---

### 2026-05-14 — UWB anchor mimarisi: Teacher-Student Knowledge Distillation + Sensing-Optimized Asimetrik geometri

**Karar:** PDF'in `[N, 6, 32, 2]` slot-perspektifli tensörü **Teacher (oracle) eğitim hedefi** olarak korunur; gerçek donanım `[N, 4, 32, 2]` anchor-perspektifli olarak **Student input** verir. Model, Student CIR'dan Teacher CIR'ı tahmin etmeyi öğrenir (KD).

**Anchor yerleşimi — "Sensing-Optimized Asimetrik":**

| Anchor | Konum (m) | Rolü |
|---|---|---|
| A1 — `ceiling_near` | (4.6, 2.5, 2.95) | Tavan merkez 1, raf üstünde |
| A2 — `ceiling_far` | (5.4, 2.5, 2.95) | Tavan merkez 2, yanal kayma |
| A3 — `corner_diagonal` | (0.5, 0.5, 2.7) | Çapraz köşe, uzak referans |
| A4 — `shelf_top` | (5.0, 2.5, 2.30) | Raf üstü, rezonatör SNR için yakın |

Tüm konfigürasyon [`configs/scene.yaml`](configs/scene.yaml) içinde.

**Why:**
- 4 anchor uzayda dağıtılınca her slotun benzersiz mesafe parmak izi oluşur (multistatic radar mantığı).
- Teacher-Student paired dataset modelin "ham 4 anchor CIR'ından 6 slot bilgisi"ni öğrenmesini sağlar — distillation hedef tensörü ground-truth gibi davranır.
- A4 raf üstü ≤ 1 m → pasif rezonatör SNR'ını gürültü tabanının üstüne çıkarır.

**How to apply:**
- [`data_synthesis/generate_hybrid_sionna.py`](data_synthesis/generate_hybrid_sionna.py) iki path-solve yapar: oracle (6 RX slot merkezleri) + anchor (4 RX). Aynı sahne snapshot'unda eş zamanlı.
- Teacher CIR gürültüsüz ideal; Student CIR'a path-loss + AGC + CFO + WiFi/BT + AWGN uygulanır.
- Kayıp: `L = L_task(student_pred) + λ · KD(student_features, teacher_features)`. KD ağırlığı eğitim sırasında ayarlanır.
- Mitsuba sahne dosyası: [`scenes/aether_classroom.xml`](scenes/aether_classroom.xml).

---

## Çalışma İlkeleri (Claude Code'a Notlar)

- **Parça Parça İlerle:** Tüm dökümanı tek seferde kusup "hadi yap" demek yerine, önce **Sionna Veri Sentezi (Modül 1)** ile başla. O modülün çıktısını (`.npy` dosyalarını) kontrol etmeden 2. adıma geçme.
- **Fiziği Hatırlat:** Claude bazen saf yazılımcı gibi düşünüp *"Gürültüyü boşver, ben modeli büyütürüm"* diyebilir. Ona her zaman *"Biz radyo fiziğiyle (CSI/UWB) uğraşıyoruz, gürültü modelin içinde değil, **verinin üretiminde** çözülmeli"* diye ayarı ver.
- **Dinamikliği Sorgula:** Kod üretildiğinde bak bakalım; rastgelelik (`random.uniform`, `np.random` vb.) gerçekten her döngüde çalışıyor mu? **Statik bir sahne üretip geçmesine izin verme.**

---

## Klasör Yapısı

```
PROJESON/
├── README.md                       # Bu dosya — PDF'in tam içeriği
├── requirements.txt                # Python bağımlılıkları
├── .gitignore
├── configs/
│   ├── scene.yaml                          # PRESET: classroom_default (baseline)
│   ├── scene_warehouse_large.yaml          # PRESET: warehouse_large
│   ├── scene_office_small.yaml             # PRESET: office_small
│   ├── scene_lab_medium.yaml               # PRESET: lab_medium
│   ├── scene_room_low_ceiling.yaml         # PRESET: room_low_ceiling
│   ├── antenna.yaml                # 8 ESP32, 28 link, 2.412 GHz, HT40
│   ├── training.yaml               # batch / lr / GRL λ schedule / Phase 1-2
│   └── resonator_db.yaml           # SpectralBarcodeDB (4-bit ID + Hamming = 7-bit codeword)
│
├── data_synthesis/                 # ── Modül 1: Sionna 2.0 Veri Sentezi
│   ├── __init__.py
│   ├── scene_builder.py            # sionna.rt.Scene + BSDF (Lambertian/Specular)
│   ├── ray_tracing.py              # compute_paths (max_depth=5), RadioMapSolver
│   ├── resonator_inject.py         # S(f) Lorentzian formülü, 7-bit Hamming codeword
│   ├── noise_models.py             # path-loss + AGC + CFO + WiFi/BT + AWGN
│   ├── channel_dataset.py          # Hibrit kanal veri seti sınıfı (PyTorch)
│   ├── generate_hybrid.py          # legacy stub (yerini generate_hybrid_sionna aldı)
│   ├── generate_hybrid_sionna.py   # UWB ENTRY POINT — Teacher-Student paired CIR
│   └── generate_wifi_csi_sionna.py # WiFi ENTRY POINT — 28 link diferansiyel CSI
│
├── scenes/                         # Mitsuba 3 sahne dosyaları (5 preset)
│   ├── aether_classroom.xml        # 10×5×3 m, mevcut baseline
│   ├── aether_warehouse_large.xml  # 15×10×5 m, büyük depo
│   ├── aether_office_small.xml     # 6×4×2.5 m, küçük ofis (alçıpan)
│   ├── aether_lab_medium.xml       # 8×6×3 m, lab
│   └── aether_room_low_ceiling.xml # 10×5×2.2 m, düşük tavan
│
├── preprocessing/                  # ── Modül 2: 6-adımlı Pre-processing
│   ├── __init__.py
│   ├── phase_sanitize.py           # Adım 1: unwrapping, lineer faz, statik offset
│   ├── agc_normalize.py            # Adım 2: AGC, μ±3σ clipping, log-scale (dB)
│   ├── differential_csi.py         # Adım 3: CSI_Δ + baseline drift
│   ├── dwt_denoise.py              # Adım 4: db4, soft-threshold
│   ├── uwb_resample.py             # Adım 5: Sinc-resample, first-path align
│   └── tensor_format.py            # Adım 6: [B, 28, 2, 108], [-1, 1] norm
│
├── models/                         # ── Modül 3 + 4
│   ├── __init__.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── csi_encoder.py          # CSILinkEncoder + LinkGeometryEmbed + 4×8 Transformer
│   │   ├── uwb_encoder.py          # 3-Tier (32 + 64 + 64 → 128)
│   │   └── cross_attention.py      # UWB→CSI cross-attn + 6 slot embedding (FiLM)
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── detection_head.py       # Sparse ROI sigmoid, n_slots=6
│   │   ├── tsdf_decoder.py         # Bayesian TSDF (μ + σ²)
│   │   ├── recognition_head.py     # 4 sınıf, CrossEntropy (ROI-only)
│   │   ├── identification_head.py  # 64-d L2-norm barkod, TripletMarginLoss
│   │   └── tracker.py              # Mikro-Doppler (T=100, BCE/CE/MSE)
│   └── fused_model.py              # FusedCSIUWBNet — pipeline'ın birleşim noktası
│
├── adaptation/                     # ── Modül 5: Sim-to-Real
│   ├── __init__.py
│   ├── dann.py                     # GRL + Domain Discriminator (λ schedule)
│   ├── residual_adapter.py         # x + scale·adapter(x); Phase 1 / Phase 2
│   └── distillation.py             # Teacher-Student (gecikme < 200 ms)
│
├── backend/                        # ── FastAPI + WebSocket
│   ├── __init__.py
│   ├── main.py                     # FastAPI app + router
│   ├── inference_server.py         # 10-20 Hz predict loop
│   └── ws_stream.py                # WS paketi: detection_mask/materials/barcode/sigma
│
├── dashboard/                      # ── Three.js Voxel Digital Twin
│   ├── index.html                  # Three.js + canvas
│   ├── main.js                     # Scene/camera/orbit + WS client
│   └── voxel_renderer.js           # InstancedMesh, opacity=μ, color=material/ID
│
├── data/                           # .npy dosyaları (gitignore)
├── checkpoints/                    # Eğitilmiş ağırlıklar (gitignore)
└── notebooks/                      # Keşif/görselleştirme defterleri
```

---

## Kurulum

```powershell
# Python 3.10+ önerilir, GPU (CUDA 11.8+) Sionna RT için şart.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Not:** Sionna 2.0, TensorFlow + CUDA bağımlılıklarıyla gelir; gerçek RT simülasyonları için bir NVIDIA GPU şart koşulur. CPU üzerinde sadece pre-processing ve model çıkarımı denenebilir.

---

## Hızlı Başlangıç (Ekip Arkadaşı için)

Repo'yu yeni klonladıysan, kodu uçtan uca görmenin en hızlı yolu:

### 1) Bağımlılıkları kur
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**Not:** Sionna + TensorFlow CUDA istiyor; sadece kodu görmek/dashboard çalıştırmak için **PyTorch + FastAPI** yeterli (Sionna kurulumunu atlayabilirsin). CPU üzerinde her şey çalışıyor; Sionna'sız sadece **yeni veri üretemezsin**.

### 2) Dashboard'u aç (model olmadan, mock mode)
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Sonra tarayıcıda: <http://127.0.0.1:8000/dashboard/> (Three.js voxel sahnesi, sahte verilerle 20 Hz tick).

WebSocket health: <http://127.0.0.1:8000/telemetry>

### 3) Mevcut mini veri üzerinde ablation testi koştur (CPU'da ~13 dk)
**Önkoşul:** `data/<preset>/` klasörlerinde 5×1000 sample. Yoksa Colab'da [COLAB_RUN.txt](COLAB_RUN.txt) hücrelerini koştur (Drive mount + Sionna üretim, ~30 dk L4 GPU).

```powershell
python -m training.ablation --epochs 2 --batch_size 16 --device cpu
```
Çıktı: `training/ablation_results.json` (5 varyant × det F1/rec_acc/id_bit/süre).

### 4) Tek modül smoke test
```powershell
# Preprocessing
python -c "from data_synthesis.multi_domain_dataset import MultiDomainDataset; print(len(MultiDomainDataset()))"

# Forward pass
python -c "import torch; from models.fused_model import FusedCSIUWBNet; m = FusedCSIUWBNet(); print(sum(p.numel() for p in m.parameters()), 'param')"
```

### Klasör yapısı
| Klasör | İçerik |
|---|---|
| [data_synthesis/](data_synthesis/) | Sionna RT veri üretimi + layout variant + multi-domain dataset |
| [preprocessing/](preprocessing/) | 6 adımlı pipeline (phase sanitize → AGC → diff CSI → DWT → UWB resample → tensor format) |
| [models/encoders/](models/encoders/) | CSI Transformer + UWB 3-Tier + Cross-Attention (DualStreamEncoder 2.48M) |
| [models/heads/](models/heads/) | Detection + Bayesian TSDF + Recognition + Identification + SlidingTracker |
| [models/fused_model.py](models/fused_model.py) | FusedCSIUWBNet (2.91M) + compute_loss multi-task |
| [adaptation/](adaptation/) | DANN GRL + ResidualAdapter + StudentFusedNet KD (605k) |
| [backend/](backend/) | FastAPI + WS + InferenceEngine async 50 ms master tick |
| [dashboard/](dashboard/) | Three.js voxel + HUD + WS client (CDN, build adımı yok) |
| [training/](training/) | train_baseline + train_dann_phase + evaluate + 5 varyantlı ablation |
| [configs/](configs/) | 5 preset scene.yaml + training.yaml |
| [scenes/](scenes/) | Mitsuba 3 XML (5 preset, runtime'da Sionna RT'ye yüklenir) |
| [upcoming/](upcoming/) | Modül planları + risk envanteri (FAZ 1-7 ✅ + Risk listesi) |
| [COLAB_RUN.txt](COLAB_RUN.txt) | Colab'da Drive mount → veri üretimi step-by-step |

### Git'te olmayanlar
- `data/<preset>/*.npy` (mini veri ~191 MB, Colab'da üretilir)
- `checkpoints/*.pt` (henüz production eğitim yapılmadı)
- `.venv/`, `__pycache__/`

---

## Geçerli Durum (2026-05-19)

```
[●●●●●●●●●●]  93% — Tüm modüller + FAZ 0.5 sensor placement hazır, production veri kaldı

✅ FAZ 0    Veri Sentezi      data_synthesis/ (UWB + WiFi + path_params + 5 preset)
✅ FAZ 1    Preprocessing     preprocessing/ (6 dosya + wrapper, 8.65 ms/sample)
✅ FAZ 2    Encoders          models/encoders/ (DualStreamEncoder 2.48M, 21 ms/sample)
✅ FAZ 3    Brain             models/heads/ + fused_model.py (FusedCSIUWBNet 2.91M, 18 ms/sample, multi-task loss OK)
✅ FAZ 4    Adaptation        adaptation/ (DANN GRL + ResidualAdapter + StudentFusedNet KD, 605k param 12 ms/sample)
✅ FAZ 5    Backend+Dashboard backend/main.py (FastAPI + WS) + dashboard/* (Three.js voxel InstancedMesh, real-time UI)
✅ FAZ 6    Training+Ablation training/ (MultiDomainDataset + train_baseline + train_dann_phase + eval + 5 ablation varyantı). MEVCUT 5K VERİYLE MİNİ ABLATION ✅ (2 epoch × 5 varyant, det_f1 0.70-0.76, ~13 dk toplam CPU).
✅ FAZ 0.5  Sensor Placement  data_synthesis/layout_variants.py + UWB/WiFi script entegrasyon + MultiDomainDataset layout-aware. Birim test geçti (s0 vs s500 link_geo farkı ~35 cm). 5k eski veri backward-compat ✓.
○  PROD    Tam ölçek eğitim   5 preset × 5 variant × 20k = 500k Colab GPU, 30-50 epoch, gerçek sim2real metrik
```

Detaylı durum + risk envanteri: [upcoming/](upcoming/) klasörü (8 TXT).
Tasarım kararları + revizyon kaydı: aşağıdaki "Tasarım Kararları" bölümü.

---

## Geliştirme Yol Haritası

> **Kural:** Her modülün çıktısı doğrulanmadan bir sonrakine geçme.

| Sıra | Modül | Tamamlanma Kriteri |
|---|---|---|
| 1a | **UWB Veri Sentezi** ([data_synthesis/generate_hybrid_sionna.py](data_synthesis/generate_hybrid_sionna.py)) | Teacher-Student paired UWB CIR + path_params: `uwb_cir_oracle.npy` `[N,6,32,2]` + `uwb_cir_anchor.npy` `[N,6,32,2]` + path_params. **✅ Hibrit topoloji + Round-Robin TDMA 2026-05-15.** |
| 1b | **WiFi CSI Sentezi** ([data_synthesis/generate_wifi_csi_sionna.py](data_synthesis/generate_wifi_csi_sionna.py)) | 28 link diferansiyel CSI + path_params_csi: `csivec_delta.npy` `[N,28,108,2]`. Batched (tek PathSolver/pass), 8 ESP32 (4 master + 4 satellite). **✅ 2026-05-15.** |
| 1c | **Multi-Domain Üretim** (5 preset, COLAB_RUN H14-H18) | 5 preset × 1k mini = **5k baseline (B stratejisi)** ✅ doğrulandı 2026-05-15. Production: 5 × 10k veya 5 × 100k. Her preset paired check True ✓. |
| 1d | **Mini Veri Lokal** | 5 preset × 12 dosya = **60 dosya, 191.4 MB**, paired check 5/5 True ✅ 2026-05-15. |
| 2 | **Pre-processing** ([preprocessing/](preprocessing/)) | **✅ 2026-05-15 TAMAMLANDI.** 6 stub dosya → gerçek kod (phase_sanitize, agc_normalize, differential_csi AdaptiveBaseline, dwt_denoise, uwb_resample, tensor_format) + `preprocess_csi/uwb/path_params` wrapper. Birim test: gerçek 5k veri ile `8.46 ms/sample` CSI + `0.18 ms/sample` UWB, output `[-1, 1]` per-link norm. |
| 3 | **Encoders** ([models/encoders/](models/encoders/)) | **✅ 2026-05-15 TAMAMLANDI.** DualStreamEncoder = CSITransformerEncoder (4×8 layer, 192→256) + UWBEncoder (3-Tier: Summary+CIR+CFR → 128) + UWBCSICrossAttention (slot embedding + FiLM) + 3 aux head (CSI path, UWB path, oracle proj). **2.48M parameter, 21 ms/sample CPU inference, NaN/Inf yok.** |
| 4 | **Brain** ([models/heads/](models/heads/) + [models/fused_model.py](models/fused_model.py)) | **✅ 2026-05-15 TAMAMLANDI.** Detection + Bayesian TSDF (μ+σ²) + Recognition (masked CE) + Identification (triplet+BCE, 64-d L2 + 7-bit Hamming) + SlidingTracker (2D-CNN Doppler) + FusedCSIUWBNet orchestrator + compute_loss multi-task. **2.91M parameter, 18 ms/sample CPU inference, multi-task loss OK (det 0.68, rec 1.28, id 1.33, tsdf 0.13).** |
| 5a | **Adaptation** ([adaptation/](adaptation/)) | **✅ 2026-05-15 TAMAMLANDI.** DANN (GRL + DomainDiscriminator + λ schedule warmup 3 epoch + cap 0.7) + ResidualAdapter (zero-init identity + Phase 1/2 freeze) + StudentFusedNet (605k param, 1.5x hız) + KD loss (repr + logit + hard, T=4). GRL gradient = -λ·grad doğrulandı. |
| 5b | **Backend+Dashboard** ([backend/](backend/) + [dashboard/](dashboard/)) | **✅ 2026-05-15 TAMAMLANDI.** FastAPI lifespan + 4 HTTP endpoint (/, /telemetry, /reset_baseline) + 2 WS endpoint (/ws/ingest, /ws/predict) + `InferenceEngine` async master tick 50 ms loop (sync_buffer + AdaptiveBaseline + preprocess + model + TrackingEngine + broadcast). Dashboard: Three.js InstancedMesh 3072 voxel (6 slot × 8³) + oda/raf wireframe + WS client (auto-reconnect) + HUD (FPS, inference_ms, slot status, material legend). Mock mode (model=None) + real model end-to-end test geçti (~144 ms/tick CPU, 200 ms hedef içinde). |
| 6 | **Training+Ablation** ([training/](training/) + [data_synthesis/multi_domain_dataset.py](data_synthesis/multi_domain_dataset.py)) | **✅ 2026-05-19 TAMAMLANDI.** `MultiDomainDataset` 5 preset birleştirici (5000 sample memmap, paired aux + domain_id + cached link_geo) + `collate_with_preprocess` batch-level preprocessing. `train_baseline` (multi-task supervised) + `train_dann_phase` (DANN binary domain) + `distill` CLI fazı. `evaluate` master loop (det F1/precision/recall, rec masked acc, id bit acc + full codeword acc). `run_ablation` 5 varyant (full / no_path_aux / no_dann / no_kd / minimal) JSON dump + özet tablo. **LR scheduler (2026-05-19 EK):** `create_scheduler` = LinearLR warmup (2 ep) + CosineAnnealingLR (1e-3→1e-5) `SequentialLR` zinciri. Multi-domain için **tek scheduler** (interleaved shuffle batching → domain shift "anı" yok → restart yok). `train_*` ve `run_*_ablation` scheduler-aware, history'ye `lr` kaydı. Birim test 4ep × 500 sample: peak'te (ep 2) L 3.09 → 2.61 büyük düşüş, cosine inişi yumuşak. **Mevcut 5k veri × 2 epoch CPU testi (scheduler öncesi):** det_f1 0.70–0.76, rec_acc ~0.25 (= 4-sınıf random, 2 epoch yetersiz — production'da 30-50 epoch + 5×100k veri gerekli), id_bit ~0.50–0.54, eğitim 122-198 s/varyant, toplam ~13 dk. Pipeline end-to-end sağlam ✓. |
| 0.5 | **Sensor Placement Randomization** ([data_synthesis/layout_variants.py](data_synthesis/layout_variants.py)) | **✅ 2026-05-19 TAMAMLANDI.** Modelin spesifik anchor/ESP32 koordinatlarını ezberlemesini engelleyen ön-üretim katmanı. `generate_layout_variants(base_nodes, n_variants=5, jitter_xy=0.3 m, jitter_z=0.1 m, room_clip=0.3 m, seed)` → 8 düğümün hepsine deterministic gaussian jitter + oda sınırı clip. `v=0` her zaman identity (backward compat); `v>=1` deterministic per-variant seed. UWB ve WiFi script main() loop'ları variant rotation kullanır (sample i → variant `i // (n/n_variants)` → cfg_variant geçirilir). Çıktıya `layout_ids.npy [N]` + `layout_meta.yaml` (her variant için 8 node konum referansı) eklenir. `MultiDomainDataset` sample'a göre doğru variant link_geo'sunu cache'den döndürür (layout-aware). Eski 5k veri **backward-compat**: `layout_ids.npy` yoksa hepsi 0 → scene.yaml'dan tek link_geo. Birim test: 1k mock × 5 variant × jitter → s0 vs s500 link_geo farkı ~35 cm, intra-variant fark = 0, deterministic seed garantili. Production: 5 preset × 5 variant × 20k = 500k. |

---

*Bu README, `DATASHEETTTTTTT.pdf` (yazar: metin bora özke, oluşturma: 2026-05-13) içeriğinin tamamını birebir korur. PDF'teki dil, ton ve formüller bilinçli olarak değiştirilmemiştir.*
