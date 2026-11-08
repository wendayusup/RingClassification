# Industrial Bearing Defect Inspection System (Ring Classification)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Tugas Besar Artificial Intelligence Technology (AIT) - Semester 5**  
> **Politeknik Manufaktur Bandung (POLMAN Bandung)**  
> Sistem Computer Vision & Deep Learning berbasis **Pure Convolutional Neural Network (CNN)** dengan integrasi **Explainable AI (Grad-CAM)** dan **Metrology Filtering** untuk inspeksi cacat permukaan ring bearing manufaktur secara *real-time*.

---

## Daftar Isi
1. [Latar Belakang dan Fitur Utama](#latar-belakang-dan-fitur-utama)
2. [Arsitektur Proyek dan Pipeline (Stage 1 - 6)](#arsitektur-proyek-dan-pipeline-stage-1---6)
3. [Arsitektur Model (Pure CNN 7-Stage)](#arsitektur-model-pure-cnn-7-stage)
4. [Explainable AI (XAI) dan Defect Localization](#explainable-ai-xai-dan-defect-localization)
5. [Instalasi dan Persyaratan Sistem](#instalasi-dan-persyaratan-sistem)
6. [Panduan Menjalankan Sistem](#panduan-menjalankan-sistem)
7. [Hasil Evaluasi dan Visualisasi](#hasil-evaluasi-dan-visualisasi)
8. [Panduan Push ke GitHub (HTTPS)](#panduan-push-ke-github-https)

---

## Latar Belakang dan Fitur Utama

Dalam industri manufaktur presisi, cacat mikro pada permukaan ring bearing (seperti goresan, retakan, deformasi benturan, atau ketidakrataan permukaan) dapat menyebabkan kegagalan bearing fatal pada mesin rotasi berkecepatan tinggi.

Proyek ini menghadirkan sistem inspeksi visual cerdas otomatis yang mencakup:
- **Zero Data Leakage Pipeline**: Pemisahan dataset (Train 71%, Val 12.5%, Test 16.5%) berbasis ID sampel fisik unik (Sample 01-24) untuk menjamin keabsahan evaluasi ilmiah.
- **Offline Data Augmentation**: Multiplikasi variasi orientasi (rotasi 90 deg, 180 deg, 270 deg, flip horizontal/vertikal, variasi pencahayaan) hingga 9x lipat per sampel.
- **Custom Pure CNN Architecture**: Jaringan CNN 7-tahap tanpa transfer learning pra-latih, dilengkapi Batch Normalization dan Dropout (0.4) untuk generalisasi tinggi.
- **Dynamic Optimization**: Pelatihan dengan Adam Optimizer, L2 Regularization (Weight Decay), serta Dynamic Learning Rate Scheduler (ReduceLROnPlateau).
- **Explainable AI (XAI) dan Metrology**: Kombinasi Grad-CAM, Morphological Black-Hat Transform, dan Sobel Gradient untuk mendeteksi serta menandai titik koordinat cacat secara objektif.
- **Interactive Web Inspection Dashboard**: Antarmuka web modern (Flask) dengan fitur Live Webcam Capture, Drag-and-Drop Image Upload, visualisasi side-by-side titik cacat QC, dan pengukuran latensi inferensi real-time.

---

## Arsitektur Proyek dan Pipeline (Stage 1 - 6)

```text
RingClassification/
├── .vscode/                     # Konfigurasi IDE dan Python path analysis
├── dataset_raw/                 # Citra mentah hasil pemotretan dan augmentasi offline
│   ├── defect_ring/             # 216 citra ring bearing cacat
│   └── normal_ring/             # 216 citra ring bearing normal
├── dataset_split/               # Dataset terdistribusi bebas data leakage
│   ├── train/                   # 153 defect + 153 normal (ID 01 - 17)
│   ├── val/                     # 27 defect + 27 normal   (ID 18 - 20)
│   └── test/                    # 36 defect + 36 normal   (ID 21 - 24)
├── onceCoding/                  # Skrip pemrosesan dataset awal
│   ├── augment_and_rename.py    # Standardisasi nama dan augmentasi geometri/iluminasi
│   └── split_dataset.py         # Distribusi dataset terisolasi per sample ID
├── results/                     # Hasil plot, kurva metrik, dan visualisasi
│   ├── 01_data_loader_preview.png
│   ├── 03_training_history.png
│   ├── 04_confusion_matrix.png
│   └── 05_inference_result.png
├── saved_models/                # Checkpoint model terbaik
│   └── pure_cnn_best.pth        # Bobot model Pure CNN optimal (~24 MB)
├── step01_data_preparation/     # [STAGE 1] Preprocessing dan PyTorch DataLoader
│   ├── __init__.py
│   └── dataset_loader.py
├── step02_model_architecture/   # [STAGE 2] Definisi Arsitektur Pure CNN
│   ├── __init__.py
│   └── pure_cnn.py
├── step03_training/             # [STAGE 3] Training loop dan Hyperparameter tuning
│   └── train.py
├── step04_evaluation/           # [STAGE 4] Evaluasi kuantitatif pada Test Set
│   └── evaluate.py
├── step05_inference/            # [STAGE 5] Inferensi deterministik dan XAI Localizer
│   ├── __init__.py
│   ├── inference.py
│   └── defect_localizer.py
├── step06_web_app/              # [STAGE 6] Industrial Web Dashboard (Flask)
│   ├── __init__.py
│   ├── app.py
│   └── templates/
│       └── index.html
├── run_web_app.py               # Quick launcher untuk web aplikasi
├── requirements.txt             # Dependensi pustaka Python
├── .gitignore                   # Konfigurasi pengecualian file Git
└── README.md                    # Dokumentasi lengkap proyek
```

---

## Arsitektur Model (Pure CNN 7-Stage)

Model dirancang dengan topologi berurutan untuk mengekstraksi fitur visual dari level rendah hingga representasi cacat abstrak:

```text
[Input Grayscale: 1 x 128 x 128]
       │
       ▼
[BLOCK 1] Conv2D (1 -> 16, k=3, p=1) + BatchNorm2D + ReLU + MaxPool2D(2x2)
       │  Output: (16 x 64 x 64) -> Ekstraksi tepi dan kontur ring
       ▼
[BLOCK 2] Conv2D (16 -> 32, k=3, p=1) + BatchNorm2D + ReLU + MaxPool2D(2x2)
       │  Output: (32 x 32 x 32) -> Ekstraksi tekstur permukaan dan lengkungan
       ▼
[BLOCK 3] Conv2D (32 -> 64, k=3, p=1) + BatchNorm2D + ReLU + MaxPool2D(2x2)
       │  Output: (64 x 16 x 16) -> Representasi pola cacat/anomali kompleks
       ▼
[Stage 5: Flatten] 64 x 16 x 16 = 16.384 fitur 1D
       │
       ▼
[Stage 6: Dense / Fully Connected] Linear(16384 -> 128) + ReLU + Dropout(p=0.40)
       │
       ▼
[Stage 7: Output Layer] Linear(128 -> 2) -> Logits [Defect, Normal]
```

---

## Explainable AI (XAI) dan Defect Localization

Untuk menjamin transparansi keputusan, Stage 5 menerapkan Unified Objective Industrial Metrology Suite:
1. **Mathematical Morphological Black-Hat**: `Closing(I) - I` untuk mengekstraksi goresan mikro gelap pada latar terang.
2. **Sobel Gradient Magnitude**: Menghitung turunan spasial intensitas piksel.
3. **Gradient-weighted Class Activation Mapping (Grad-CAM)**: Menghitung gradien bobot aktivasi pada lapisan konvolusi terakhir (`conv3`) terhadap prediksi kelas cacat.
4. **Automated QC Annotation**: Mengidentifikasi puncak anomali dan menandai koordinat cacat menggunakan titik merah (red dots) pada citra asli untuk memudahkan operator QC manufaktur.

---

## Instalasi dan Persyaratan Sistem

### Kebutuhan Sistem
- **Python**: Versi 3.10 atau lebih tinggi
- **OS**: Windows 10/11, Ubuntu 20.04+, atau macOS
- **Hardware**: CPU (Intel/AMD) atau GPU NVIDIA dengan CUDA support (opsional)

### Langkah Instalasi
1. Clone repository ini:
   ```bash
   git clone https://github.com/<username-anda>/<nama-repo>.git
   cd RingClassification
   ```

2. Buat dan aktifkan virtual environment (direkomendasikan):
   ```bash
   # Windows (Command Prompt / PowerShell)
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS / Git Bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

---

## Panduan Menjalankan Sistem

### 1. Menjalankan Dashboard Web Inspeksi
Jalankan skrip launcher:
```bash
python run_web_app.py
```
Akses antarmuka melalui browser pada URL: `http://127.0.0.1:5000`

Fitur dashboard:
- File Upload: Drag and Drop citra ring bearing.
- Live Webcam Mode: Ambil foto langsung melalui webcam.
- One-Click Presets: Menguji langsung citra dari Test Set yang belum pernah dilihat model.
- Visualisasi Komparasi: Citra asli berdampingan dengan citra teranotasi titik cacat QC dan metrik probabilitas.

---

### 2. Menjalankan Pipeline Per Tahapan (CLI)

- Tahap 1: Verifikasi DataLoader dan Tampilkan Sampel Preview
  ```bash
  python step01_data_preparation/dataset_loader.py
  ```

- Tahap 2: Verifikasi Arsitektur Model Pure CNN
  ```bash
  python step02_model_architecture/pure_cnn.py
  ```

- Tahap 3: Pelatihan Model (Training Loop)
  ```bash
  python step03_training/train.py
  ```
  Model bobot terbaik akan otomatis tersimpan di `saved_models/pure_cnn_best.pth` dan kurva pelatihan tersimpan di `results/03_training_history.png`.

- Tahap 4: Evaluasi Ilmiah pada Test Set (Confusion Matrix dan Classification Report)
  ```bash
  python step04_evaluation/evaluate.py
  ```

- Tahap 5: Inferensi Mandiri dan XAI Metrology
  ```bash
  python step05_inference/inference.py
  ```

---

## Hasil Evaluasi dan Visualisasi

Semua hasil eksperimen otomatis terekam dalam direktori `results/`:
- `results/01_data_loader_preview.png`: Preview batch tensor citra grayscale (128x128).
- `results/03_training_history.png`: Grafik pergerakan Training vs Validation Loss dan Accuracy per epoch.
- `results/04_confusion_matrix.png`: Heatmap Confusion Matrix pada 72 citra Test Set independen.
- `results/05_inference_result.png`: Visualisasi multi-layer metrologi (Black-Hat, Sobel, Grad-CAM, Red Dot Annotation).

---

## Panduan Push ke GitHub (HTTPS)

Berikut adalah instruksi terminal/bash untuk mengunggah proyek ini ke GitHub menggunakan protokol HTTPS:

### Langkah 1: Siapkan Repository di GitHub
1. Buka GitHub dan buat repository baru (misal bernama `RingClassification`).
2. Jangan centang opsi "Add a README file", "Add .gitignore", atau "Choose a license" karena file tersebut sudah tersedia di proyek ini.
3. Salin URL HTTPS repository Anda (contoh: `https://github.com/<username>/RingClassification.git`).

---

### Langkah 2: Eksekusi Perintah di Terminal / Git Bash

Buka terminal di folder root `RingClassification`, lalu jalankan urutan perintah berikut:

```bash
# 1. Inisialisasi git
git init

# 2. Konfigurasi identitas Git Anda
git config user.name "Nama Lengkap Anda"
git config user.email "email_anda@example.com"

# 3. Perbesar buffer HTTP Git agar file model (24 MB) dan dataset terunggah lancar
git config http.postBuffer 524288000

# 4. Tambahkan seluruh file proyek ke staging area
git add .

# 5. Lakukan commit pertama
git commit -m "feat: complete bearing defect inspection system with CNN and Web GUI"

# 6. Pastikan branch utama bernama 'main'
git branch -M main

# 7. Hubungkan remote repository GitHub (ganti URL dengan link repo Anda)
git remote add origin https://github.com/<USERNAME_ANDA>/RingClassification.git

# 8. Unggah (push) ke GitHub
git push -u origin main
```

---

### Catatan Mengenai Autentikasi GitHub (PAT)
Sejak Agustus 2021, GitHub tidak lagi menerima password akun biasa saat melakukan git push via HTTPS.

Jika terminal meminta password:
1. Buat Personal Access Token (Classic) di GitHub:
   - Buka: Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
   - Klik Generate new token (classic)
   - Berikan nama dan centang checklist repo
   - Klik Generate token lalu salin token yang dihasilkan (`ghp_xxxxxxxxxxxxxxxxxxxx`).
2. Saat terminal meminta:
   - Username: Masukkan username GitHub Anda.
   - Password: Masukkan Personal Access Token yang tadi disalin.

---

## Kontributor
- Mahasiswa Semester 5 - Program Studi Teknologi Rekayasa Otomasi / Manufaktur
- **Politeknik Manufaktur Bandung (POLMAN Bandung)**
- Mata Kuliah: *Artificial Intelligence Technology (AIT)*
