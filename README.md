# IMPLEMENTASI METODE ENSEMBLE LEARNING UNTUK PREDIKSI HARGA PROPERTI RESIDENSIAL DI WILAYAH DKI JAKARTA

Dokumentasi ini menjelaskan proyek analisis data dan pemodelan prediktif harga properti residensial di wilayah DKI Jakarta. Proyek ini disusun oleh **Grup E** sebagai bagian dari tugas mata kuliah Machine Learning, Program Studi Statistika, Universitas Jenderal Soedirman.

---

## 👥 Anggota Kelompok E
1. **Abdurrahman Yusuf** (K1D024058)
2. **Salsabella Setya Choerunissa** (K1D024060)
3. **Eriani Hardianti** (K1D024062)
4. **Rey El Islamay** (K1D024070)

### 🏛️ Lembaga Akademik
* **Program Studi**: Statistika  
* **Fakultas**: Matematika dan Ilmu Pengetahuan Alam  
* **Universitas**: Universitas Jenderal Soedirman (UNSOED)  
* **Kota/Tahun**: Purwokerto, 2026  

---

## 📌 Ringkasan Proyek
Sektor properti residensial di DKI Jakarta memiliki dinamika harga yang sangat kompleks, bervarians lebar, dan bergerak secara non-linear. Penaksiran harga konvensional yang bersifat manual sering kali subjektif dan lambat. Proyek ini bertujuan untuk membangun model prediksi otomatis yang objektif, transparan, dan akurat untuk membantu memitigasi asimetri informasi di pasar properti sekunder serta mempercepat manajemen risiko kredit perbankan.

Berdasarkan taksonomi machine learning, proyek ini diklasifikasikan sebagai **Persoalan Regresi (Regression Problem)** karena variabel targetnya berupa harga properti residensial yang bernilai numerik kontinu dalam mata uang Rupiah.

---

## 📊 Dataset & Fitur
Dataset utama yang digunakan adalah `jakarta_house.csv` yang berisi **10.000 baris data** transaksi rumah di DKI Jakarta. Fitur-fitur dalam dataset meliputi:

| Nama Kolom | Deskripsi | Tipe Data |
|---|---|---|
| `price` | Harga properti residensial dalam mata uang Rupiah (Variabel Target) | Numerik (Kontinu) |
| `city` | Wilayah kota administrasi di DKI Jakarta (Jakarta Barat, Pusat, Selatan, Timur, Utara) | Kategorik (Nominal) |
| `district` | Wilayah kecamatan lokasi properti berada (253 kecamatan unik) | Kategorik (Nominal) |
| `bed_rooms` | Jumlah kamar tidur yang tersedia | Numerik (Diskrit) |
| `bath_rooms` | Jumlah kamar mandi yang tersedia | Numerik (Diskrit) |
| `carport` | Kapasitas area parkir kendaraan / carport | Numerik (Diskrit) |
| `land_area` | Luas tanah properti dalam meter persegi (m²) | Numerik (Kontinu) |
| `building_area` | Luas bangunan properti dalam meter persegi (m²) | Numerik (Kontinu) |

---

## ⚙️ Metodologi & Alur Kerja (Workflow)
Alur kerja yang diterapkan di dalam Jupyter Notebook `Project_ML_Grup E.ipynb` dan Laporan Proyek terbagi menjadi beberapa tahapan sistematis:

### 1. Eksplorasi Data (EDA) & Identifikasi Kualitas Data
* **Statistik Deskriptif**: Menganalisis nilai sebaran (mean, median, standar deviasi, min, dan max) dari setiap variabel numerik serta nilai frekuensi dari variabel kategorik.
* **Pengecekan Kualitas Data**:
  * Mengidentifikasi *missing values* kecil pada `land_area` (3 baris) dan `building_area` (5 baris).
  * Menemukan **526 baris data terduplikat**.
  * Mendeteksi adanya data pencilan (outliers) dan anomali ekstrim, seperti harga rumah Rp1.850.000 yang tidak realistis untuk DKI Jakarta, serta jumlah kamar tidur/mandi yang mencapai 63 unit.

### 2. Pembersihan Data & Rekayasa Fitur (Feature Engineering)
* **Penghapusan Duplikat**: Menghapus seluruh baris data duplikat.
* **Pembersihan Anomali**: Menghapus baris data rumah dengan harga terlalu rendah (di bawah Rp100 juta) dan data dengan jumlah kamar tidur/mandi bernilai 0.
* **Rekayasa Fitur (Feature Engineering)**: Membuat tiga fitur baru untuk meningkatkan informasi data:
  1. `building_land_ratio`: Rasio luas bangunan terhadap luas tanah (`building_area` / `land_area`).
  2. `total_rooms`: Jumlah total kamar tidur dan kamar mandi (`bed_rooms` + `bath_rooms`).
  3. `area_per_bedroom`: Rasio luas bangunan per kamar tidur (`building_area` / `bed_rooms`).

### 3. Arsitektur Preprocessing Pipeline & Stratifikasi
* **Transformasi Target**: Menerapkan transformasi logaritma (`np.log1p`) pada variabel target `price` untuk mereduksi kecondongan distribusi yang menjulur tajam ke kanan (right-skewed distribution).
* **Pembagian Dataset**: Membagi data menjadi *Train Set* dan *Test Set* dengan proporsi **80:20**.
* **Preprocessing Pipeline (ColumnTransformer)**:
  * **Fitur Numerik**: Diimputasi menggunakan nilai median (`SimpleImputer`) untuk mengisi missing values, kemudian diskalakan menggunakan `StandardScaler`.
  * **Fitur Kategorik**: Diimputasi menggunakan nilai modus (`SimpleImputer` strategi `most_frequent`), kemudian dikodekan menggunakan `OneHotEncoder`.
  * Pendekatan pipeline ini memastikan proses pembelajaran terhindar dari kebocoran data (*data leakage*).

### 4. Pemodelan & Tuning Hyperparameter
Eksperimen membandingkan empat jenis model regresi dengan karakteristik matematis yang berbeda:
1. **Linear Regression** (Baseline parametrik sederhana).
2. **Ridge Regression** (Regresi linear dengan regularisasi L2, $\alpha = 1.0$).
3. **Random Forest Regressor** (Ensemble berbasis *bagging* / pohon keputusan).
4. **XGBoost Regressor** (Ensemble berbasis *gradient boosting*).

**Optimasi Hyperparameter**: Model XGBoost dipilih sebagai kandidat untuk optimasi menggunakan `RandomizedSearchCV` dengan 3-Fold Cross Validation. Ruang parameter pencarian meliputi:
* `n_estimators`: `[100, 200, 300]`
* `max_depth`: `[4, 6, 8]`
* `learning_rate`: `[0.01, 0.05, 0.1, 0.2]`
* `subsample`: `[0.8, 1.0]`
* `colsample_bytree`: `[0.8, 1.0]`

---

## 📈 Hasil Eksperimen & Evaluasi

Evaluasi performa model dihitung menggunakan data uji (*test set*) yang belum pernah dilihat sebelumnya. Seluruh hasil prediksi yang bernilai logaritma dikembalikan ke skala Rupiah asli menggunakan fungsi inversi eksponensial `np.expm1()` sebelum metrik dihitung.

### 1. Perbandingan Performa Awal Model (Sebelum Tuning)
| No | Model Algoritma | MAE (Rupiah) | RMSE (Rupiah) | R² Score |
|---|---|---|---|---|
| 1 | **Baseline — Linear Regression** | Rp4.873.481.000 | Rp33.062.210.000 | 0,7300 |
| 2 | **Ridge Regression ($\alpha = 1.0$)** | Rp4.920.624.000 | Rp34.174.820.000 | 0,7400 |
| 3 | **Random Forest Regressor** | Rp3.042.631.000 | Rp9.982.091.000 | 0,8800 |
| 4 | **XGBoost Regressor** | Rp2.745.802.000 | Rp8.848.488.000 | 0,8900 |

### 2. Performa XGBoost Sebelum vs Setelah Hyperparameter Tuning
| Metrik Evaluasi | Sebelum Tuning (XGBoost) | Setelah Tuning (XGBoost Tuned) | Peningkatan Efisiensi |
|---|---|---|---|
| **MAE (Rupiah)** | Rp2.745.802.000 | **Rp2.621.293.349,19** | Penurunan kesalahan $\approx$ **Rp124,5 Juta** |
| **RMSE (Rupiah)** | Rp8.848.488.000 | **Rp8.304.090.781,79** | Penurunan kesalahan $\approx$ **Rp544,3 Juta** |
| **R² Score** | 0,8900 | **0,9041** | Peningkatan akurasi $\approx$ **1,41%** |

*Model **XGBoost Tuned** merupakan model terbaik yang mampu menjelaskan **90,41%** variabilitas harga properti residensial di DKI Jakarta.*

### 3. Analisis Sisaan (Residual Analysis)
* **Plot Prediksi vs Aktual**: Titik koordinat pengujian mengelompok secara rapat di sekitar garis diagonal regresi ideal ($Y_{\text{aktual}} = Y_{\text{prediksi}}$) pada rentang harga rumah dominan (di bawah Rp20 Miliar).
* **Plot Residual vs Predicted**: Menunjukkan sebaran galat (error) yang acak dan konstan di sekitar garis error = 0 (skala logaritma), mengindikasikan model bebas dari gejala heteroskedastisitas dan telah menyerap seluruh pola data dengan baik.

### 4. Kontribusi Fitur Terpenting (Feature Importance)
Berikut adalah daftar 10 fitur yang paling dominan dalam memengaruhi estimasi harga rumah di DKI Jakarta berdasarkan model XGBoost terbaik:
| Peringkat | Nama Variabel Fitur | Skor Kepentingan (Importance Score) | Tipe Variabel |
|---|---|---|---|
| 1 | `land_area` (Luas Tanah) | 0,1271 | Numerik Asli |
| 2 | `city_Jakarta Timur` | 0,0636 | Kategorik (Kota) |
| 3 | `building_area` (Luas Bangunan) | 0,0501 | Numerik Asli |
| 4 | `district_Kebayoran Baru` | 0,0478 | Kategorik (Kecamatan) |
| 5 | `city_Jakarta Selatan` | 0,0415 | Kategorik (Kota) |
| 6 | `district_Menteng` | 0,0395 | Kategorik (Kecamatan) |
| 7 | `district_Pondok Indah` | 0,0330 | Kategorik (Kecamatan) |
| 8 | `city_Jakarta Pusat` | 0,0270 | Kategorik (Kota) |
| 9 | `district_Pantai Indah Kapuk` | 0,0210 | Kategorik (Kecamatan) |
| 10 | `city_Jakarta Barat` | 0,0200 | Kategorik (Kota) |

---

## 🔑 Kesimpulan & Saran
### Kesimpulan
* Metode **Ensemble Learning** berbasis pohon (bagging dan boosting) terbukti jauh lebih unggul dan kokoh dalam memetakan harga properti non-linear di DKI Jakarta dibandingkan model regresi linear biasa.
* Model akhir **XGBoost Tuned** memperoleh nilai **$R^2$ sebesar 0,9041**, yang berarti model mampu menerangkan 90,41% variasi harga rumah di dataset.
* **Luas Tanah (`land_area`)** dan **Luas Bangunan (`building_area`)** merupakan faktor penentu utama harga, didukung kuat oleh faktor spasial lokasi administratif (seperti wilayah Jakarta Timur/Selatan serta kecamatan premium Kebayoran Baru, Menteng, dan Pondok Indah).

### Saran Pengembangan
* **Integrasi Data Spasial Baru**: Menambahkan fitur eksternal seperti jarak ke CBD, akses stasiun MRT/LRT/Tol, sekolah, rumah sakit, dan tingkat kerawanan banjir.
* **Algoritma Alternatif**: Mencoba algoritma optimasi tabel lainnya seperti `LightGBM` dan `CatBoost`, serta pemodelan berbasis `Stacking Ensemble`.
* **Implementasi Praktis**: Mengemas model terpilih ke dalam aplikasi berbasis web (menggunakan Streamlit atau Flask) sebagai sistem pendukung keputusan otomatis bagi masyarakat, perbankan, dan investor.

---

## 📁 Struktur Folder Proyek
```text
project_kel/
│
├── jakarta_house.csv                                                   # Dataset utama (10.000 listings rumah)
├── Project_ML_Grup E.ipynb                                             # Jupyter Notebook / Script Python analisis & pemodelan
├── LAPORAN PROJECT IMPLEMENTASI METODE ENSEMBLE LEARNING ... .pdf     # Laporan lengkap implementasi proyek (PDF)
└── README.md                                                           # File dokumentasi ini (Markdown)
```

---

## 🔗 Tautan Eksternal Proyek
* **Repositori GitHub**: [Project-ML-Grup-E](https://github.com/Rellamm/Project-ML-Grup-E)
* **Google Colab Notebook**: [Project ML Grup E - Notebook](https://colab.research.google.com/drive/1Zgj1BUptKpSDhi6KcjC35Y6G-0E2PIPc?usp=sharing)
* **Google Gemini Chat Share**: [Gemini AI Chat History](https://share.gemini.google/NWtvXiefE1Wq)
