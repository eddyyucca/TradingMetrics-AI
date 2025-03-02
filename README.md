# 🚀 TradingMetrics-AI

> Developed by Eddy Adha Saputra

Alat analisis teknikal untuk trading crypto dengan visualisasi berwarna, rekomendasi manajemen risiko, fitur live monitoring, dan kecerdasan buatan.

[![GitHub](https://img.shields.io/badge/GitHub-TradingMetrics--AI-black?style=flat&logo=github)](https://github.com/eddyyucca/TradingMetrics-AI)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ko-Fi](https://img.shields.io/badge/Ko--fi-F16061?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/eddyyucca)

## ✨ Fitur

- 📊 Analisis multi-indikator (RSI, MACD, Bollinger Bands, dll)
- 💹 Support untuk 40+ cryptocurrency populer (BTC, ETH, BNB, dll)
- 🎯 Rekomendasi entry point dan exit levels
- ⚠️ Manajemen risiko dengan multiple stop loss
- 📈 Take profit berdasarkan volatilitas
- 🎨 Output berwarna di terminal
- 🔄 Mode live monitoring untuk pantau beberapa aset sekaligus
- 📱 Tampilan tabel keputusan trading real-time
- ⏱️ Interval refresh yang dapat disesuaikan
- 🧠 **NEW:** Analisis lanjutan dengan kecerdasan buatan
- 🔍 **NEW:** Deteksi pola candlestick otomatis
- 📊 **NEW:** Analisis konteks pasar dan support/resistance

## 🗂 Struktur Kode

```
TradingMetrics-AI/
│
├── main.py              # Program utama dengan semua fitur
├── analysis.py          # Logika analisis teknikal
├── decision.py          # Sistem pengambilan keputusan
│
├── intelligence/        # Modul kecerdasan buatan
│   ├── pattern_recognition.py  # Deteksi pola candlestick
│   ├── market_context.py       # Analisis konteks pasar
│   ├── risk_manager.py         # Manajemen risiko canggih
│   ├── ml_models.py            # Prediksi machine learning
│   └── ai_advisor.py           # Rekomendasi AI terintegrasi
│
├── intelligence_integration.py  # Integrasi semua modul AI
│
├── config/              # Direktori konfigurasi
├── models/              # Model machine learning tersimpan
├── data/                # Data untuk analisis
├── analysis_results/    # Hasil analisis tersimpan
│
└── requirements.txt     # Dependencies
```

### Penjelasan File

- `main.py`: Program utama dengan semua fitur termasuk AI
- `analysis.py`: Fungsi-fungsi analisis teknikal
- `decision.py`: Algoritma keputusan trading
- `intelligence/`: Direktori dengan modul kecerdasan buatan
- `intelligence_integration.py`: Integrasi semua modul AI

## 🛠 Teknologi

- Python 3.8+
- Pandas & NumPy untuk analisis data
- PyGecko API untuk data market
- Scikit-learn untuk machine learning
- Tabulate untuk tampilan tabel
- Threading untuk monitoring live
- Colorama & Termcolor untuk visualisasi

## 📦 Instalasi

1. Pastikan Python 3.8+ terinstall

```bash
python --version
```

2. Clone repository

```bash
git clone https://github.com/eddyyucca/TradingMetrics-AI.git
cd TradingMetrics-AI
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Install dependensi tambahan untuk fitur AI (opsional)

```bash
pip install scikit-learn joblib matplotlib
```

## 🚀 Cara Penggunaan

```bash
python main.py
```

### Menu Utama

Pilih opsi yang tersedia:

1. **Analisis Single Crypto** - Analisis mendalam untuk satu aset
2. **Mode Live Monitoring** - Pantau beberapa aset secara real-time
3. **Kelola Aset yang Dipantau** - Tambah/hapus aset untuk monitoring
4. **Analisis Lanjutan dengan AI** - Analisis komprehensif dengan AI
5. **Keluar** - Keluar dari program

### Analisis Single Crypto

- Pilih dari 40 cryptocurrency populer
- Pilih timeframe (15m, 30m, 1h, 4h)
- Lihat analisis detail dan keputusan trading
- Opsi untuk menambahkan ke daftar pantauan live

### Mode Live Monitoring

- Pantau beberapa aset secara bersamaan
- Tabel keputusan trading yang update secara otomatis
- Diurutkan berdasarkan tingkat keyakinan tertinggi
- Refresh otomatis dengan interval yang dapat disesuaikan

### Analisis Lanjutan dengan AI

- Deteksi pola candlestick otomatis
- Analisis konteks pasar dengan deteksi fase (uptrend, downtrend, dll)
- Identifikasi level support dan resistance kunci
- Prediksi arah harga dengan machine learning
- Rekomendasi trading komprehensif dengan tingkat keyakinan
- Opsi untuk menyimpan hasil analisis ke file

## 📊 Contoh Output Live Monitoring

```
=== LIVE TRADING DECISIONS ===
Last Update: 2025-03-04 01:10:25
+------------+------------+------------+------------+------------+------------+-------------------+
| Coin       | Harga      | Aksi       | Keyakinan  | Timeframe  | Volatilitas| Waktu Update     |
+============+============+============+============+============+============+===================+
| BTC/USDT   | $65,890.12 | STRONG_BUY | 95.7%      | 1h         | 2.31%      | 01:10:22         |
+------------+------------+------------+------------+------------+------------+-------------------+
| ETH/USDT   | $3,127.45  | BUY        | 78.3%      | 15m        | 3.45%      | 01:10:23         |
+------------+------------+------------+------------+------------+------------+-------------------+
| SOL/USDT   | $128.67    | HOLD       | 62.1%      | 4h         | 4.82%      | 01:10:24         |
+------------+------------+------------+------------+------------+------------+-------------------+
| DOGE/USDT  | $0.1432    | SELL       | 82.5%      | 30m        | 5.67%      | 01:10:25         |
+------------+------------+------------+------------+------------+------------+-------------------+
```

## 📋 Contoh Output Analisis AI

```
=== AI TRADING ADVISOR (DOGE/15m) ===
Recommended Action: SELL
Confidence: 34.0%

Trading Advice:
• Weak sell signal with 34.0% confidence. Consider reducing position size.
• Market is in a downtrend - capital preservation should be priority.
• Consider waiting for bounce to resistance at $0.20 for better exit.
• Recommended position size: $5477.07
• Consider scaling in to reduce entry risk
• Low confidence signal - higher risk

Analysis Contributors:
• Technical Analysis: 14.0
• Market Context: -10.6
• Pattern Recognition: 0.0
• Machine Learning: -17.0

Market Context:
• Market Phase: Downtrend (Strong downtrend detected, momentum is negative)
• Volatility: 0.52%

Support Levels:
• S1: $0.20 (Strength: 99%)

Resistance Levels:
• R1: $0.20 (Strength: 95%)
• R2: $0.22 (Strength: 40%)

Risk Management:
• Risk Profile: Medium
• Recommended Position Size: $5477.07
• Risk Amount: $17.00

Machine Learning Prediction:
• Predicted Direction: DOWN
• Confidence: 67.9%
• Model Freshness: Fresh (just trained)
```

## 🔄 Fitur Tambahan

- **Penyimpanan Konfigurasi**: Aset yang dipantau dan interval refresh disimpan antar sesi
- **Pemantauan Multi-Timeframe**: Pantau aset yang sama di berbagai timeframe sekaligus
- **Peringkat Keputusan**: Keputusan trading diurutkan berdasarkan keyakinan tertinggi
- **User Interface yang Lebih Baik**: Menu terstruktur dengan navigasi yang jelas
- **Penyimpanan Hasil Analisis**: Simpan hasil analisis AI ke file teks untuk referensi

## 📡 Sumber Data

Program ini menggunakan CoinGecko API untuk mendapatkan data cryptocurrency, mengatasi masalah SSL/API yang umum terjadi pada API lain.

## ⚠️ Disclaimer

- Ini adalah alat analisis teknikal, bukan rekomendasi trading
- Selalu gunakan manajemen risiko yang baik
- Past performance tidak menjamin hasil di masa depan
- DYOR (Do Your Own Research)

## 📄 Lisensi

MIT License - lihat file [LICENSE](LICENSE) untuk detail

## 🤝 Kontribusi

Kontribusi selalu diterima! Feel free untuk membuat pull request atau melaporkan issues.

## 👨‍💻 Developer

**Eddy Adha Saputra**

[![GitHub](https://img.shields.io/badge/GitHub-eddyyucca-black?style=flat&logo=github)](https://github.com/eddyyucca)
[![Email](https://img.shields.io/badge/Email-eddyyucca%40gmail.com-red?style=flat&logo=gmail)](mailto:eddyyucca@gmail.com)
[![Ko-Fi](https://img.shields.io/badge/Support%20Me%20on%20Ko--fi-F16061?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/eddyyucca)

---

<p align="center">
  <a href="https://ko-fi.com/eddyyucca">
    <img src="https://storage.ko-fi.com/cdn/kofi3.png?v=3" alt="Buy Me A Coffee at ko-fi.com" height="45">
  </a>
</p>

Made with ❤️ in Indonesia
