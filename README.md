# 🚀 TradingMetrics-AI

> Developed by Eddy Adha Saputra

Alat analisis teknikal untuk trading crypto dengan visualisasi berwarna dan rekomendasi manajemen risiko.

[![GitHub](https://img.shields.io/badge/GitHub-TradingMetrics--AI-black?style=flat&logo=github)](https://github.com/eddyyucca/TradingMetrics-AI)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Fitur

- 📊 Analisis multi-indikator (RSI, MACD, Bollinger Bands, dll)
- 💹 Support untuk BTC, DOGE, SHIB, dan FLOKI
- 🎯 Rekomendasi entry point dan exit levels
- ⚠️ Manajemen risiko dengan multiple stop loss
- 📈 Take profit berdasarkan volatilitas
- 🎨 Output berwarna di terminal

## 🗂 Struktur Kode

```
TradingMetrics-AI/
│
├── main.py              # Program utama dan UI
├── analysis.py          # Logika analisis teknikal
├── decision.py          # Sistem pengambilan keputusan
├── indicators.py        # Perhitungan indikator teknikal
├── requirements.txt     # Dependencies
└── README.md            # Dokumentasi
```

### Penjelasan File

- `main.py`: User interface dan logika utama program
- `analysis.py`: Fungsi-fungsi analisis teknikal
- `decision.py`: Algoritma keputusan trading
- `indicators.py`: Implementasi indikator (RSI, MACD, dll)

## 🛠 Teknologi

- Python 3.8+
- Pandas untuk analisis data
- Requests untuk API calls
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

## 🚀 Cara Penggunaan

### 1. Menjalankan Program

```bash
python main.py
```

### 2. Menu Utama

Pilih opsi yang tersedia:

- Pilih crypto (1-4)
  1. BTC/USDT
  2. DOGE/USDT
  3. SHIB/USDT
  4. FLOKI/USDT

### 3. Pilih Timeframe

- 15 Menit
- 30 Menit
- 1 Jam
- 4 Jam

### 4. Membaca Hasil Analisis

Program akan menampilkan:

- Harga saat ini
- Rekomendasi entry points
- Level stop loss (ketat, sedang, longgar)
- Target take profit (aman, sedang, agresif)
- Kekuatan sinyal beli/jual
- Analisis indikator teknikal

### 5. Manajemen Risiko

- Gunakan stop loss yang disarankan
- Perhatikan ukuran posisi yang direkomendasikan
- Ikuti risk:reward ratio yang ditampilkan

## 📊 Contoh Output

```
=== BTC/USDT Analysis ===
Timeframe: 15m
Current Price: $44,235.75

Signal Strength:
BUY:  ================ 80%
SELL: ====== 30%

Recommended Levels:
Stop Loss: $43,794.39
Take Profit: $44,897.45

Risk Management:
Position Size: Rp 150,000
Risk per Trade: 2%
```

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

---

Made with ❤️ in Indonesia
