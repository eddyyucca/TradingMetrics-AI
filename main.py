import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from termcolor import colored
import colorama
from analysis import analyze_indicators
from decision import make_decision
import logging
import json
from pycoingecko import CoinGeckoAPI
import threading
import os
import sys
from tabulate import tabulate

colorama.init()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Variabel global untuk thread live dan konfigurasi
live_running = False
live_thread = None
tracked_coins = []
refresh_interval = 60  # Refresh data setiap 60 detik secara default

def print_banner():
    banner = colored("""
╔═══════════════════════════════════════════════════════════╗
║           TradingMetrics-AI - v2.1.0                      ║
╠═══════════════════════════════════════════════════════════╣
║ Developed by github.com/eddyyucca                         ║
║ Alat Analisis Teknikal Trading Cryptocurrency             ║
║ + Fitur Live Decision                                     ║
╚═══════════════════════════════════════════════════════════╝
""", 'green', attrs=['bold'])
    print(banner)

    """Display startup banner with tool information and loading animation"""
    import time
    # Loading animation
    loading_stages = [
        "Memuat modul analisis...",
        "Menginisialisasi konfigurasi...",
        "Menyiapkan data historis...",
        "Mengonfigurasi indikator teknikal...",
        "Membangun model keputusan...",
        "Siap untuk digunakan :)"
    ]
    for stage in loading_stages:
        print(colored("• ", 'green') + colored(stage, 'white'))
        time.sleep(0.3)  # Simulasi proses loading lebih cepat
    
    # Tampilkan waktu dan informasi tambahan
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n")
    print(colored(f"Waktu Inisialisasi: {current_time}", 'yellow'))
    logger.info("TradingMetrics-AI diinisialisasi")

def ambil_data_crypto(simbol, interval="15m", limit=100):
    try:
        # Mapping interval ke days untuk CoinGecko
        days_map = {
            "15m": 1,     # 1 hari data dengan interval 15 menit
            "30m": 2,     # 2 hari data dengan interval 30 menit
            "1h": 7,      # 7 hari data dengan interval 1 jam
            "4h": 30      # 30 hari data dengan interval 4 jam
        }
        
        # Mapping simbol Binance ke id CoinGecko
        coin_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'DOGE': 'dogecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOT': 'polkadot',
            'SHIB': 'shiba-inu',
            'MATIC': 'matic-network',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'PEPE': 'pepe',
            'MEME': 'meme',
            'BONK': 'bonk',
            'WLD': 'worldcoin-wld',
            'INJ': 'injective-protocol',
            'SUI': 'sui',
            'FLOKI': 'floki'
        }
        
        if simbol not in coin_map:
            raise ValueError(f"Symbol {simbol} tidak ditemukan dalam mapping CoinGecko")
            
        logger.info(f"Mengambil data {simbol} menggunakan CoinGecko API")
        
        cg = CoinGeckoAPI()
        
        # Dapatkan data harga dengan interval
        data = cg.get_coin_market_chart_by_id(
            id=coin_map[simbol],
            vs_currency='usd',
            days=days_map[interval]
        )
        
        # Membuat DataFrame dari data
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        # Konversi timestamp ke format datetime
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
        
        # Gabungkan data
        df = pd.merge_asof(prices, volumes, on='timestamp')
        
        # Perkirakan nilai open, high, low
        # Ini adalah estimasi untuk menggantikan data OHLC lengkap
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] * 1.002  # Estimasi sederhana
        df['low'] = df['close'] * 0.998   # Estimasi sederhana
        
        # Isi nilai yang hilang di awal
        df['open'] = df['open'].fillna(df['close'])
        
        # Reorder kolom agar sesuai dengan format sebelumnya
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Subsample data berdasarkan interval yang dipilih
        if interval == '15m':
            df = df.iloc[::1]  # Ambil setiap baris
        elif interval == '30m':
            df = df.iloc[::2]  # Ambil setiap 2 baris
        elif interval == '1h':
            df = df.iloc[::4]  # Ambil setiap 4 baris
        elif interval == '4h':
            df = df.iloc[::16]  # Ambil setiap 16 baris
        
        # Terbatas pada jumlah baris yang diminta
        df = df.tail(limit)
        
        logger.info(f"Berhasil mengambil {len(df)} baris data untuk {simbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching CoinGecko data: {str(e)}")
        print(f"Error fetching data: {str(e)}")
        print("Gagal mengambil data. Silakan coba lagi.")
        return None

def hitung_level_resiko(df, harga_sekarang, modal_awal=150000):
    volatilitas = df['close'].pct_change().std() * 100
    atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    rata_atr = atr.mean()
    
    if volatilitas > 5:
        persen_sl = 0.02
        rasio_tp = 3
    elif volatilitas > 3:
        persen_sl = 0.015
        rasio_tp = 2.5
    else:
        persen_sl = 0.01
        rasio_tp = 2

    support_kuat = df['low'].tail(20).min()
    support_lemah = harga_sekarang - rata_atr
    resistance_lemah = harga_sekarang + rata_atr
    resistance_kuat = df['high'].tail(20).max()

    jumlah_resiko = modal_awal * 0.02
    ukuran_posisi = jumlah_resiko / persen_sl
    
    return {
        'zona_entry': {
            'beli_kuat': support_kuat,
            'beli_lemah': support_lemah,
            'jual_lemah': resistance_lemah,
            'jual_kuat': resistance_kuat
        },
        'stop_loss': {
            'ketat': harga_sekarang * (1 - persen_sl),
            'sedang': harga_sekarang * (1 - persen_sl * 1.5),
            'longgar': harga_sekarang * (1 - persen_sl * 2)
        },
        'take_profit': {
            'aman': harga_sekarang * (1 + persen_sl * rasio_tp),
            'sedang': harga_sekarang * (1 + persen_sl * rasio_tp * 1.5),
            'agresif': harga_sekarang * (1 + persen_sl * rasio_tp * 2)
        },
        'ukuran_posisi': {
            'disarankan': ukuran_posisi,
            'minimum': modal_awal * 0.5,
            'maksimum': modal_awal * 2
        },
        'volatilitas': volatilitas
    }

def hitung_indikator_tambahan(df):
    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    fib_levels = {
        '0.236': high - (diff * 0.236),
        '0.382': high - (diff * 0.382),
        '0.618': high - (diff * 0.618),
        '0.786': high - (diff * 0.786)
    }
    
    pp = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
    r1 = (2 * pp) - df['low'].iloc[-1]
    s1 = (2 * pp) - df['high'].iloc[-1]
    r2 = pp + (df['high'].iloc[-1] - df['low'].iloc[-1])
    s2 = pp - (df['high'].iloc[-1] - df['low'].iloc[-1])
    
    return {
        'ema': {
            'ema9': df['EMA9'].iloc[-1],
            'ema20': df['EMA20'].iloc[-1],
            'ema50': df['EMA50'].iloc[-1]
        },
        'obv': df['OBV'].iloc[-1],
        'fibonacci': fib_levels,
        'pivot_points': {
            'pp': pp,
            'r1': r1,
            'r2': r2,
            's1': s1,
            's2': s2
        }
    }

def analisis_momentum(df):
    momentum = df['close'].diff(periods=10).iloc[-1]
    tren_kekuatan = 0
    
    if df['close'].iloc[-1] > df['close'].rolling(window=20).mean().iloc[-1]:
        tren_kekuatan += 1
    if df['close'].iloc[-1] > df['close'].rolling(window=50).mean().iloc[-1]:
        tren_kekuatan += 1
        
    volume_rata = df['volume'].mean()
    volume_sekarang = df['volume'].iloc[-1]
    volume_strength = volume_sekarang / volume_rata
    
    return {
        'momentum': momentum,
        'tren_kekuatan': tren_kekuatan,
        'volume_strength': volume_strength
    }

def format_output_crypto(analisis, simbol, timeframe, df):
    harga_sekarang = analisis['current_price']
    level_resiko = hitung_level_resiko(df, harga_sekarang)
    indikator = hitung_indikator_tambahan(df)
    momentum = analisis_momentum(df)
    
    output = colored(f"\n=== Analisis {simbol}/USDT ===\n", 'cyan', attrs=['bold'])
    output += colored(f"Timeframe: {timeframe}\n", 'cyan')
    output += colored(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", 'cyan')
    output += colored(f"Sumber Data: CoinGecko API\n", 'cyan')
    
    output += colored("\nInformasi Harga:\n", 'yellow', attrs=['bold'])
    output += f"Harga Sekarang: {colored(f'${harga_sekarang:.8f}', 'green')}\n"
    output += f"Volatilitas: {colored(f'{level_resiko['volatilitas']:.2f}%', 'yellow')}\n"
    
    entry = level_resiko['zona_entry']
    output += colored("\nZona Entry yang Disarankan:\n", 'yellow', attrs=['bold'])
    output += f"Beli Kuat:    {colored(f'${entry['beli_kuat']:.8f}', 'green')}\n"
    output += f"Beli Lemah:   {colored(f'${entry['beli_lemah']:.8f}', 'green')}\n"
    output += f"Jual Lemah:   {colored(f'${entry['jual_lemah']:.8f}', 'red')}\n"
    output += f"Jual Kuat:    {colored(f'${entry['jual_kuat']:.8f}', 'red')}\n"
    
    sl = level_resiko['stop_loss']
    output += colored("\nLevel Stop Loss:\n", 'yellow', attrs=['bold'])
    output += f"Ketat:    {colored(f'${sl['ketat']:.8f}', 'red')}\n"
    output += f"Sedang:   {colored(f'${sl['sedang']:.8f}', 'red')}\n"
    output += f"Longgar:  {colored(f'${sl['longgar']:.8f}', 'red')}\n"
    
    tp = level_resiko['take_profit']
    output += colored("\nLevel Take Profit:\n", 'yellow', attrs=['bold'])
    output += f"Aman:     {colored(f'${tp['aman']:.8f}', 'green')}\n"
    output += f"Sedang:   {colored(f'${tp['sedang']:.8f}', 'green')}\n"
    output += f"Agresif:  {colored(f'${tp['agresif']:.8f}', 'green')}\n"
    
    pos = level_resiko['ukuran_posisi']
    output += colored("\nUkuran Posisi yang Disarankan (IDR):\n", 'yellow', attrs=['bold'])
    output += f"Minimum:     {colored(f'Rp {pos['minimum']:,.0f}', 'cyan')}\n"
    output += f"Disarankan:  {colored(f'Rp {pos['disarankan']:,.0f}', 'cyan')}\n"
    output += f"Maksimum:    {colored(f'Rp {pos['maksimum']:,.0f}', 'cyan')}\n"
    
    output += colored("\nAnalisis Tambahan:\n", 'yellow', attrs=['bold'])
    output += f"EMA (9/20/50): {colored(f'${indikator['ema']['ema9']:.2f}', 'cyan')} / "
    output += f"{colored(f'${indikator['ema']['ema20']:.2f}', 'cyan')} / "
    output += f"{colored(f'${indikator['ema']['ema50']:.2f}', 'cyan')}\n"
    output += f"Volume Strength: {colored(f'{momentum['volume_strength']:.2f}x', 'cyan')}\n"
    output += f"Tren Kekuatan: {colored(f'{momentum['tren_kekuatan']}/2', 'cyan')}\n"
    
    output += colored("\nKekuatan Sinyal:\n", 'yellow', attrs=['bold'])
    warna_beli = 'green' if analisis['total_buy'] > 60 else 'white'
    warna_jual = 'red' if analisis['total_sell'] > 60 else 'white'
    
    output += f"BELI: {colored('='* int(analisis['total_buy']/2), warna_beli)} {colored(f'{analisis['total_buy']}%', warna_beli)}\n"
    output += f"JUAL: {colored('='* int(analisis['total_sell']/2), warna_jual)} {colored(f'{analisis['total_sell']}%', warna_jual)}\n"
    
    output += colored("\nCatatan Manajemen Resiko:\n", 'red', attrs=['bold'])
    if level_resiko['volatilitas'] > 5:
        output += "• VOLATILITAS TINGGI - Gunakan stop loss yang lebih longgar!\n"
    output += "• Selalu gunakan stop loss\n"
    output += "• Jangan resiko lebih dari 2% modal\n"
    output += "• Pertimbangkan untuk masuk/keluar secara bertahap\n"
    
    return output

def get_color_for_action(action):
    """Mendapatkan warna berdasarkan aksi/keputusan trading"""
    if action in ['STRONG_BUY', 'BUY']:
        return 'green'
    elif action == 'HOLD':
        return 'yellow'
    elif action in ['SELL', 'STRONG_SELL']:
        return 'red'
    return 'white'

def print_live_decision_table(decisions):
    """Menampilkan tabel keputusan live dari semua aset yang dipantau"""
    
    # Header tabel
    headers = ["Coin", "Harga", "Aksi", "Keyakinan", "Timeframe", "Volatilitas", "Waktu Update"]
    
    # Data tabel
    table_data = []
    
    for coin, data in decisions.items():
        if data:  # Periksa jika data tersedia
            row = [
                coin,
                f"${data['price']:.4f}",
                colored(data['action'], get_color_for_action(data['action'])),
                f"{data['confidence']:.1f}%",
                data['timeframe'],
                f"{data['volatility']:.2f}%",
                data['timestamp'].strftime('%H:%M:%S')
            ]
            table_data.append(row)
    
    # Mengurutkan tabel berdasarkan keyakinan (dari tinggi ke rendah)
    table_data.sort(key=lambda x: float(x[3].replace('%', '')), reverse=True)
    
    # Cetak tabel
    if table_data:
        print(colored("\n=== LIVE TRADING DECISIONS ===", 'cyan', attrs=['bold']))
        print(colored(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'yellow'))
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(colored("\nBelum ada data keputusan trading. Silakan tambahkan aset untuk dipantau.", 'yellow'))

def run_live_monitoring():
    """Fungsi untuk menjalankan pemantauan trading secara live"""
    
    global live_running, tracked_coins, refresh_interval
    
    live_decisions = {}
    
    while live_running:
        # Clear console/terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_banner()
        print(colored("\n=== MODE LIVE MONITORING ===", 'green', attrs=['bold']))
        print(colored(f"Memantau {len(tracked_coins)} aset dengan interval refresh {refresh_interval} detik", 'yellow'))
        print(colored("Tekan Ctrl+C untuk menghentikan mode live", 'yellow'))
        
        # Proses setiap aset yang dipantau
        for coin_config in tracked_coins:
            simbol = coin_config['symbol']
            timeframe = coin_config['timeframe']
            
            try:
                df = ambil_data_crypto(simbol, timeframe)
                
                if df is not None:
                    analisis = analyze_indicators(df)
                    keputusan = make_decision(analisis, df)
                    level_resiko = hitung_level_resiko(df, analisis['current_price'])
                    
                    # Simpan keputusan untuk ditampilkan
                    live_decisions[f"{simbol}/USDT"] = {
                        'price': analisis['current_price'],
                        'action': keputusan['action'],
                        'confidence': keputusan['confidence'],
                        'timeframe': timeframe,
                        'volatility': level_resiko['volatilitas'],
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                logger.error(f"Error processing {simbol}: {str(e)}")
                print(f"Error pada {simbol}: {str(e)}")
        
        # Tampilkan tabel keputusan
        print_live_decision_table(live_decisions)
        
        # Tampilkan menu cepat
        print(colored("\nMenu Cepat:", 'cyan'))
        print("1. Tambah aset untuk dipantau")
        print("2. Hapus aset dari pantauan")
        print("3. Ubah interval refresh")
        print("0. Kembali ke menu utama")
        
        # Waktu tunggu sebelum refresh
        for i in range(refresh_interval, 0, -1):
            sys.stdout.write(f"\rRefresh dalam {i} detik...   ")
            sys.stdout.flush()
            
            # Cek apakah proses masih harus berjalan
            if not live_running:
                break
                
            # Tunggu 1 detik
            time.sleep(1)
        
        if not live_running:
            break

def start_live_monitoring():
    """Memulai thread untuk live monitoring"""
    
    global live_running, live_thread, tracked_coins
    
    if live_running:
        print(colored("Mode live monitoring sudah berjalan!", 'yellow'))
        return
    
    if not tracked_coins:
        print(colored("Tidak ada aset yang dipantau. Silakan tambahkan minimal 1 aset.", 'red'))
        return
    
    live_running = True
    live_thread = threading.Thread(target=run_live_monitoring)
    live_thread.daemon = True
    live_thread.start()
    
    print(colored("Mode live monitoring dimulai. Thread ID:", 'green'), live_thread.ident)

def stop_live_monitoring():
    """Menghentikan thread live monitoring"""
    
    global live_running, live_thread
    
    if not live_running:
        print(colored("Mode live monitoring tidak sedang berjalan.", 'yellow'))
        return
    
    live_running = False
    if live_thread:
        live_thread.join(timeout=2)
        print(colored("Mode live monitoring dihentikan.", 'yellow'))
    
    live_thread = None

def manage_tracked_coins(cryptos, timeframes):
    """Menu untuk mengelola aset yang dipantau"""
    
    global tracked_coins, refresh_interval
    
    while True:
        print(colored("\n=== Kelola Aset yang Dipantau ===", 'cyan', attrs=['bold']))
        
        # Tampilkan daftar aset yang sedang dipantau
        if tracked_coins:
            print(colored("\nAset yang Sedang Dipantau:", 'yellow'))
            for i, coin in enumerate(tracked_coins, 1):
                print(f"{i}. {coin['symbol']}/USDT ({coin['timeframe']})")
        else:
            print(colored("\nBelum ada aset yang dipantau.", 'yellow'))
        
        print(colored("\nOpsi:", 'yellow'))
        print("1. Tambah Aset")
        print("2. Hapus Aset")
        print("3. Ubah Interval Refresh (saat ini:", refresh_interval, "detik)")
        print("0. Kembali")
        
        pilihan = input(colored("\nMasukkan pilihan (0-3): ", 'green'))
        
        if pilihan == '0':
            break
        
        elif pilihan == '1':
            # Tambah aset baru
            print(colored("\nPilih Crypto:", 'yellow'))
            for key, value in cryptos.items():
                print(f"{key}. {value}/USDT")
            
            coin_choice = input(colored("\nMasukkan pilihan: ", 'green'))
            
            if coin_choice not in cryptos:
                print(colored("Pilihan tidak valid!", 'red'))
                continue
            
            print(colored("\nPilih Timeframe:", 'yellow'))
            for key, value in timeframes.items():
                print(f"{key}. {value}")
            
            tf_choice = input(colored("\nMasukkan timeframe: ", 'green'))
            
            if tf_choice not in timeframes:
                print(colored("Timeframe tidak valid!", 'red'))
                continue
            
            # Cek apakah aset sudah ada dalam pemantauan
            simbol = cryptos[coin_choice]
            timeframe = timeframes[tf_choice]
            
            exists = False
            for coin in tracked_coins:
                if coin['symbol'] == simbol and coin['timeframe'] == timeframe:
                    exists = True
                    break
            
            if exists:
                print(colored(f"{simbol}/USDT dengan timeframe {timeframe} sudah dipantau!", 'yellow'))
            else:
                tracked_coins.append({
                    'symbol': simbol,
                    'timeframe': timeframe
                })
                print(colored(f"{simbol}/USDT ({timeframe}) ditambahkan ke pemantauan.", 'green'))
        
        elif pilihan == '2':
            # Hapus aset
            if not tracked_coins:
                print(colored("Tidak ada aset yang dipantau.", 'yellow'))
                continue
            
            print(colored("\nPilih Aset untuk Dihapus:", 'yellow'))
            for i, coin in enumerate(tracked_coins, 1):
                print(f"{i}. {coin['symbol']}/USDT ({coin['timeframe']})")
            
            del_choice = input(colored("\nMasukkan nomor aset: ", 'green'))
            
            try:
                del_idx = int(del_choice) - 1
                if 0 <= del_idx < len(tracked_coins):
                    removed = tracked_coins.pop(del_idx)
                    print(colored(f"{removed['symbol']}/USDT ({removed['timeframe']}) dihapus dari pemantauan.", 'green'))
                else:
                    print(colored("Nomor tidak valid!", 'red'))
            except ValueError:
                print(colored("Input tidak valid!", 'red'))
        
        elif pilihan == '3':
            # Ubah interval refresh
            try:
                new_interval = int(input(colored("\nMasukkan interval refresh baru (dalam detik, min 30): ", 'green')))
                
                if new_interval < 30:
                    print(colored("Interval minimal adalah 30 detik untuk menghindari rate limiting API.", 'yellow'))
                    new_interval = 30
                
                refresh_interval = new_interval
                print(colored(f"Interval refresh diubah menjadi {refresh_interval} detik.", 'green'))
            except ValueError:
                print(colored("Input tidak valid! Gunakan angka.", 'red'))

def main():
    print_banner()
    cryptos = {
        '1': 'BTC',
        '2': 'DOGE',
        '3': 'SHIB',
        '4': 'FLOKI',
        '5': 'ETH',
        '6': 'BNB',
        '7': 'XRP',
        '8': 'ADA',
        '9': 'SOL',
        '10': 'DOT',
        '11': 'MATIC',
        '12': 'AVAX',
        '13': 'LINK',
        '14': 'UNI',
        '15': 'PEPE',
        '16': 'MEME',
        '17': 'BONK',
        '18': 'WLD',
        '19': 'INJ',
        '20': 'SUI'
    }
    
    timeframes = {
        '1': '15m',
        '2': '30m',
        '3': '1h',
        '4': '4h'
    }
    
    while True:
        # Clear console/terminal jika tidak dalam mode live
        if not live_running:
            os.system('cls' if os.name == 'nt' else 'clear')
            print_banner()
        
        print(colored("\n=== MENU UTAMA ===", 'cyan', attrs=['bold']))
        print(colored("1. Analisis Single Crypto", 'yellow'))
        print(colored("2. Mode Live Monitoring", 'yellow'))
        print(colored("3. Kelola Aset yang Dipantau", 'yellow'))
        print(colored("0. Keluar", 'yellow'))
        
        pilihan_menu = input(colored("\nMasukkan pilihan (0-3): ", 'green'))
        
        if pilihan_menu == '0':
            # Pastikan untuk menghentikan thread live jika masih berjalan
            if live_running:
                stop_live_monitoring()
            break
        
        elif pilihan_menu == '1':
            # Analisis Single Crypto (mode asli)
            
            # Jika mode live sedang berjalan, hentikan dulu
            if live_running:
                stop_live_monitoring()
            
            # Clear console/terminal
            os.system('cls' if os.name == 'nt' else 'clear')
            print_banner()
            print(colored("\n=== Alat Analisis Crypto ===", 'cyan', attrs=['bold']))
            print(colored("Data diambil dari CoinGecko API", 'yellow'))
            print(colored("\nPilih Crypto:", 'yellow'))
            for key, value in cryptos.items():
                print(f"{key}. {value}/USDT")
            print("0. Kembali")
            
            pilihan = input(colored("\nMasukkan pilihan (0-20): ", 'green'))
            
            if pilihan == '0':
                continue
                
            if pilihan not in cryptos:
                print(colored("Pilihan tidak valid!", 'red'))
                continue
                
            print(colored("\nPilih Timeframe:", 'yellow'))
            print("1. 15 Menit")
            print("2. 30 Menit")
            print("3. 1 Jam")
            print("4. 4 Jam")
            
            pilihan_tf = input(colored("\nMasukkan timeframe (1-4): ", 'green'))
            
            if pilihan_tf not in timeframes:
                print(colored("Timeframe tidak valid!", 'red'))
                continue
            
            print(colored("\nMengambil dan menganalisis data...", 'yellow'))
            df = ambil_data_crypto(cryptos[pilihan], timeframes[pilihan_tf])
            
            if df is not None:
                analisis = analyze_indicators(df)
                keputusan = make_decision(analisis, df)
                
                print(format_output_crypto(analisis, cryptos[pilihan], timeframes[pilihan_tf], df))
                
                print(colored("\n=== Keputusan Trading ===", 'magenta', attrs=['bold']))
                warna_aksi = {
                    'STRONG_BUY': 'green',
                    'BUY': 'green',
                    'HOLD': 'yellow',
                    'SELL': 'red',
                    'STRONG_SELL': 'red'
                }
                
                print(f"Aksi: {colored(keputusan['action'], warna_aksi.get(keputusan['action'], 'white'))}")
                print(f"Keyakinan: {colored(f'{keputusan['confidence']:.1f}%', 'cyan')}")
                print(f"Level Resiko: {colored(keputusan['risk_level'], 'yellow')}")
                
                print(colored("\nAlasan:", 'yellow'))
                for alasan in keputusan['reason']:
                    print(colored(f"• {alasan}", 'white'))
                
                # Tambahkan opsi untuk menambahkan ke pemantauan live
                print(colored("\nOpsi Tambahan:", 'cyan'))
                print("1. Tambahkan ke pemantauan live")
                print("0. Kembali ke menu")
                
                pilihan_tambahan = input(colored("\nMasukkan pilihan: ", 'green'))
                
                if pilihan_tambahan == '1':
                    # Cek apakah sudah ada di tracked_coins
                    simbol = cryptos[pilihan]
                    timeframe = timeframes[pilihan_tf]
                    
                    exists = False
                    for coin in tracked_coins:
                        if coin['symbol'] == simbol and coin['timeframe'] == timeframe:
                            exists = True
                            break
                    
                    if exists:
                        print(colored(f"{simbol}/USDT dengan timeframe {timeframe} sudah dipantau!", 'yellow'))
                    else:
                        tracked_coins.append({
                            'symbol': simbol,
                            'timeframe': timeframe
                        })
                        print(colored(f"{simbol}/USDT ({timeframe}) ditambahkan ke pemantauan.", 'green'))
            else:
                print(colored("Gagal mengambil data. Silakan coba lagi.", 'red'))
            
            input(colored("\nTekan Enter untuk melanjutkan...", 'green'))
        
        elif pilihan_menu == '2':
            # Mode Live Monitoring
            
            if not tracked_coins:
                print(colored("Tidak ada aset yang dipantau. Silakan tambahkan minimal 1 aset terlebih dahulu.", 'red'))
                input(colored("\nTekan Enter untuk melanjutkan...", 'green'))
                continue
            
            if live_running:
                print(colored("Mode live monitoring sudah berjalan!", 'yellow'))
                input(colored("\nTekan Enter untuk melanjutkan...", 'green'))
                continue
            
            start_live_monitoring()
            
            # Menunggu sampai user menekan tombol untuk kembali
            input(colored("\nMode live berjalan di background. Tekan Enter untuk kembali ke menu utama...", 'green'))
        
        elif pilihan_menu == '3':
            # Kelola Aset yang Dipantau
            
            # Jika mode live sedang berjalan, hentikan dulu
            if live_running:
                stop_live_monitoring()
            
            manage_tracked_coins(cryptos, timeframes)

def simpan_data_konfigurasi():
    """Simpan data konfigurasi dan aset yang dipantau"""
    try:
        config = {
            'tracked_coins': tracked_coins,
            'refresh_interval': refresh_interval
        }
        
        with open('tradingmetrics_config.json', 'w') as f:
            json.dump(config, f)
        
        logger.info("Konfigurasi berhasil disimpan")
    except Exception as e:
        logger.error(f"Error menyimpan konfigurasi: {str(e)}")

def load_data_konfigurasi():
    """Muat data konfigurasi dari file"""
    global tracked_coins, refresh_interval
    
    try:
        if os.path.exists('tradingmetrics_config.json'):
            with open('tradingmetrics_config.json', 'r') as f:
                config = json.load(f)
            
            tracked_coins = config.get('tracked_coins', [])
            refresh_interval = config.get('refresh_interval', 60)
            
            logger.info(f"Konfigurasi dimuat: {len(tracked_coins)} aset, interval {refresh_interval}s")
        else:
            logger.info("File konfigurasi tidak ditemukan, menggunakan nilai default")
    except Exception as e:
        logger.error(f"Error memuat konfigurasi: {str(e)}")

if __name__ == "__main__":
    try:
        # Muat konfigurasi sebelum memulai
        load_data_konfigurasi()
        
        # Jalankan program utama
        main()
        
        # Simpan konfigurasi saat keluar
        simpan_data_konfigurasi()
        
    except KeyboardInterrupt:
        print(colored("\nProgram dihentikan oleh user.", 'yellow'))
        
        # Pastikan untuk menghentikan thread live jika masih berjalan
        if live_running:
            stop_live_monitoring()
        
        # Simpan konfigurasi saat keluar dengan Ctrl+C
        simpan_data_konfigurasi()
        
    except Exception as e:
        logger.error(f"Error tidak terduga: {str(e)}")
        print(colored(f"\nTerjadi kesalahan: {str(e)}", 'red'))