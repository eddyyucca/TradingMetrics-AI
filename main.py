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

colorama.init()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_banner():
    banner = colored("""
╔═══════════════════════════════════════════════════════════╗
║           TradingMetrics-AI - v1.0.0                      ║
╠═══════════════════════════════════════════════════════════╣
║ Developed by Eddy Adha Saputra                           ║
║ Alat Analisis Teknikal Trading Cryptocurrency            ║
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
        time.sleep(0.5)  # Simulasi proses loading
    
    # Tampilkan waktu dan informasi tambahan
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n")
    print(colored(f"Waktu Inisialisasi: {current_time}", 'yellow'))
    logger.info("TradingMetrics-AI diinisialisasi")
def ambil_data_crypto(simbol, interval="15m", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": f"{simbol}USDT",
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                       'taker_buy_quote', 'ignore'])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
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
        print(colored("\n=== Alat Analisis Crypto ===", 'cyan', attrs=['bold']))
        print(colored("\nPilih Crypto:", 'yellow'))
        for key, value in cryptos.items():
            print(f"{key}. {value}/USDT")
        print("0. Keluar")
        
        pilihan = input(colored("\nMasukkan pilihan (0-20): ", 'green'))
        
        if pilihan == '0':
            break
            
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
      
# Dengan ini:
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
        else:
            print(colored("Gagal mengambil data. Silakan coba lagi.", 'red'))
        
        input(colored("\nTekan Enter untuk melanjutkan...", 'green'))

if __name__ == "__main__":
    main()