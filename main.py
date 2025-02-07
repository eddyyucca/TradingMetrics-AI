import requests
import pandas as pd
import numpy as np
from datetime import datetime
from termcolor import colored
import colorama
from analysis import analyze_indicators
from decision import make_decision

# Inisialisasi colorama
colorama.init()

def ambil_data_crypto(simbol, interval="15m", limit=100):
    """Mengambil data crypto dari Binance"""
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
        print(colored(f"Error mengambil data: {str(e)}", 'red'))
        return None

def hitung_level_resiko(df, harga_sekarang, modal_awal=150000):
    """Menghitung level manajemen resiko"""
    volatilitas = df['close'].pct_change().std() * 100
    atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    rata_atr = atr.mean()
    
    # Level resiko berdasarkan volatilitas
    if volatilitas > 5:  # Volatilitas tinggi
        persen_sl = 0.02  # 2%
        rasio_tp = 3      # 1:3 risk:reward
    elif volatilitas > 3:  # Volatilitas sedang
        persen_sl = 0.015  # 1.5%
        rasio_tp = 2.5    # 1:2.5 risk:reward
    else:  # Volatilitas rendah
        persen_sl = 0.01  # 1%
        rasio_tp = 2      # 1:2 risk:reward

    # Hitung zona entry
    support_kuat = df['low'].tail(20).min()
    support_lemah = harga_sekarang - rata_atr
    resistance_lemah = harga_sekarang + rata_atr
    resistance_kuat = df['high'].tail(20).max()

    # Hitung ukuran posisi
    jumlah_resiko = modal_awal * 0.02  # maksimal resiko 2%
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

def format_output_crypto(analisis, simbol, timeframe, df):
    """Format hasil analisis dengan warna"""
    harga_sekarang = analisis['current_price']
    level_resiko = hitung_level_resiko(df, harga_sekarang)
    
    output = colored(f"\n=== Analisis {simbol}/USDT ===\n", 'cyan', attrs=['bold'])
    output += colored(f"Timeframe: {timeframe}\n", 'cyan')
    output += colored(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", 'cyan')
    
    # Informasi harga
    output += colored("\nInformasi Harga:\n", 'yellow', attrs=['bold'])
    output += f"Harga Sekarang: {colored(f'${harga_sekarang:.8f}', 'green')}\n"
    output += f"Volatilitas: {colored(f'{level_resiko['volatilitas']:.2f}%', 'yellow')}\n"
    
    # Zona entry
    output += colored("\nZona Entry yang Disarankan:\n", 'yellow', attrs=['bold'])
    entry = level_resiko['zona_entry']
    output += f"Beli Kuat:    {colored(f'${entry['beli_kuat']:.8f}', 'green')}\n"
    output += f"Beli Lemah:   {colored(f'${entry['beli_lemah']:.8f}', 'green')}\n"
    output += f"Jual Lemah:   {colored(f'${entry['jual_lemah']:.8f}', 'red')}\n"
    output += f"Jual Kuat:    {colored(f'${entry['jual_kuat']:.8f}', 'red')}\n"
    
    # Level Stop Loss
    output += colored("\nLevel Stop Loss:\n", 'yellow', attrs=['bold'])
    sl = level_resiko['stop_loss']
    output += f"Ketat:    {colored(f'${sl['ketat']:.8f}', 'red')}\n"
    output += f"Sedang:   {colored(f'${sl['sedang']:.8f}', 'red')}\n"
    output += f"Longgar:  {colored(f'${sl['longgar']:.8f}', 'red')}\n"
    
    # Level Take Profit
    output += colored("\nLevel Take Profit:\n", 'yellow', attrs=['bold'])
    tp = level_resiko['take_profit']
    output += f"Aman:     {colored(f'${tp['aman']:.8f}', 'green')}\n"
    output += f"Sedang:   {colored(f'${tp['sedang']:.8f}', 'green')}\n"
    output += f"Agresif:  {colored(f'${tp['agresif']:.8f}', 'green')}\n"
    
    # Ukuran posisi
    pos = level_resiko['ukuran_posisi']
    output += colored("\nUkuran Posisi yang Disarankan (IDR):\n", 'yellow', attrs=['bold'])
    output += f"Minimum:     {colored(f'Rp {pos['minimum']:,.0f}', 'cyan')}\n"
    output += f"Disarankan:  {colored(f'Rp {pos['disarankan']:,.0f}', 'cyan')}\n"
    output += f"Maksimum:    {colored(f'Rp {pos['maksimum']:,.0f}', 'cyan')}\n"
    
    # Kekuatan sinyal
    output += colored("\nKekuatan Sinyal:\n", 'yellow', attrs=['bold'])
    warna_beli = 'green' if analisis['total_buy'] > 60 else 'white'
    warna_jual = 'red' if analisis['total_sell'] > 60 else 'white'
    
    output += f"BELI: {colored('='* int(analisis['total_buy']/2), warna_beli)} {colored(f'{analisis['total_buy']}%', warna_beli)}\n"
    output += f"JUAL: {colored('='* int(analisis['total_sell']/2), warna_jual)} {colored(f'{analisis['total_sell']}%', warna_jual)}\n"
    
    # Peringatan resiko
    output += colored("\nCatatan Manajemen Resiko:\n", 'red', attrs=['bold'])
    if level_resiko['volatilitas'] > 5:
        output += "• VOLATILITAS TINGGI - Gunakan stop loss yang lebih longgar!\n"
    output += "• Selalu gunakan stop loss\n"
    output += "• Jangan resiko lebih dari 2% modal\n"
    output += "• Pertimbangkan untuk masuk/keluar secara bertahap\n"
    
    return output

def main():
    cryptos = {
        '1': 'BTC',
        '2': 'DOGE',
        '3': 'SHIB',
        '4': 'FLOKI'
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
        print("1. BTC/USDT")
        print("2. DOGE/USDT")
        print("3. SHIB/USDT")
        print("4. FLOKI/USDT")
        print("0. Keluar")
        
        pilihan = input(colored("\nMasukkan pilihan (0-4): ", 'green'))
        
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