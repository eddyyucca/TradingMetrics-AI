import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def calculate_rsi(prices, period=14):
    """Menghitung RSI"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def fetch_eurusd_data(interval="1m", limit=100):
    """Mengambil data EUR/USD dari Binance"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "EURUSDT",  # Changed to EURUSDT
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
        
        # Convert price and volume columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def analyze_signals(df):
    """Analisis sinyal trading"""
    # Hitung indikator
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA25'] = df['close'].rolling(window=25).mean()
    df['RSI'] = calculate_rsi(df['close'].values)
    df['momentum'] = df['close'].diff()
    
    # Ambil data terakhir
    current_price = df['close'].iloc[-1]
    current_ma7 = df['MA7'].iloc[-1]
    current_ma25 = df['MA25'].iloc[-1]
    current_rsi = df['RSI'].iloc[-1]
    current_momentum = df['momentum'].iloc[-1]
    
    # Signal strength dihitung dari kombinasi indikator
    buy_strength = 0
    sell_strength = 0
    
    # MA Crossover (30%)
    if current_ma7 > current_ma25:
        buy_strength += 30
    else:
        sell_strength += 30
        
    # RSI (25%)
    if current_rsi < 30:
        buy_strength += 25
    elif current_rsi > 70:
        sell_strength += 25
        
    # Momentum (25%)
    if current_momentum > 0:
        buy_strength += 25
    else:
        sell_strength += 25
        
    # Price to MA (20%)
    if current_price > current_ma7:
        buy_strength += 20
    else:
        sell_strength += 20
    
    return {
        'price': round(current_price, 4),  # Rounded to 4 decimals for forex
        'ma7': round(current_ma7, 4),
        'ma25': round(current_ma25, 4),
        'rsi': round(current_rsi, 2),
        'buy_percentage': buy_strength,
        'sell_percentage': sell_strength
    }

def main():
    timeframes = {
        '1': '1m',
        '2': '5m',
        '3': '15m',
        '4': '30m',
        '5': '1h',
        '6': '4h'
    }
    
    while True:
        print("\n=== EUR/USD Signal Analysis ===")
        print("\nPilih Timeframe:")
        print("1. 1 Menit")
        print("2. 5 Menit")
        print("3. 15 Menit")
        print("4. 30 Menit")
        print("5. 1 Jam")
        print("6. 4 Jam")
        print("0. Keluar")
        
        choice = input("\nMasukkan pilihan (0-6): ")
        
        if choice == '0':
            break
            
        if choice not in timeframes:
            print("Pilihan tidak valid!")
            continue
        
        # Ambil dan analisis data
        print("\nMenganalisis data...")
        df = fetch_eurusd_data(timeframes[choice], 100)
        
        if df is not None:
            analysis = analyze_signals(df)
            
            print("\n=== Hasil Analisis EUR/USD ===")
            print(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Timeframe: {timeframes[choice]}")
            print(f"\nHarga EUR/USD: ${analysis['price']}")
            print(f"RSI: {analysis['rsi']}")
            print("\nSignal Strength:")
            print(f"BUY:  {'=' * int(analysis['buy_percentage']/2)} {analysis['buy_percentage']}%")
            print(f"SELL: {'=' * int(analysis['sell_percentage']/2)} {analysis['sell_percentage']}%")
            
            # Tambahan detail sinyal
            print("\nDetail Sinyal:")
            print(f"MA7: ${analysis['ma7']}")
            print(f"MA25: ${analysis['ma25']}")
            
            print("\nCatatan:")
            print("* Ini adalah analisis teknikal, bukan rekomendasi trading")
            print("* Selalu gunakan money management dan risk management")
            print("* Past performance tidak menjamin hasil di masa depan")
            
        else:
            print("Gagal mengambil data. Silakan coba lagi.")
        
        input("\nTekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    main()