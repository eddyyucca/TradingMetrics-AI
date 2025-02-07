import requests
import pandas as pd
from datetime import datetime
from analysis import analyze_indicators, format_analysis_output

def fetch_data(symbol="EURUSDT", interval="1m", limit=100):
    """Fetch market data"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def main():
    timeframes = {
        '1': '1m',
        '2': '5m',
        '3': '15m',
        '4': '30m',
        '5': '1h',
        '6': '4h'
    }
    
    pairs = {
        '1': 'EURUSDT',
        '2': 'BTCUSDT',
        '3': 'GBPUSDT'
    }
    
    while True:
        print("\n=== Market Analysis Tool ===")
        print("\nPilih Pair:")
        print("1. EUR/USD")
        print("2. BTC/USD")
        print("3. GBP/USD")
        print("0. Keluar")
        
        pair_choice = input("\nMasukkan pilihan pair (0-3): ")
        
        if pair_choice == '0':
            break
            
        if pair_choice not in pairs:
            print("Pilihan pair tidak valid!")
            continue
            
        print("\nPilih Timeframe:")
        print("1. 1 Menit")
        print("2. 5 Menit")
        print("3. 15 Menit")
        print("4. 30 Menit")
        print("5. 1 Jam")
        print("6. 4 Jam")
        
        tf_choice = input("\nMasukkan pilihan timeframe (1-6): ")
        
        if tf_choice not in timeframes:
            print("Pilihan timeframe tidak valid!")
            continue
        
        print("\nMengambil dan menganalisis data...")
        df = fetch_data(pairs[pair_choice], timeframes[tf_choice])
        
        if df is not None:
            analysis = analyze_indicators(df)
            
            print(f"\n=== Hasil Analisis {pairs[pair_choice]} ===")
            print(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Timeframe: {timeframes[tf_choice]}")
            
            print(format_analysis_output(analysis))
            
            print("\nCatatan:")
            print("* Ini adalah analisis teknikal, bukan rekomendasi trading")
            print("* Selalu gunakan money management dan risk management")
            print("* Past performance tidak menjamin hasil di masa depan")
        else:
            print("Gagal mengambil data. Silakan coba lagi.")
        
        input("\nTekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    main()