# test_imports.py
try:
    import pandas
    import numpy
    import requests
    import pycoingecko
    import tabulate
    import termcolor
    import colorama
    import dateutil
    import pytz
    
    print("Semua modul berhasil diimpor!")
    
    # Verifikasi pycoingecko
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    btc_price = cg.get_price(ids='bitcoin', vs_currencies='usd')
    print(f"Test API CoinGecko: Harga Bitcoin saat ini: ${btc_price['bitcoin']['usd']}")
    
    print("Semua dependensi terinstal dengan benar dan berfungsi!")
    
except ImportError as e:
    print(f"Error mengimpor modul: {e}")
    print("Silakan coba instal ulang dependensi yang bermasalah.")
except Exception as e:
    print(f"Error lain dalam pengujian: {e}")