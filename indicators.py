import numpy as np
import pandas as pd

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
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

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(df, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k_line = ((df['close'] - low_min) / (high_max - low_min)) * 100
    d_line = k_line.rolling(window=d_period).mean()
    
    return k_line, d_line

def calculate_bollinger_bands(prices, period=20, std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    return upper_band, sma, lower_band

def calculate_adx(df, period=14):
    """Calculate ADX"""
    df = df.copy()
    df['TR'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)),
                                  abs(df['low'] - df['close'].shift(1))))
    df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > 
                        (df['low'].shift(1) - df['low']),
                        np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['-DM'] = np.where((df['low'].shift(1) - df['low']) > 
                        (df['high'] - df['high'].shift(1)),
                        np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    df['TR14'] = df['TR'].rolling(window=period).mean()
    df['+DI14'] = (df['+DM'].rolling(window=period).mean() / df['TR14']) * 100
    df['-DI14'] = (df['-DM'].rolling(window=period).mean() / df['TR14']) * 100
    df['DX'] = abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14']) * 100
    adx = df['DX'].rolling(window=period).mean()
    
    return adx, df['+DI14'], df['-DI14']