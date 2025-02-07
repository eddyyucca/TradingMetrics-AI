import numpy as np
from indicators import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_bollinger_bands, calculate_adx
)

def analyze_indicators(df):
    """Analyze all indicators and return percentage signals"""
    signals = {}
    
    # RSI Analysis (20%)
    rsi = calculate_rsi(df['close'].values)
    current_rsi = rsi[-1]
    signals['RSI'] = {
        'value': round(current_rsi, 2),
        'buy_strength': 20 if current_rsi < 25 else (10 if current_rsi < 40 else 0),
        'sell_strength': 20 if current_rsi > 75 else (10 if current_rsi > 60 else 0)
    }
    
    # MACD Analysis (20%)
    macd, signal, hist = calculate_macd(df['close'])
    current_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    signals['MACD'] = {
        'value': round(current_hist, 8),
        'buy_strength': 20 if current_hist > 0 and current_hist > prev_hist else 0,
        'sell_strength': 20 if current_hist < 0 and current_hist < prev_hist else 0
    }
    
    # Stochastic Analysis (15%)
    k_line, d_line = calculate_stochastic(df)
    current_k = k_line.iloc[-1]
    current_d = d_line.iloc[-1]
    signals['Stochastic'] = {
        'value': round(current_k, 2),
        'buy_strength': 15 if current_k < 20 and current_k > current_d else 0,
        'sell_strength': 15 if current_k > 80 and current_k < current_d else 0
    }
    
    # Bollinger Bands Analysis (25%)
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    current_price = df['close'].iloc[-1]
    bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100
    signals['Bollinger'] = {
        'value': round(bb_position, 2),
        'buy_strength': 25 if bb_position < 20 else (15 if bb_position < 40 else 0),
        'sell_strength': 25 if bb_position > 80 else (15 if bb_position > 60 else 0)
    }
    
    # ADX Analysis (20%)
    adx, plus_di, minus_di = calculate_adx(df)
    current_adx = adx.iloc[-1]
    signals['ADX'] = {
        'value': round(current_adx, 2),
        'buy_strength': 20 if current_adx > 25 and plus_di.iloc[-1] > minus_di.iloc[-1] else 0,
        'sell_strength': 20 if current_adx > 25 and plus_di.iloc[-1] < minus_di.iloc[-1] else 0
    }
    
    # Calculate total signals
    total_buy = sum(indicator['buy_strength'] for indicator in signals.values())
    total_sell = sum(indicator['sell_strength'] for indicator in signals.values())
    
    return {
        'indicators': signals,
        'total_buy': total_buy,
        'total_sell': total_sell,
        'current_price': round(current_price, 8)
    }