import numpy as np
from indicators import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_bollinger_bands, calculate_adx
)

def analyze_indicators(df):
    """Analyze all indicators and return percentage signals with optimizations for cryptocurrency markets"""
    signals = {}
    
    # RSI Analysis (20%)
    # Cryptocurrency markets tend to have wider RSI ranges compared to traditional markets
    rsi = calculate_rsi(df['close'].values)
    current_rsi = rsi[-1]
    rsi_trend = 'up' if current_rsi > rsi[-2] else 'down'
    signals['RSI'] = {
        'value': round(current_rsi, 2),
        'trend': rsi_trend,
        'buy_strength': 20 if current_rsi < 30 else (15 if current_rsi < 40 else (5 if rsi_trend == 'up' and current_rsi < 50 else 0)),
        'sell_strength': 20 if current_rsi > 70 else (15 if current_rsi > 60 else (5 if rsi_trend == 'down' and current_rsi > 50 else 0))
    }
    
    # MACD Analysis (20%)
    # Added signal line crossover detection for stronger signals
    macd, signal, hist = calculate_macd(df['close'])
    current_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    macd_cross = (macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1])
    macd_cross_down = (macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1])
    
    signals['MACD'] = {
        'value': round(current_hist, 8),
        'buy_strength': 20 if macd_cross else (15 if current_hist > 0 and current_hist > prev_hist else (5 if current_hist > 0 else 0)),
        'sell_strength': 20 if macd_cross_down else (15 if current_hist < 0 and current_hist < prev_hist else (5 if current_hist < 0 else 0))
    }
    
    # Stochastic Analysis (15%)
    # Optimized for crypto's higher volatility
    k_line, d_line = calculate_stochastic(df)
    current_k = k_line.iloc[-1]
    current_d = d_line.iloc[-1]
    prev_k = k_line.iloc[-2]
    
    # Check for stochastic crossover (more reliable signal)
    stoch_cross_up = (k_line.iloc[-2] < d_line.iloc[-2] and current_k > current_d)
    stoch_cross_down = (k_line.iloc[-2] > d_line.iloc[-2] and current_k < current_d)
    
    signals['Stochastic'] = {
        'value': round(current_k, 2),
        'buy_strength': 15 if stoch_cross_up else (10 if current_k < 20 else (5 if current_k > prev_k and current_k < 40 else 0)),
        'sell_strength': 15 if stoch_cross_down else (10 if current_k > 80 else (5 if current_k < prev_k and current_k > 60 else 0))
    }
    
    # Bollinger Bands Analysis (25%)
    # Crypto often shows strong momentum after touching bands
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100
    
    # Check for band touches and bounces
    lower_band_touch = any(df['low'].iloc[-3:] <= lower.iloc[-3:])
    upper_band_touch = any(df['high'].iloc[-3:] >= upper.iloc[-3:])
    price_momentum = current_price > prev_price
    
    signals['Bollinger'] = {
        'value': round(bb_position, 2),
        'buy_strength': 25 if lower_band_touch and price_momentum else (20 if bb_position < 10 else (15 if bb_position < 30 else 0)),
        'sell_strength': 25 if upper_band_touch and not price_momentum else (20 if bb_position > 90 else (15 if bb_position > 70 else 0))
    }
    
    # ADX Analysis (15%)
    adx, plus_di, minus_di = calculate_adx(df)
    current_adx = adx.iloc[-1]
    di_cross_up = (plus_di.iloc[-2] < minus_di.iloc[-2] and plus_di.iloc[-1] > minus_di.iloc[-1])
    di_cross_down = (plus_di.iloc[-2] > minus_di.iloc[-2] and plus_di.iloc[-1] < minus_di.iloc[-1])
    
    signals['ADX'] = {
        'value': round(current_adx, 2),
        'buy_strength': 15 if di_cross_up else (10 if current_adx > 25 and plus_di.iloc[-1] > minus_di.iloc[-1] else 0),
        'sell_strength': 15 if di_cross_down else (10 if current_adx > 25 and plus_di.iloc[-1] < minus_di.iloc[-1] else 0)
    }
    
    # Volume Analysis (10%) - New indicator specific for crypto
    volume = df['volume']
    avg_volume = volume.rolling(window=20).mean()
    current_volume = volume.iloc[-1]
    volume_ratio = current_volume / avg_volume.iloc[-1] if not np.isnan(avg_volume.iloc[-1]) else 1.0
    
    # Volume with price direction
    price_up = current_price > df['close'].iloc[-2]
    
    signals['Volume'] = {
        'value': round(volume_ratio, 2),
        'buy_strength': 10 if volume_ratio > 1.5 and price_up else (5 if volume_ratio > 1.2 and price_up else 0),
        'sell_strength': 10 if volume_ratio > 1.5 and not price_up else (5 if volume_ratio > 1.2 and not price_up else 0)
    }
    
    # MA Cross Analysis (10%) - New indicator specific for crypto trends
    short_ma = df['close'].rolling(window=9).mean()
    long_ma = df['close'].rolling(window=21).mean()
    
    ma_cross_up = (short_ma.iloc[-2] <= long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1])
    ma_cross_down = (short_ma.iloc[-2] >= long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1])
    
    signals['MA_Cross'] = {
        'value': round(short_ma.iloc[-1] - long_ma.iloc[-1], 8),
        'buy_strength': 10 if ma_cross_up else (5 if short_ma.iloc[-1] > long_ma.iloc[-1] else 0),
        'sell_strength': 10 if ma_cross_down else (5 if short_ma.iloc[-1] < long_ma.iloc[-1] else 0)
    }
    
    # Calculate total signals - adjusted weights to accommodate new indicators
    # Each indicator's max contribution is adjusted to total 100%
    total_buy = sum(indicator['buy_strength'] for indicator in signals.values())
    total_sell = sum(indicator['sell_strength'] for indicator in signals.values())
    
    # Add some metadata for improved display and decision-making
    return {
        'indicators': signals,
        'total_buy': min(total_buy, 100),  # Cap at 100%
        'total_sell': min(total_sell, 100),  # Cap at 100%
        'current_price': round(current_price, 8),
        'price_change_24h': round((current_price / df['close'].iloc[-24 if len(df) > 24 else 0] - 1) * 100, 2) if len(df) > 1 else 0,
        'volume_change_24h': round((current_volume / volume.iloc[-24 if len(df) > 24 else 0] - 1) * 100, 2) if len(df) > 1 else 0
    }