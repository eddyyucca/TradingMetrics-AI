# market_context.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_market_phases(df, short_period=10, long_period=50):
    """
    Mendeteksi fase pasar: uptrend, downtrend, ranging atau choppy
    """
    # Calculate EMAs
    df['short_ema'] = df['close'].ewm(span=short_period).mean()
    df['long_ema'] = df['close'].ewm(span=long_period).mean()
    
    # Calculate EMA slope (rate of change)
    df['short_ema_slope'] = df['short_ema'].pct_change(5) * 100
    df['long_ema_slope'] = df['long_ema'].pct_change(10) * 100
    
    # Current values
    current_close = df['close'].iloc[-1]
    current_short_ema = df['short_ema'].iloc[-1]
    current_long_ema = df['long_ema'].iloc[-1]
    short_ema_slope = df['short_ema_slope'].iloc[-1]
    long_ema_slope = df['long_ema_slope'].iloc[-1]
    
    # Calculate average true range for volatility
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Current ATR as percentage of price
    atr_percent = (df['atr'].iloc[-1] / current_close) * 100
    
    # Calculate price range as percentage (identifies ranging market)
    highest_high = df['high'].rolling(window=20).max().iloc[-1]
    lowest_low = df['low'].rolling(window=20).min().iloc[-1]
    price_range_percent = ((highest_high - lowest_low) / lowest_low) * 100
    
    # Determine market phase
    if current_short_ema > current_long_ema and short_ema_slope > 0 and long_ema_slope > 0:
        phase = "uptrend"
        strength = min(70 + (short_ema_slope * 2), 100)  # Higher slope = stronger trend
        description = "Strong uptrend detected, momentum is positive"
    elif current_short_ema < current_long_ema and short_ema_slope < 0 and long_ema_slope < 0:
        phase = "downtrend"
        strength = min(70 + (abs(short_ema_slope) * 2), 100)  # Higher negative slope = stronger trend
        description = "Strong downtrend detected, momentum is negative"
    elif price_range_percent < 8 and atr_percent < 3:
        phase = "ranging"
        strength = 60  # Moderate confidence in ranging market
        description = "Price is moving sideways in a tight range"
    elif (abs(short_ema_slope) > 2 * abs(long_ema_slope) or 
          (abs(short_ema_slope - long_ema_slope) > 1 and atr_percent > 4)):
        phase = "choppy"
        strength = 50 + min(atr_percent * 5, 30)  # Higher volatility = more choppy
        description = "Market is volatile and choppy, showing indecision"
    elif current_short_ema > current_long_ema:
        phase = "weak_uptrend"
        strength = 40 + (short_ema_slope * 5)
        description = "Weak uptrend, exercise caution"
    else:
        phase = "weak_downtrend"
        strength = 40 + (abs(short_ema_slope) * 5)
        description = "Weak downtrend, exercise caution"
    
    return {
        "phase": phase,
        "strength": strength,
        "description": description,
        "atr_percent": atr_percent,
        "price_range_percent": price_range_percent,
        "short_slope": short_ema_slope,
        "long_slope": long_ema_slope
    }

def detect_support_resistance(df, lookback=100, threshold_percent=1.0):
    """
    Deteksi level support dan resistance penting
    lookback: jumlah candle untuk dianalisis
    threshold_percent: persentase jarak minimum antara level
    """
    # Use a subset of data based on lookback
    data = df.iloc[-min(lookback, len(df)):]
    
    # Get highs and lows
    highs = data['high'].values
    lows = data['low'].values
    
    # Identify local maxima and minima
    resistance_levels = []
    support_levels = []
    
    # Function to check if a point is a local maximum/minimum
    def is_local_max(idx, values, window=5):
        if idx < window or idx >= len(values) - window:
            return False
        left = values[idx - window:idx]
        right = values[idx + 1:idx + window + 1]
        return values[idx] > max(left) and values[idx] > max(right)
    
    def is_local_min(idx, values, window=5):
        if idx < window or idx >= len(values) - window:
            return False
        left = values[idx - window:idx]
        right = values[idx + 1:idx + window + 1]
        return values[idx] < min(left) and values[idx] < min(right)
    
    # Find local maxima and minima
    for i in range(len(highs)):
        if is_local_max(i, highs):
            resistance_levels.append(highs[i])
        
        if is_local_min(i, lows):
            support_levels.append(lows[i])
    
    # Function to cluster nearby levels
    def cluster_levels(levels, threshold):
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Initialize clusters
        clusters = [[sorted_levels[0]]]
        
        # Cluster nearby levels
        for level in sorted_levels[1:]:
            if level - clusters[-1][-1] <= threshold:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        
        # Calculate average for each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    # Current price for threshold calculation
    current_price = df['close'].iloc[-1]
    threshold = current_price * (threshold_percent / 100)
    
    # Cluster levels
    support_clusters = cluster_levels(support_levels, threshold)
    resistance_clusters = cluster_levels(resistance_levels, threshold)
    
    # Filter relevant levels (close to current price)
    current_price = df['close'].iloc[-1]
    relevant_support = [lvl for lvl in support_clusters if lvl < current_price]
    relevant_resistance = [lvl for lvl in resistance_clusters if lvl > current_price]
    
    # Calculate strength based on proximity and number of tests
    def calculate_strength(level, data):
        # Proximity factor: closer = stronger
        price_diff = abs(level - current_price) / current_price
        proximity_factor = max(0, 1 - (price_diff * 10))  # 0-1 range
        
        # Count tests (price approaching within 0.5% of level)
        test_threshold = level * 0.005
        test_count = sum(1 for low in data['low'] if abs(low - level) < test_threshold) + \
                    sum(1 for high in data['high'] if abs(high - level) < test_threshold)
        
        # Test factor: more tests = stronger
        test_factor = min(1, test_count / 5)  # Cap at 5 tests
        
        return (proximity_factor * 0.6 + test_factor * 0.4) * 100  # Scale to 0-100
    
    # Sort by proximity to current price
    relevant_support.sort(key=lambda x: current_price - x)
    relevant_resistance.sort(key=lambda x: x - current_price)
    
    # Format results
    supports = [{"level": level, "strength": calculate_strength(level, data)} 
               for level in relevant_support[:3]]  # Top 3 support levels
    
    resistances = [{"level": level, "strength": calculate_strength(level, data)} 
                  for level in relevant_resistance[:3]]  # Top 3 resistance levels
    
    return {
        "supports": supports,
        "resistances": resistances
    }

def analyze_market_context(df):
    """
    Analisis lengkap konteks pasar untuk pengambilan keputusan
    """
    # Get overall market phase
    market_phase = calculate_market_phases(df)
    
    # Detect support/resistance levels
    levels = detect_support_resistance(df)
    
    # Calculate proximity to nearest support/resistance
    current_price = df['close'].iloc[-1]
    nearest_support = levels['supports'][0]['level'] if levels['supports'] else None
    nearest_resistance = levels['resistances'][0]['level'] if levels['resistances'] else None
    
    # Calculate proximity as percentage
    support_proximity = ((current_price - nearest_support) / current_price * 100) if nearest_support else None
    resistance_proximity = ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
    
    # Calculate relative strength compared to recent highs/lows
    high_20d = df['high'].rolling(window=20).max().iloc[-1]
    low_20d = df['low'].rolling(window=20).min().iloc[-1]
    price_position = (current_price - low_20d) / (high_20d - low_20d) if (high_20d - low_20d) > 0 else 0.5
    
    # Determine buy/sell signals based on market context
    buy_strength = 0
    sell_strength = 0
    context_signals = []
    
    # 1. Market phase signals
    if market_phase['phase'] in ['uptrend', 'weak_uptrend']:
        buy_strength += market_phase['strength'] * 0.3  # 30% weight for trend
        context_signals.append(f"Market in {market_phase['phase']}: {market_phase['description']}")
    elif market_phase['phase'] in ['downtrend', 'weak_downtrend']:
        sell_strength += market_phase['strength'] * 0.3  # 30% weight for trend
        context_signals.append(f"Market in {market_phase['phase']}: {market_phase['description']}")
    else:
        context_signals.append(f"Market in {market_phase['phase']}: {market_phase['description']}")
    
    # 2. Support/Resistance signals
    if support_proximity is not None and support_proximity < 3:
        buy_strength += (30 - support_proximity * 10)  # Closer to support = stronger buy
        context_signals.append(f"Price near strong support level (${nearest_support:.2f})")
    
    if resistance_proximity is not None and resistance_proximity < 3:
        sell_strength += (30 - resistance_proximity * 10)  # Closer to resistance = stronger sell
        context_signals.append(f"Price near strong resistance level (${nearest_resistance:.2f})")
    
    # 3. Price position signals
    if price_position < 0.2:  # Near 20-day low
        buy_strength += 15
        context_signals.append("Price near 20-day low, potential oversold condition")
    elif price_position > 0.8:  # Near 20-day high
        sell_strength += 15
        context_signals.append("Price near 20-day high, potential overbought condition")
    
    # 4. Volatility signals
    if market_phase['atr_percent'] > 5:
        context_signals.append(f"High volatility detected ({market_phase['atr_percent']:.2f}%), exercise caution")
        # Reduce both signals in high volatility
        buy_strength *= 0.8
        sell_strength *= 0.8
    
    return {
        "market_phase": market_phase,
        "support_resistance": levels,
        "price_position": price_position,
        "context_signals": context_signals,
        "buy_strength": min(buy_strength, 100),  # Cap at 100
        "sell_strength": min(sell_strength, 100),  # Cap at 100
        "volatility": market_phase['atr_percent']
    }