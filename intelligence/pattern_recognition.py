# pattern_recognition.py

import numpy as np
import pandas as pd

def detect_doji(df, tolerance=0.05):
    """
    Deteksi pola Doji (open dan close hampir sama)
    tolerance: persentase perbedaan yang diperbolehkan antara open dan close
    """
    results = []
    
    for i in range(len(df) - 1, max(len(df) - 5, 0) - 1, -1):
        row = df.iloc[i]
        body_size = abs(row['close'] - row['open'])
        candle_range = row['high'] - row['low']
        
        if candle_range == 0:  # Prevent division by zero
            continue
            
        body_percent = body_size / candle_range
        
        # Doji has very small body compared to total range
        if body_percent <= tolerance:
            results.append({
                'type': 'doji',
                'index': i,
                'strength': 1 - body_percent,  # stronger when body is smaller
                'signal': 'neutral',
                'description': 'Doji pattern indicates indecision in the market'
            })
    
    return results

def detect_hammer(df, body_ratio=0.3, shadow_ratio=2.0):
    """
    Deteksi pola Hammer dan Inverted Hammer
    body_ratio: maksimum rasio body terhadap total range
    shadow_ratio: minimum rasio shadow terhadap body
    """
    results = []
    
    for i in range(len(df) - 1, max(len(df) - 5, 0) - 1, -1):
        row = df.iloc[i]
        body_size = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range == 0 or body_size == 0:  # Prevent division by zero
            continue
            
        body_percent = body_size / total_range
        
        # Calculate upper and lower shadows
        if row['close'] >= row['open']:  # Bullish candle
            upper_shadow = row['high'] - row['close']
            lower_shadow = row['open'] - row['low']
        else:  # Bearish candle
            upper_shadow = row['high'] - row['open']
            lower_shadow = row['close'] - row['low']
            
        # For hammer, body should be small and lower shadow should be long
        if body_percent <= body_ratio:
            if lower_shadow / body_size >= shadow_ratio and upper_shadow / body_size < 0.5:
                results.append({
                    'type': 'hammer',
                    'index': i,
                    'strength': min(lower_shadow / body_size / shadow_ratio, 2.0),  # cap strength
                    'signal': 'bullish',
                    'description': 'Hammer pattern suggests potential bullish reversal'
                })
            elif upper_shadow / body_size >= shadow_ratio and lower_shadow / body_size < 0.5:
                results.append({
                    'type': 'inverted_hammer',
                    'index': i,
                    'strength': min(upper_shadow / body_size / shadow_ratio, 2.0),  # cap strength
                    'signal': 'bullish',
                    'description': 'Inverted Hammer pattern suggests potential bullish reversal'
                })
    
    return results

def detect_engulfing(df):
    """Deteksi pola Bullish dan Bearish Engulfing"""
    results = []
    
    for i in range(len(df) - 1, max(len(df) - 5, 0), -1):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        curr_body_size = abs(curr['close'] - curr['open'])
        prev_body_size = abs(prev['close'] - prev['open'])
        
        # Skip if bodies are very small
        if curr_body_size < 0.001 or prev_body_size < 0.001:
            continue
            
        # Bullish Engulfing: current candle bullish, previous bearish, current body engulfs previous
        if (curr['close'] > curr['open'] and  # Current bullish
            prev['close'] < prev['open'] and  # Previous bearish
            curr['close'] >= prev['open'] and  # Close above previous open
            curr['open'] <= prev['close']):    # Open below previous close
            
            strength = curr_body_size / prev_body_size
            
            results.append({
                'type': 'bullish_engulfing',
                'index': i,
                'strength': min(strength, 2.0),  # cap strength at 2.0
                'signal': 'bullish',
                'description': 'Bullish Engulfing pattern indicates potential upward reversal'
            })
            
        # Bearish Engulfing: current candle bearish, previous bullish, current body engulfs previous
        elif (curr['close'] < curr['open'] and  # Current bearish
              prev['close'] > prev['open'] and  # Previous bullish
              curr['close'] <= prev['open'] and  # Close below previous open
              curr['open'] >= prev['close']):    # Open above previous close
              
            strength = curr_body_size / prev_body_size
            
            results.append({
                'type': 'bearish_engulfing',
                'index': i,
                'strength': min(strength, 2.0),  # cap strength at 2.0
                'signal': 'bearish',
                'description': 'Bearish Engulfing pattern indicates potential downward reversal'
            })
    
    return results

def detect_morning_evening_star(df, doji_tolerance=0.1, body_ratio=0.5):
    """Deteksi pola Morning Star dan Evening Star"""
    results = []
    
    if len(df) < 3:  # Need at least 3 candles
        return results
        
    for i in range(len(df) - 1, max(len(df) - 5, 1), -1):
        # Need 3 candles for this pattern
        curr = df.iloc[i]
        middle = df.iloc[i-1]
        first = df.iloc[i-2]
        
        # Calculate body sizes
        curr_body = abs(curr['close'] - curr['open'])
        middle_body = abs(middle['close'] - middle['open'])
        first_body = abs(first['close'] - first['open'])
        
        # Middle candle should have a small body (doji-like)
        if middle_body / (middle['high'] - middle['low']) > doji_tolerance:
            continue
            
        # For both patterns, first and third candles should have substantial bodies
        if first_body < 0.001 or curr_body < 0.001:
            continue
            
        # Morning Star: first bearish, gap down, third bullish
        if (first['close'] < first['open'] and  # First bearish
            curr['close'] > curr['open'] and    # Current bullish
            max(middle['open'], middle['close']) < first['close'] and  # Gap down after first
            curr_body / first_body >= body_ratio):  # Current body substantial compared to first
            
            results.append({
                'type': 'morning_star',
                'index': i,
                'strength': 1.5,
                'signal': 'bullish',
                'description': 'Morning Star pattern indicates potential bullish reversal'
            })
            
        # Evening Star: first bullish, gap up, third bearish
        elif (first['close'] > first['open'] and  # First bullish
              curr['close'] < curr['open'] and    # Current bearish
              min(middle['open'], middle['close']) > first['close'] and  # Gap up after first
              curr_body / first_body >= body_ratio):  # Current body substantial compared to first
              
            results.append({
                'type': 'evening_star',
                'index': i,
                'strength': 1.5,
                'signal': 'bearish',
                'description': 'Evening Star pattern indicates potential bearish reversal'
            })
    
    return results

def detect_all_patterns(df):
    """Deteksi semua pola candle dan return hasil"""
    patterns = []
    
    # Deteksi berbagai pola
    patterns.extend(detect_doji(df))
    patterns.extend(detect_hammer(df))
    patterns.extend(detect_engulfing(df))
    patterns.extend(detect_morning_evening_star(df))
    
    # Sort patterns by index, with most recent first
    patterns.sort(key=lambda x: x['index'])
    
    # Return only patterns from the most recent 3 candles
    recent_patterns = [p for p in patterns if p['index'] >= len(df) - 3]
    
    return recent_patterns

def analyze_patterns(df):
    """
    Analisis pola candlestick dan return sinyal dan skor
    Returns: Dict dengan sinyal beli/jual dan penjelasan
    """
    patterns = detect_all_patterns(df)
    
    # Initialize signals
    buy_signals = []
    sell_signals = []
    buy_strength = 0
    sell_strength = 0
    
    # Analyze detected patterns
    for pattern in patterns:
        if pattern['signal'] == 'bullish':
            buy_signals.append(f"{pattern['type']} ({pattern['description']})")
            buy_strength += pattern['strength'] * 10  # Scale to 0-100
        elif pattern['signal'] == 'bearish':
            sell_signals.append(f"{pattern['type']} ({pattern['description']})")
            sell_strength += pattern['strength'] * 10  # Scale to 0-100
    
    return {
        'patterns': patterns,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'buy_strength': min(buy_strength, 100),  # Cap at 100
        'sell_strength': min(sell_strength, 100)  # Cap at 100
    }