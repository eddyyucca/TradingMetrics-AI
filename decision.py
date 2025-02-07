import numpy as np
from datetime import datetime

def calculate_trend(df, short_period=10, medium_period=20, long_period=50):
    """Calculate trend strength using multiple timeframes"""
    df['SMA_short'] = df['close'].rolling(window=short_period).mean()
    df['SMA_medium'] = df['close'].rolling(window=medium_period).mean()
    df['SMA_long'] = df['close'].rolling(window=long_period).mean()
    
    current_price = df['close'].iloc[-1]
    
    # Determine trend strength
    short_trend = 1 if current_price > df['SMA_short'].iloc[-1] else -1
    medium_trend = 1 if current_price > df['SMA_medium'].iloc[-1] else -1
    long_trend = 1 if current_price > df['SMA_long'].iloc[-1] else -1
    
    trend_strength = (short_trend + medium_trend + long_trend) / 3
    return trend_strength

def calculate_volume_trend(df, period=20):
    """Calculate volume trend"""
    avg_volume = df['volume'].rolling(window=period).mean()
    current_volume = df['volume'].iloc[-1]
    return current_volume / avg_volume.iloc[-1]

def make_decision(analysis, df):
    """
    Make trading decision based on all indicators and trend
    Returns: dict with decision details and confidence level
    """
    trend_strength = calculate_trend(df)
    volume_trend = calculate_volume_trend(df)
    
    # Get current market conditions
    total_buy = analysis['total_buy']
    total_sell = analysis['total_sell']
    indicators = analysis['indicators']
    
    # Initialize decision metrics
    decision = {
        'action': 'HOLD',
        'confidence': 0,
        'reason': [],
        'risk_level': 'MEDIUM',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Strong trend conditions (35% weight)
    trend_score = 0
    if trend_strength > 0.5:
        trend_score = 35
        decision['reason'].append("Strong uptrend detected")
    elif trend_strength < -0.5:
        trend_score = -35
        decision['reason'].append("Strong downtrend detected")
    
    # Indicator consensus (35% weight)
    indicator_score = 0
    if total_buy > total_sell:
        indicator_score = (total_buy - total_sell) * 0.35
    else:
        indicator_score = (total_sell - total_buy) * -0.35
    
    # Volume analysis (30% weight)
    volume_score = 0
    if volume_trend > 1.5:
        volume_score = 30 if trend_score > 0 else -30
        decision['reason'].append("High volume confirming trend")
    elif volume_trend < 0.5:
        decision['reason'].append("Low volume - weak signals")
    
    # Calculate total confidence score
    total_score = trend_score + indicator_score + volume_score
    
    # Determine action based on total score
    if total_score > 60:
        decision['action'] = 'STRONG_BUY'
        decision['risk_level'] = 'LOW'
        decision['reason'].append("Multiple indicators showing strong buy signals")
    elif total_score > 30:
        decision['action'] = 'BUY'
        decision['risk_level'] = 'MEDIUM'
        decision['reason'].append("Positive signals with moderate strength")
    elif total_score < -60:
        decision['action'] = 'STRONG_SELL'
        decision['risk_level'] = 'LOW'
        decision['reason'].append("Multiple indicators showing strong sell signals")
    elif total_score < -30:
        decision['action'] = 'SELL'
        decision['risk_level'] = 'MEDIUM'
        decision['reason'].append("Negative signals with moderate strength")
    else:
        decision['action'] = 'HOLD'
        decision['risk_level'] = 'HIGH'
        decision['reason'].append("Mixed signals - no clear direction")
    
    decision['confidence'] = abs(total_score)
    
    # Add risk warnings for crypto
    if abs(total_score) < 40:
        decision['reason'].append("Low confidence signal - higher risk")
    if volume_trend < 0.7:
        decision['reason'].append("Low volume - consider waiting")
    if indicators['RSI']['value'] > 85 or indicators['RSI']['value'] < 15:
        decision['reason'].append("Extreme RSI - potential reversal")
    
    return decision