import numpy as np
from datetime import datetime

def calculate_trend(df, short_period=8, medium_period=21, long_period=55):
    """Calculate trend strength using multiple timeframes - optimized for crypto volatility"""
    df['EMA_short'] = df['close'].ewm(span=short_period).mean()  # Changed to EMA for faster response
    df['SMA_medium'] = df['close'].rolling(window=medium_period).mean()
    df['SMA_long'] = df['close'].rolling(window=long_period).mean()
    
    current_price = df['close'].iloc[-1]
    
    # Determine trend strength with weighted importance
    short_trend = 1.5 if current_price > df['EMA_short'].iloc[-1] else -1.5  # Higher weight for short term
    medium_trend = 1 if current_price > df['SMA_medium'].iloc[-1] else -1
    long_trend = 0.5 if current_price > df['SMA_long'].iloc[-1] else -0.5  # Lower weight for long term
    
    # Trend alignment adds extra strength
    aligned = (short_trend > 0 and medium_trend > 0 and long_trend > 0) or (short_trend < 0 and medium_trend < 0 and long_trend < 0)
    alignment_bonus = 0.5 if aligned else 0
    
    trend_strength = (short_trend + medium_trend + long_trend) / 3 + alignment_bonus
    return trend_strength

def calculate_volume_trend(df, period=14):
    """Calculate volume trend with improved outlier handling for crypto markets"""
    # Use EMA for volume to reduce impact of outliers
    avg_volume = df['volume'].ewm(span=period).mean()
    current_volume = df['volume'].iloc[-1]
    
    # Handle extreme volume spikes
    volume_ratio = current_volume / avg_volume.iloc[-1] if not np.isnan(avg_volume.iloc[-1]) else 1.0
    
    # Check for volume change direction
    recent_volume_change = df['volume'].iloc[-1] > df['volume'].iloc[-2]
    price_change = df['close'].iloc[-1] > df['close'].iloc[-2]
    
    # Volume and price in same direction is stronger signal
    direction_bonus = 0.2 if (recent_volume_change == price_change) else 0
    
    return min(volume_ratio + direction_bonus, 3.0)  # Cap at 3.0 to prevent extreme values

def calculate_volatility(df, window=14):
    """Calculate current market volatility"""
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    current_volatility = volatility.iloc[-1] * 100  # Convert to percentage
    
    if np.isnan(current_volatility):
        return 1.0  # Default medium volatility
    
    return current_volatility

def make_decision(analysis, df):
    """
    Make trading decision based on all indicators, trend, and market conditions
    Returns: dict with decision details and confidence level
    """
    trend_strength = calculate_trend(df)
    volume_trend = calculate_volume_trend(df)
    volatility = calculate_volatility(df)
    
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
    
    # Strong trend conditions (30% weight)
    trend_score = 0
    if trend_strength > 0.8:
        trend_score = 30
        decision['reason'].append("Strong uptrend detected")
    elif trend_strength > 0.3:
        trend_score = 15
        decision['reason'].append("Moderate uptrend detected")
    elif trend_strength < -0.8:
        trend_score = -30
        decision['reason'].append("Strong downtrend detected")
    elif trend_strength < -0.3:
        trend_score = -15
        decision['reason'].append("Moderate downtrend detected")
    
    # Indicator consensus (40% weight)
    indicator_score = 0
    if total_buy > total_sell:
        indicator_score = (total_buy - total_sell) * 0.4
        if total_buy > 70:
            decision['reason'].append(f"Strong buy signal from indicators ({total_buy}%)")
        else:
            decision['reason'].append(f"Moderate buy signal from indicators ({total_buy}%)")
    else:
        indicator_score = (total_sell - total_buy) * -0.4
        if total_sell > 70:
            decision['reason'].append(f"Strong sell signal from indicators ({total_sell}%)")
        else:
            decision['reason'].append(f"Moderate sell signal from indicators ({total_sell}%)")
    
    # Volume analysis (20% weight)
    volume_score = 0
    if volume_trend > 1.8:
        volume_score = 20 if trend_score > 0 else -20
        decision['reason'].append("High volume confirming trend")
    elif volume_trend > 1.2:
        volume_score = 10 if trend_score > 0 else -10
        decision['reason'].append("Above average volume")
    elif volume_trend < 0.6:
        decision['reason'].append("Low volume - signals may be weak")
    
    # Volatility impact (10% weight)
    volatility_score = 0
    if volatility > 5:  # High volatility
        decision['reason'].append(f"High volatility ({volatility:.2f}%) - increased risk")
        volatility_score = -10  # High volatility generally increases risk
    elif volatility < 1:  # Low volatility
        decision['reason'].append(f"Low volatility ({volatility:.2f}%) - potential breakout soon")
        
    # Calculate total confidence score
    total_score = trend_score + indicator_score + volume_score + volatility_score
    
    # Determine action based on total score
    if total_score > 65:
        decision['action'] = 'STRONG_BUY'
        decision['risk_level'] = 'LOW' if volatility < 3 else 'MEDIUM'
        decision['reason'].append("Multiple indicators showing strong buy signals")
    elif total_score > 30:
        decision['action'] = 'BUY'
        decision['risk_level'] = 'MEDIUM'
        decision['reason'].append("Positive signals with moderate strength")
    elif total_score < -65:
        decision['action'] = 'STRONG_SELL'
        decision['risk_level'] = 'LOW' if volatility < 3 else 'MEDIUM'
        decision['reason'].append("Multiple indicators showing strong sell signals")
    elif total_score < -30:
        decision['action'] = 'SELL'
        decision['risk_level'] = 'MEDIUM'
        decision['reason'].append("Negative signals with moderate strength")
    else:
        decision['action'] = 'HOLD'
        decision['risk_level'] = 'HIGH' if abs(total_score) < 15 else 'MEDIUM'
        decision['reason'].append("Mixed signals - no clear direction")
    
    decision['confidence'] = abs(total_score)
    
    # Add specific cryptocurrency insights
    if 'RSI' in indicators:
        if indicators['RSI']['value'] > 80:
            decision['reason'].append(f"Overbought RSI ({indicators['RSI']['value']}) - potential reversal")
        elif indicators['RSI']['value'] < 20:
            decision['reason'].append(f"Oversold RSI ({indicators['RSI']['value']}) - potential reversal")
    
    if 'Bollinger' in indicators:
        if indicators['Bollinger']['value'] > 90:
            decision['reason'].append("Price near upper Bollinger Band - high resistance")
        elif indicators['Bollinger']['value'] < 10:
            decision['reason'].append("Price near lower Bollinger Band - strong support")
    
    # Adjust risk level based on volatility
    if volatility > 4:
        if decision['risk_level'] == 'LOW':
            decision['risk_level'] = 'MEDIUM'
        elif decision['risk_level'] == 'MEDIUM':
            decision['risk_level'] = 'HIGH'
            
    # Add transaction cost consideration for low-confidence signals
    if decision['confidence'] < 30 and decision['action'] != 'HOLD':
        decision['reason'].append("Low confidence signal - consider transaction costs")
    
    return decision