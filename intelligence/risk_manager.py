# risk_manager.py

import numpy as np
import pandas as pd
from datetime import datetime

def calculate_volatility_metrics(df, window=14):
    """
    Menghitung berbagai metrik volatilitas untuk aset
    """
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate rolling volatility (standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(window)
    
    # Calculate Average True Range (ATR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=window).mean()
    df['atr_percent'] = (df['atr'] / df['close']) * 100
    
    # Calculate price swings over different periods
    df['swing_1d'] = abs(df['high'] - df['low']) / df['close'] * 100
    df['swing_3d'] = abs(df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()) / df['close'] * 100
    df['swing_7d'] = abs(df['high'].rolling(window=7).max() - df['low'].rolling(window=7).min()) / df['close'] * 100
    
    current_close = df['close'].iloc[-1]
    
    return {
        'daily_volatility': df['volatility'].iloc[-1] * 100,  # as percentage
        'atr': df['atr'].iloc[-1],
        'atr_percent': df['atr_percent'].iloc[-1],
        'swing_1d': df['swing_1d'].iloc[-1],
        'swing_3d': df['swing_3d'].iloc[-1],
        'swing_7d': df['swing_7d'].iloc[-1],
        'current_price': current_close
    }

def determine_risk_profile(crypto_symbol, volatility_metrics):
    """
    Menentukan profil risiko berdasarkan simbol crypto dan metrik volatilitas
    """
    # Base risk categories for different cryptocurrencies
    low_risk_cryptos = ['BTC', 'ETH']
    medium_risk_cryptos = ['BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'LINK', 'MATIC', 'AVAX']
    # All others are considered high risk
    
    # Determine base risk level from crypto category
    if crypto_symbol in low_risk_cryptos:
        base_risk = 'low'
        base_risk_score = 1
    elif crypto_symbol in medium_risk_cryptos:
        base_risk = 'medium'
        base_risk_score = 2
    else:
        base_risk = 'high'
        base_risk_score = 3
    
    # Adjust risk level based on current volatility
    volatility_score = 0
    
    # Check daily volatility
    daily_vol = volatility_metrics['daily_volatility']
    if daily_vol > 7:  # Very high daily volatility
        volatility_score += 2
    elif daily_vol > 4:  # High daily volatility
        volatility_score += 1
    
    # Check recent price swings
    if volatility_metrics['swing_3d'] > 15:  # Large 3-day swings
        volatility_score += 1
    
    # Final risk level determination
    total_risk_score = base_risk_score + volatility_score
    
    if total_risk_score <= 1:
        risk_level = 'low'
    elif total_risk_score <= 3:
        risk_level = 'medium'
    else:
        risk_level = 'high'
    
    return {
        'base_risk': base_risk,
        'risk_level': risk_level,
        'risk_score': total_risk_score,
        'volatility_score': volatility_score
    }

def calculate_position_size(account_balance, risk_percent, risk_level, entry_price, stop_loss):
    """
    Menghitung ukuran posisi berdasarkan manajemen risiko
    """
    # Adjust risk percentage based on risk level
    adjusted_risk = risk_percent
    if risk_level == 'high':
        adjusted_risk = risk_percent * 0.7  # 30% reduction for high risk assets
    elif risk_level == 'medium':
        adjusted_risk = risk_percent * 0.85  # 15% reduction for medium risk assets
    
    # Maximum amount willing to risk
    risk_amount = account_balance * (adjusted_risk / 100)
    
    # Calculate stop loss distance
    if stop_loss > 0:
        stop_distance_percent = abs(entry_price - stop_loss) / entry_price * 100
        
        # If stop is too tight (less than 1%), adjust it
        if stop_distance_percent < 1:
            stop_distance_percent = 1
            adjusted_stop = entry_price * 0.99  # 1% adjusted stop
        else:
            adjusted_stop = stop_loss
    else:
        # Default to 2-5% stop based on risk level
        if risk_level == 'low':
            stop_distance_percent = 2
        elif risk_level == 'medium':
            stop_distance_percent = 3
        else:  # high
            stop_distance_percent = 5
            
        adjusted_stop = entry_price * (1 - stop_distance_percent / 100)
    
    # Calculate position size
    position_size = risk_amount / (stop_distance_percent / 100 * entry_price)
    
    # Calculate units to buy
    units = position_size / entry_price
    
    return {
        'position_size': position_size,
        'units': units,
        'risk_amount': risk_amount,
        'adjusted_risk_percent': adjusted_risk,
        'stop_distance_percent': stop_distance_percent,
        'adjusted_stop': adjusted_stop
    }

def calculate_optimal_stops(df, entry_price, risk_level, action, volatility_metrics):
    """
    Menghitung level stop loss dan take profit yang optimal
    """
    # Get ATR value
    atr = volatility_metrics['atr']
    atr_percent = volatility_metrics['atr_percent']
    
    # Default multipliers based on risk level
    if risk_level == 'low':
        sl_tight = 1.5
        sl_normal = 2.0
        sl_wide = 3.0
        tp_conservative = 2.0
        tp_moderate = 3.0
        tp_aggressive = 5.0
    elif risk_level == 'medium':
        sl_tight = 2.0
        sl_normal = 3.0
        sl_wide = 4.0
        tp_conservative = 2.5
        tp_moderate = 4.0
        tp_aggressive = 6.0
    else:  # high
        sl_tight = 3.0
        sl_normal = 4.0
        sl_wide = 5.0
        tp_conservative = 3.0
        tp_moderate = 5.0
        tp_aggressive = 8.0
    
    # Calculate stops and targets using ATR
    if action in ['BUY', 'STRONG_BUY']:
        stop_tight = entry_price * (1 - (atr_percent * sl_tight / 100))
        stop_normal = entry_price * (1 - (atr_percent * sl_normal / 100))
        stop_wide = entry_price * (1 - (atr_percent * sl_wide / 100))
        
        target_conservative = entry_price * (1 + (atr_percent * tp_conservative / 100))
        target_moderate = entry_price * (1 + (atr_percent * tp_moderate / 100))
        target_aggressive = entry_price * (1 + (atr_percent * tp_aggressive / 100))
    else:  # SELL actions
        stop_tight = entry_price * (1 + (atr_percent * sl_tight / 100))
        stop_normal = entry_price * (1 + (atr_percent * sl_normal / 100))
        stop_wide = entry_price * (1 + (atr_percent * sl_wide / 100))
        
        target_conservative = entry_price * (1 - (atr_percent * tp_conservative / 100))
        target_moderate = entry_price * (1 - (atr_percent * tp_moderate / 100))
        target_aggressive = entry_price * (1 - (atr_percent * tp_aggressive / 100))
    
    # Find nearest support/resistance for stop placement
    df_subset = df.tail(50)  # Use last 50 candles
    
    # For buy orders, find support levels below current price
    if action in ['BUY', 'STRONG_BUY']:
        lower_lows = df_subset[df_subset['low'] < df_subset['low'].shift(1)]['low']
        support_levels = lower_lows[lower_lows < entry_price].sort_values(ascending=False)
        
        # If there's a support level between stop_normal and stop_wide, use it
        for support in support_levels:
            if stop_wide < support < stop_normal:
                stop_normal = support
                break
    
    # For sell orders, find resistance levels above current price
    else:
        higher_highs = df_subset[df_subset['high'] > df_subset['high'].shift(1)]['high']
        resistance_levels = higher_highs[higher_highs > entry_price].sort_values()
        
        # If there's a resistance level between stop_normal and stop_wide, use it
        for resistance in resistance_levels:
            if stop_normal < resistance < stop_wide:
                stop_normal = resistance
                break
    
    # Calculate risk-reward ratios
    if action in ['BUY', 'STRONG_BUY']:
        rr_conservative = (target_conservative - entry_price) / (entry_price - stop_normal)
        rr_moderate = (target_moderate - entry_price) / (entry_price - stop_normal)
        rr_aggressive = (target_aggressive - entry_price) / (entry_price - stop_normal)
    else:
        rr_conservative = (entry_price - target_conservative) / (stop_normal - entry_price)
        rr_moderate = (entry_price - target_moderate) / (stop_normal - entry_price)
        rr_aggressive = (entry_price - target_aggressive) / (stop_normal - entry_price)
    
    return {
        'stop_loss': {
            'tight': round(stop_tight, 8),
            'normal': round(stop_normal, 8),
            'wide': round(stop_wide, 8)
        },
        'take_profit': {
            'conservative': round(target_conservative, 8),
            'moderate': round(target_moderate, 8),
            'aggressive': round(target_aggressive, 8)
        },
        'risk_reward': {
            'conservative': round(rr_conservative, 2),
            'moderate': round(rr_moderate, 2),
            'aggressive': round(rr_aggressive, 2)
        }
    }

def analyze_risk_management(df, crypto_symbol, entry_price, action, account_balance=1000, risk_percent=2):
    """
    Analisis lengkap manajemen risiko dan return rekomendasi
    """
    # Calculate volatility metrics
    volatility_metrics = calculate_volatility_metrics(df)
    
    # Determine risk profile
    risk_profile = determine_risk_profile(crypto_symbol, volatility_metrics)
    
    # Calculate optimal stops and targets
    levels = calculate_optimal_stops(df, entry_price, risk_profile['risk_level'], action, volatility_metrics)
    
    # Calculate position size (using normal stop loss)
    position_info = calculate_position_size(
        account_balance, 
        risk_percent, 
        risk_profile['risk_level'], 
        entry_price, 
        levels['stop_loss']['normal']
    )
    
    # Generate risk management tips based on profile
    risk_tips = []
    
    if risk_profile['risk_level'] == 'high':
        risk_tips.append("High-risk asset - consider reducing position size")
        risk_tips.append("Use wider stops to accommodate volatility")
        risk_tips.append("Take partial profits earlier to reduce exposure")
    elif risk_profile['risk_level'] == 'medium':
        risk_tips.append("Consider scaling in to reduce entry risk")
        risk_tips.append("Monitor support/resistance levels for stop placement")
    else:
        risk_tips.append("Consider trailing stops to maximize profit potential")
        risk_tips.append("Look for higher timeframe confluence for entries")
    
    # Add risk:reward tips
    moderate_rr = levels['risk_reward']['moderate']
    if moderate_rr < 1.5:
        risk_tips.append(f"Low risk:reward ratio ({moderate_rr}) - consider finding better setup")
    elif moderate_rr > 3:
        risk_tips.append(f"Excellent risk:reward ratio ({moderate_rr}) - consider larger position")
    
    return {
        'risk_profile': risk_profile,
        'volatility_metrics': volatility_metrics,
        'position_info': position_info,
        'risk_levels': levels,
        'risk_tips': risk_tips
    }