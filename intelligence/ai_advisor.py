# ai_advisor.py

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def load_market_conditions():
    """
    Load current market conditions database
    """
    try:
        if os.path.exists('data/market_conditions.json'):
            with open('data/market_conditions.json', 'r') as file:
                return json.load(file)
        else:
            return {
                'bitcoin_dominance': 50,
                'global_market_cap': 1000000000000,
                'market_sentiment': 'neutral',
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
    except Exception as e:
        print(f"Error loading market conditions: {str(e)}")
        return {
            'bitcoin_dominance': 50,
            'global_market_cap': 1000000000000,
            'market_sentiment': 'neutral',
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }

def load_crypto_profiles():
    """
    Load cryptocurrency specific profiles
    """
    try:
        if os.path.exists('data/crypto_profiles.json'):
            with open('data/crypto_profiles.json', 'r') as file:
                return json.load(file)
        else:
            # Default profiles for common cryptos
            return {
                'BTC': {'correlation_to_market': 0.9, 'avg_volatility': 3.5, 'category': 'store_of_value'},
                'ETH': {'correlation_to_market': 0.8, 'avg_volatility': 4.5, 'category': 'smart_contract'},
                'BNB': {'correlation_to_market': 0.7, 'avg_volatility': 5.0, 'category': 'exchange'},
                'XRP': {'correlation_to_market': 0.6, 'avg_volatility': 6.0, 'category': 'payment'},
                'ADA': {'correlation_to_market': 0.7, 'avg_volatility': 7.0, 'category': 'smart_contract'},
                'SOL': {'correlation_to_market': 0.7, 'avg_volatility': 8.0, 'category': 'smart_contract'},
                'DOGE': {'correlation_to_market': 0.5, 'avg_volatility': 10.0, 'category': 'meme'},
                'SHIB': {'correlation_to_market': 0.4, 'avg_volatility': 12.0, 'category': 'meme'},
                'DOT': {'correlation_to_market': 0.6, 'avg_volatility': 7.5, 'category': 'interoperability'},
                'MATIC': {'correlation_to_market': 0.6, 'avg_volatility': 9.0, 'category': 'scaling'},
                # Default for other coins
                'default': {'correlation_to_market': 0.5, 'avg_volatility': 8.0, 'category': 'altcoin'}
            }
    except Exception as e:
        print(f"Error loading crypto profiles: {str(e)}")
        return {'default': {'correlation_to_market': 0.5, 'avg_volatility': 8.0, 'category': 'altcoin'}}

def evaluate_context(crypto_symbol, timeframe, technical_analysis, pattern_analysis, market_context, risk_analysis, ml_prediction):
    """
    Evaluate all context data to provide holistic advice
    """
    # Load market conditions data
    market_conditions = load_market_conditions()
    
    # Load crypto profiles
    crypto_profiles = load_crypto_profiles()
    
    # Get crypto profile (or default if not found)
    crypto_profile = crypto_profiles.get(crypto_symbol, crypto_profiles['default'])
    
    # Extract key metrics from analyses
    buy_signals = technical_analysis.get('total_buy', 0)
    sell_signals = technical_analysis.get('total_sell', 0)
    
    market_phase = market_context.get('market_phase', {}).get('phase', 'unknown')
    volatility = risk_analysis.get('volatility_metrics', {}).get('atr_percent', 0)
    
    pattern_buy = pattern_analysis.get('buy_strength', 0)
    pattern_sell = pattern_analysis.get('sell_strength', 0)
    
    ml_direction = ml_prediction.get('prediction', {}).get('prediction', 'unknown')
    ml_confidence = ml_prediction.get('prediction', {}).get('confidence', 0)
    
    # Determine overall signal strength (weighted average)
    # Technical analysis: 35%
    # Market context: 25%
    # Pattern analysis: 15%
    # ML prediction: 25%
    
    tech_score = (buy_signals - sell_signals) / 100 * 35
    
    # Market context score
    if market_phase in ['uptrend', 'weak_uptrend']:
        market_score = market_context.get('buy_strength', 0) / 100 * 25
    elif market_phase in ['downtrend', 'weak_downtrend']:
        market_score = -market_context.get('sell_strength', 0) / 100 * 25
    else:
        market_score = 0
    
    # Pattern score
    pattern_score = (pattern_buy - pattern_sell) / 100 * 15
    
    # ML score
    if ml_direction == 'UP':
        ml_score = ml_confidence / 100 * 25
    elif ml_direction == 'DOWN':
        ml_score = -ml_confidence / 100 * 25
    else:
        ml_score = 0
    
    # Calculate overall score
    overall_score = tech_score + market_score + pattern_score + ml_score
    
    # Generate action advice
    if overall_score > 20:
        action = "STRONG_BUY"
    elif overall_score > 10:
        action = "BUY"
    elif overall_score < -20:
        action = "STRONG_SELL"
    elif overall_score < -10:
        action = "SELL"
    else:
        action = "HOLD"
    
    # Calculate confidence percentage
    confidence = min(abs(overall_score) * 2.5, 100)
    
    return {
        'action': action,
        'confidence': confidence,
        'overall_score': overall_score,
        'tech_contribution': tech_score,
        'market_contribution': market_score,
        'pattern_contribution': pattern_score,
        'ml_contribution': ml_score
    }

def generate_trading_advice(crypto_symbol, timeframe, analysis_results, context_eval):
    """
    Generate specific trading advice based on all analyses
    """
    # Extract parameters for advice
    action = context_eval['action']
    confidence = context_eval['confidence']
    
    tech_analysis = analysis_results.get('technical_analysis', {})
    market_context = analysis_results.get('market_context', {})
    risk_analysis = analysis_results.get('risk_analysis', {})
    ml_prediction = analysis_results.get('ml_prediction', {})
    
    # Crypto profile
    crypto_profiles = load_crypto_profiles()
    crypto_profile = crypto_profiles.get(crypto_symbol, crypto_profiles['default'])
    crypto_category = crypto_profile.get('category', 'altcoin')
    
    # Generate advice pieces
    advice = []
    
    # Entry advice
    if action in ['BUY', 'STRONG_BUY']:
        if confidence > 80:
            advice.append(f"Strong buy signal with {confidence:.1f}% confidence. Consider immediate entry.")
        elif confidence > 60:
            advice.append(f"Moderate buy signal with {confidence:.1f}% confidence. Look for entry on minor pullbacks.")
        else:
            advice.append(f"Weak buy signal with {confidence:.1f}% confidence. Consider scaling in gradually.")
            
        # Add specific entry advice
        if market_context.get('market_phase', {}).get('phase', '') in ['uptrend', 'strong_uptrend']:
            advice.append("Market is in an uptrend - trend-following strategy recommended.")
        
        # Add support level advice if available
        supports = market_context.get('support_resistance', {}).get('supports', [])
        if supports and len(supports) > 0:
            advice.append(f"Consider waiting for retracement to support at ${supports[0]['level']:.2f} for better entry.")
            
    elif action in ['SELL', 'STRONG_SELL']:
        if confidence > 80:
            advice.append(f"Strong sell signal with {confidence:.1f}% confidence. Consider immediate exit.")
        elif confidence > 60:
            advice.append(f"Moderate sell signal with {confidence:.1f}% confidence. Look to exit on strength.")
        else:
            advice.append(f"Weak sell signal with {confidence:.1f}% confidence. Consider reducing position size.")
            
        # Add specific exit advice
        if market_context.get('market_phase', {}).get('phase', '') in ['downtrend', 'strong_downtrend']:
            advice.append("Market is in a downtrend - capital preservation should be priority.")
            
        # Add resistance level advice if available
        resistances = market_context.get('support_resistance', {}).get('resistances', [])
        if resistances and len(resistances) > 0:
            advice.append(f"Consider waiting for bounce to resistance at ${resistances[0]['level']:.2f} for better exit.")
            
    else:  # HOLD
        advice.append(f"Market signals are mixed with {confidence:.1f}% confidence. Hold current positions.")
        advice.append("Wait for stronger signals before making new entries.")
    
    # Risk management advice
    if risk_analysis:
        position_info = risk_analysis.get('position_info', {})
        risk_tips = risk_analysis.get('risk_tips', [])
        
        if position_info:
            advice.append(f"Recommended position size: ${position_info.get('position_size', 0):.2f}")
            
        # Add top 2 risk tips
        for tip in risk_tips[:2]:
            advice.append(tip)
    
    # Timeframe-specific advice
    if timeframe == '15m' or timeframe == '30m':
        advice.append("Short timeframe signals can be noisy. Confirm with higher timeframe analysis.")
    elif timeframe == '1h':
        advice.append("Hourly timeframe provides balanced view. Good for intraday and swing trades.")
    elif timeframe == '4h' or timeframe == '1d':
        advice.append("Higher timeframe signals are more reliable for swing and position trades.")
    
    # Category-specific advice
    if crypto_category == 'meme':
        advice.append("Meme coins have higher volatility. Consider smaller position sizes and tighter stops.")
    elif crypto_category == 'smart_contract':
        advice.append("Smart contract platforms often move with the broader DeFi ecosystem.")
    elif crypto_category == 'store_of_value':
        advice.append("Strong correlation with overall market sentiment. Monitor broader market conditions.")
    
    # ML-specific advice if available
    if ml_prediction and ml_prediction.get('ml_insights'):
        # Add top ML insight
        advice.append(ml_prediction['ml_insights'][0])
    
    return advice

def provide_ai_advice(crypto_symbol, timeframe, analysis_results):
    """
    Main function to provide AI-powered trading advice
    """
    # Extract analysis components
    technical_analysis = analysis_results.get('technical_analysis', {})
    pattern_analysis = analysis_results.get('pattern_analysis', {})
    market_context = analysis_results.get('market_context', {})
    risk_analysis = analysis_results.get('risk_analysis', {})
    ml_prediction = analysis_results.get('ml_prediction', {})
    
    # Evaluate context
    context_eval = evaluate_context(
        crypto_symbol,
        timeframe,
        technical_analysis,
        pattern_analysis,
        market_context,
        risk_analysis,
        ml_prediction
    )
    
    # Generate advice
    advice = generate_trading_advice(crypto_symbol, timeframe, analysis_results, context_eval)
    
    # Format timestamps
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'timestamp': timestamp,
        'crypto': crypto_symbol,
        'timeframe': timeframe,
        'action': context_eval['action'],
        'confidence': context_eval['confidence'],
        'advice': advice,
        'contributors': {
            'technical': context_eval['tech_contribution'],
            'market': context_eval['market_contribution'],
            'patterns': context_eval['pattern_contribution'],
            'machine_learning': context_eval['ml_contribution']
        }
    }