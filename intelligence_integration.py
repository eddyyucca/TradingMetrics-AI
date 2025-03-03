# intelligence_integration.py

import pandas as pd
from datetime import datetime
import time
import os

# Import semua modul kecerdasan
from intelligence.pattern_recognition import analyze_patterns
from intelligence.market_context import analyze_market_context
from intelligence.risk_manager import analyze_risk_management
from intelligence.ml_models import analyze_ml_prediction
from intelligence.ai_advisor import provide_ai_advice

def run_comprehensive_analysis(df, crypto_symbol, timeframe="1h", account_balance=1000, risk_percent=2):
    """
    Menjalankan semua analisis kecerdasan dan mengembalikan hasil lengkap
    """
    print(f"Running comprehensive analysis for {crypto_symbol} on {timeframe} timeframe...")
    start_time = time.time()
    
    results = {}
    
    # 1. Analisis indikator teknikal (menggunakan fungsi analyze_indicators yang sudah ada)
    from analysis import analyze_indicators
    technical_analysis = analyze_indicators(df)
    results['technical_analysis'] = technical_analysis
    
    # 2. Analisis pola candlestick
    try:
        pattern_analysis = analyze_patterns(df)
        results['pattern_analysis'] = pattern_analysis
        print(f"✓ Pattern recognition analysis completed with {len(pattern_analysis['patterns'])} patterns detected")
    except Exception as e:
        print(f"✗ Error in pattern recognition: {str(e)}")
        results['pattern_analysis'] = {"error": str(e)}
    
    # 3. Analisis konteks pasar
    try:
        market_context = analyze_market_context(df)
        results['market_context'] = market_context
        print(f"✓ Market context analysis completed: {market_context['market_phase']['phase']} detected")
    except Exception as e:
        print(f"✗ Error in market context analysis: {str(e)}")
        results['market_context'] = {"error": str(e)}
    
    # 4. Analisis manajemen risiko
    try:
        current_price = df['close'].iloc[-1]
        action = technical_analysis.get('total_buy', 0) > technical_analysis.get('total_sell', 0)
        action_str = "BUY" if action else "SELL"
        
        risk_analysis = analyze_risk_management(df, crypto_symbol, current_price, action_str, account_balance, risk_percent)
        results['risk_analysis'] = risk_analysis
        print(f"✓ Risk management analysis completed: {risk_analysis['risk_profile']['risk_level']} risk profile")
    except Exception as e:
        print(f"✗ Error in risk management analysis: {str(e)}")
        results['risk_analysis'] = {"error": str(e)}
    
    # 5. Analisis prediksi machine learning
    # ML adalah opsional, jadi kita periksa apakah modul tersedia
    try:
        ml_prediction = analyze_ml_prediction(df, crypto_symbol)
        results['ml_prediction'] = ml_prediction
        print(f"✓ ML prediction analysis completed: {ml_prediction['prediction']['prediction']} with {ml_prediction['prediction']['confidence']:.1f}% confidence")
    except Exception as e:
        print(f"✗ Error in ML prediction analysis: {str(e)}")
        results['ml_prediction'] = {"error": str(e)}
    
    # 6. Saran AI (integrasi semua analisis)
    try:
        ai_advice = provide_ai_advice(crypto_symbol, timeframe, results)
        results['ai_advice'] = ai_advice
        print(f"✓ AI advice generated: {ai_advice['action']} with {ai_advice['confidence']:.1f}% confidence")
    except Exception as e:
        print(f"✗ Error in AI advice generation: {str(e)}")
        results['ai_advice'] = {"error": str(e)}
    
    # Catat waktu eksekusi
    execution_time = time.time() - start_time
    results['execution_time'] = execution_time
    print(f"Comprehensive analysis completed in {execution_time:.2f} seconds")
    
    return results

def format_advanced_analysis_output(analysis_results, crypto_symbol, timeframe):
    """
    Format hasil analisis lanjutan menjadi output yang terstruktur
    """
    output = ""
    
    # Format AI Advice
    ai_advice = analysis_results.get('ai_advice', {})
    if ai_advice and 'error' not in ai_advice:
        output += f"\n=== AI TRADING ADVISOR ({crypto_symbol}/{timeframe}) ===\n"
        action = ai_advice.get('action', 'UNKNOWN')
        confidence = ai_advice.get('confidence', 0)
        
        # Tentukan warna berdasarkan aksi
        if action in ['STRONG_BUY', 'BUY']:
            action_color = 'green'
        elif action in ['SELL', 'STRONG_SELL']:
            action_color = 'red'
        else:
            action_color = 'yellow'
        
        from termcolor import colored
        output += f"Recommended Action: {colored(action, action_color, attrs=['bold'])}\n"
        output += f"Confidence: {colored(f'{confidence:.1f}%', 'cyan')}\n\n"
        
        # Tampilkan Saran
        output += colored("Trading Advice:\n", 'yellow', attrs=['bold'])
        for advice_item in ai_advice.get('advice', []):
            output += f"• {advice_item}\n"
            
        # Tampilkan Kontribusi Analisis
        contributors = ai_advice.get('contributors', {})
        output += "\n" + colored("Analysis Contributors:\n", 'yellow', attrs=['bold'])
        output += f"• Technical Analysis: {contributors.get('technical', 0):.1f}\n"
        output += f"• Market Context: {contributors.get('market', 0):.1f}\n"
        output += f"• Pattern Recognition: {contributors.get('patterns', 0):.1f}\n"
        output += f"• Machine Learning: {contributors.get('machine_learning', 0):.1f}\n"
    
    # Format Pattern Recognition
    pattern_analysis = analysis_results.get('pattern_analysis', {})
    if pattern_analysis and 'error' not in pattern_analysis:
        patterns = pattern_analysis.get('patterns', [])
        if patterns:
            output += "\n" + colored("Detected Patterns:\n", 'magenta', attrs=['bold'])
            for pattern in patterns:
                signal = pattern.get('signal', 'neutral')
                signal_color = 'green' if signal == 'bullish' else ('red' if signal == 'bearish' else 'white')
                output += f"• {colored(pattern.get('type', 'unknown').title(), signal_color)}: {pattern.get('description', '')}\n"
    
    # Format Market Context
    market_context = analysis_results.get('market_context', {})
    if market_context and 'error' not in market_context:
        market_phase = market_context.get('market_phase', {})
        output += "\n" + colored("Market Context:\n", 'cyan', attrs=['bold'])
        output += f"• Market Phase: {market_phase.get('phase', 'unknown').title()} ({market_phase.get('description', '')})\n"
        output += f"• Volatility: {market_phase.get('atr_percent', 0):.2f}%\n"
        
        # Support/Resistance
        sr_levels = market_context.get('support_resistance', {})
        supports = sr_levels.get('supports', [])
        resistances = sr_levels.get('resistances', [])
        
        if supports:
            output += colored("\nSupport Levels:\n", 'green')
            for i, support in enumerate(supports[:3]):
                output += f"• S{i+1}: ${support.get('level', 0):.2f} (Strength: {support.get('strength', 0):.0f}%)\n"
                
        if resistances:
            output += colored("\nResistance Levels:\n", 'red')
            for i, resistance in enumerate(resistances[:3]):
                output += f"• R{i+1}: ${resistance.get('level', 0):.2f} (Strength: {resistance.get('strength', 0):.0f}%)\n"
    
    # Format Risk Management
    risk_analysis = analysis_results.get('risk_analysis', {})
    if risk_analysis and 'error' not in risk_analysis:
        output += "\n" + colored("Risk Management:\n", 'red', attrs=['bold'])
        
        risk_profile = risk_analysis.get('risk_profile', {})
        output += f"• Risk Profile: {risk_profile.get('risk_level', 'medium').title()}\n"
        
        position_info = risk_analysis.get('position_info', {})
        if position_info:
            output += f"• Recommended Position Size: ${position_info.get('position_size', 0):.2f}\n"
            output += f"• Risk Amount: ${position_info.get('risk_amount', 0):.2f}\n"
        
        risk_levels = risk_analysis.get('risk_levels', {})
        if risk_levels:
            sl = risk_levels.get('stop_loss', {})
            tp = risk_levels.get('take_profit', {})
            rr = risk_levels.get('risk_reward', {})
            
            output += colored("\nRecommended Levels:\n", 'yellow')
            output += f"• Stop Loss: ${sl.get('normal', 0):.2f} (Tight: ${sl.get('tight', 0):.2f}, Wide: ${sl.get('wide', 0):.2f})\n"
            output += f"• Take Profit: ${tp.get('moderate', 0):.2f} (Conservative: ${tp.get('conservative', 0):.2f}, Aggressive: ${tp.get('aggressive', 0):.2f})\n"
            output += f"• Risk:Reward Ratio: {rr.get('moderate', 0):.2f} (Conservative: {rr.get('conservative', 0):.2f}, Aggressive: {rr.get('aggressive', 0):.2f})\n"
    
    # Format ML Prediction
    ml_prediction = analysis_results.get('ml_prediction', {})
    if ml_prediction and 'error' not in ml_prediction:
        output += "\n" + colored("Machine Learning Prediction:\n", 'blue', attrs=['bold'])
        
        prediction = ml_prediction.get('prediction', {})
        direction = prediction.get('prediction', 'unknown')
        confidence = prediction.get('confidence', 0)
        direction_color = 'green' if direction == 'UP' else ('red' if direction == 'DOWN' else 'white')
        
        output += f"• Predicted Direction: {colored(direction, direction_color)}\n"
        output += f"• Confidence: {confidence:.1f}%\n"
        output += f"• Model Freshness: {ml_prediction.get('model_freshness', 'unknown')}\n"
        
        # Insights
        if 'ml_insights' in ml_prediction:
            output += "\n" + colored("ML Insights:\n", 'blue')
            for insight in ml_prediction.get('ml_insights', []):
                output += f"• {insight}\n"
    
    # Execution Time
    output += f"\nExecution Time: {analysis_results.get('execution_time', 0):.2f} seconds\n"
    
    return output