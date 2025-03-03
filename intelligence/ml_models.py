# ml_models.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

def prepare_features(df, lookback_periods=[5, 10, 20]):
    """
    Menyiapkan fitur untuk model ML dari data OHLCV
    lookback_periods: list periode untuk fitur historis
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Basic price and volume features
    data['price_change'] = data['close'].pct_change() * 100
    data['volume_change'] = data['volume'].pct_change() * 100
    data['high_low_diff'] = ((data['high'] - data['low']) / data['low']) * 100
    data['open_close_diff'] = ((data['close'] - data['open']) / data['open']) * 100
    
    # Moving averages
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    # Distance from moving averages
    data['sma_10_dist'] = ((data['close'] - data['sma_10']) / data['sma_10']) * 100
    data['sma_20_dist'] = ((data['close'] - data['sma_20']) / data['sma_20']) * 100
    data['sma_50_dist'] = ((data['close'] - data['sma_50']) / data['sma_50']) * 100
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    data['bb_std'] = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
    data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
    data['bb_width'] = ((data['bb_upper'] - data['bb_lower']) / data['bb_middle']) * 100
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # RSI calculation
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Historical features for each lookback period
    for period in lookback_periods:
        # Price momentum features
        data[f'price_momentum_{period}'] = data['close'].pct_change(periods=period) * 100
        
        # Volume momentum features
        data[f'volume_momentum_{period}'] = data['volume'].pct_change(periods=period) * 100
        
        # Volatility features
        data[f'volatility_{period}'] = data['price_change'].rolling(window=period).std()
        
        # Price direction features (proportion of up days)
        data[f'up_days_ratio_{period}'] = data['price_change'].rolling(window=period).apply(
            lambda x: (x > 0).sum() / period, raw=True
        )
    
    # Target variable: price direction in next period
    # 1 if price increases, 0 if price decreases
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    # Drop NaN values
    data = data.dropna()
    
    # Return processed data
    return data

def train_model(df, test_size=0.2, random_state=42, save_path=None):
    """
    Train a machine learning model to predict price movement direction
    """
    # Prepare features
    processed_data = prepare_features(df)
    
    # Define features and target
    feature_columns = [col for col in processed_data.columns if col not in 
                      ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
    
    X = processed_data[feature_columns]
    y = processed_data['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=random_state
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:5]
    
    # Save model if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'training_date': datetime.now().strftime('%Y-%m-%d')
        }, save_path)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'accuracy': accuracy,
        'top_features': top_features
    }

def load_model(model_path):
    """
    Load trained model and associated objects
    """
    try:
        model_data = joblib.load(model_path)
        return model_data
    except:
        return None

def predict_price_movement(df, model_data=None, model_path=None):
    """
    Predict future price movement using trained model
    """
    # If model_data is not provided, load from path
    if model_data is None and model_path:
        model_data = load_model(model_path)
        
    if model_data is None:
        # Train a new model if none is provided
        model_data = train_model(df)
    
    # Extract model components
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Prepare latest data for prediction
    processed_data = prepare_features(df)
    
    # Get the most recent record
    latest_data = processed_data.iloc[-1:][feature_columns]
    
    # Scale features
    latest_data_scaled = scaler.transform(latest_data)
    
    # Get prediction (probability of price increase)
    prediction_proba = model.predict_proba(latest_data_scaled)[0]
    
    # Get class probabilities
    down_prob = prediction_proba[0]  # Probability of price decrease
    up_prob = prediction_proba[1]    # Probability of price increase
    
    # Calculate confidence
    confidence = max(up_prob, down_prob) * 100
    
    # Determine prediction direction
    prediction = 'UP' if up_prob > down_prob else 'DOWN'
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'up_probability': up_prob * 100,
        'down_probability': down_prob * 100,
        'model_accuracy': model_data.get('accuracy', 'Unknown')
    }

def analyze_ml_prediction(df, crypto_symbol, retrain=False, model_path=None):
    """
    Perform complete ML analysis and return insights
    """
    # Define default model path if not provided
    if model_path is None:
        model_path = f"models/{crypto_symbol}_model.joblib"
    
    # Check if model exists
    model_exists = os.path.exists(model_path)
    
    if model_exists and not retrain:
        # Load existing model
        model_data = load_model(model_path)
        
        # Check model age
        if model_data and 'training_date' in model_data:
            training_date = datetime.strptime(model_data['training_date'], '%Y-%m-%d')
            days_since_training = (datetime.now() - training_date).days
            
            # If model is older than 30 days, suggest retraining
            model_freshness = "Recent" if days_since_training < 30 else "Outdated"
        else:
            model_freshness = "Unknown"
    else:
        # Train new model
        model_data = train_model(df, save_path=model_path)
        model_freshness = "Fresh (just trained)"
    
    # Get prediction
    prediction = predict_price_movement(df, model_data)
    
    # Generate insights based on prediction and model quality
    ml_insights = []
    
    if prediction['confidence'] > 70:
        insight_text = f"Strong ML signal ({prediction['confidence']:.1f}% confidence) for price to move {prediction['prediction']}"
        ml_insights.append(insight_text)
    elif prediction['confidence'] > 55:
        insight_text = f"Moderate ML signal ({prediction['confidence']:.1f}% confidence) for price to move {prediction['prediction']}"
        ml_insights.append(insight_text)
    else:
        ml_insights.append(f"Weak ML signal ({prediction['confidence']:.1f}% confidence) - direction uncertain")
    
    # Add model quality insights
    if isinstance(model_data.get('accuracy'), (int, float)):
        model_accuracy = model_data['accuracy'] * 100
        if model_accuracy < 55:
            ml_insights.append(f"Caution: Model accuracy is low ({model_accuracy:.1f}%)")
    
    ml_insights.append(f"Model freshness: {model_freshness}")
    
    # Calculate ML-based signals
    ml_buy_strength = 0
    ml_sell_strength = 0
    
    if prediction['prediction'] == 'UP':
        ml_buy_strength = prediction['confidence'] * (model_data.get('accuracy', 0.6) if isinstance(model_data.get('accuracy'), (int, float)) else 0.6)
    else:
        ml_sell_strength = prediction['confidence'] * (model_data.get('accuracy', 0.6) if isinstance(model_data.get('accuracy'), (int, float)) else 0.6)
    
    return {
        'prediction': prediction,
        'model_freshness': model_freshness,
        'ml_insights': ml_insights,
        'ml_buy_strength': ml_buy_strength,
        'ml_sell_strength': ml_sell_strength,
        'top_features': model_data.get('top_features', [])
    }