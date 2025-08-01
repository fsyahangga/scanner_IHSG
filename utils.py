import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# ---------------------------
# Technical Indicator Section
# ---------------------------

def calculate_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['Stoch'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    
    bb = ta.volatility.BollingerBands(close=df['close'])
    df['BB_bbm'] = bb.bollinger_mavg()
    df['BB_bbh'] = bb.bollinger_hband()
    df['BB_bbl'] = bb.bollinger_lband()
    
    df['Volume_Spike'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5
    
    return df

# ---------------------------
# Fundamental Placeholder
# ---------------------------

def get_fundamental_metrics(ticker):
    # Simulasi ambil data PER, PBV
    return {
        "PER": np.random.uniform(5, 20),
        "PBV": np.random.uniform(0.5, 3),
    }

# ---------------------------
# Bandarmology Placeholder
# ---------------------------

def calculate_bandarmology_score(ticker):
    return round(np.random.uniform(0, 1), 2)

# ---------------------------
# Macro Sentiment (Static/Mocked)
# ---------------------------

def get_macro_sentiment():
    # TODO: Replace with real API or macro scraper
    return "POSITIVE"

# ---------------------------
# Candlestick Pattern Detection (Simple)
# ---------------------------

def detect_candlestick_pattern(df):
    patterns = []
    for _, row in df.iterrows():
        if row['RSI'] < 30 and row['Stoch'] < 20:
            patterns.append("Hammer")
        elif row['RSI'] > 70 and row['Stoch'] > 80:
            patterns.append("Shooting Star")
        else:
            patterns.append("NoPattern")
    return patterns

# ---------------------------
# Preprocessing Utilities
# ---------------------------

def scale_features(X, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# ---------------------------
# Save/Load Utility
# ---------------------------

def save_model_and_scaler(model, scaler, name_prefix):
    joblib.dump(model, f"models/{name_prefix}_model.pkl")
    joblib.dump(scaler, f"models/{name_prefix}_scaler.pkl")

def load_model_and_scaler(name_prefix):
    model = joblib.load(f"models/{name_prefix}_model.pkl")
    scaler = joblib.load(f"models/{name_prefix}_scaler.pkl")
    return model, scaler
