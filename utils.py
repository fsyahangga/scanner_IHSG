import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# ---------------------------
# Technical Indicator Section
# ---------------------------

def calculate_indicators(df):
    df.columns = df.columns.str.lower()  # normalize column names
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
def calculate_macro_sentiment(bi_rate, inflation, usd_idr, pmi, cpi):
    score = 0
    
    if bi_rate <= 6:
        score += 0.2
    else:
        score -= 0.2

    if inflation < 3.5:
        score += 0.2
    else:
        score -= 0.2

    if usd_idr < 15500:
        score += 0.2
    else:
        score -= 0.2

    if pmi >= 50:
        score += 0.2
    else:
        score -= 0.2

    if cpi <= 3:
        score += 0.2
    else:
        score -= 0.2

    return round(score, 2)

def calculate_foreign_flow(buy_foreign, sell_foreign):
    try:
        net = buy_foreign - sell_foreign
        ratio = buy_foreign / (buy_foreign + sell_foreign + 1e-9)
        return round(net), round(ratio, 2)
    except:
        return 0, 0.0

def get_macro_sentiment():
    # Dummy static values for now (replace later with API/scraper)
    return calculate_macro_sentiment(
        bi_rate=6.25,
        inflation=2.85,
        usd_idr=15480,
        pmi=51.2,
        cpi=2.9
    )

def get_foreign_flow_data(ticker):
    # Placeholder: ganti dengan hasil scraping dari RTI/EDB
    sample_data = {
        "BBRI.JK": (1_200_000_000, 600_000_000),
        "BBCA.JK": (800_000_000, 700_000_000),
        # ...
    }
    return calculate_foreign_flow(*sample_data.get(ticker, (0, 0)))


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
