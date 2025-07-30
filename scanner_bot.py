import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load pre-trained models
rf_model = joblib.load("random_forest_model.pkl")
dnn_model = load_model("deep_learning_model.h5")
scaler = joblib.load("scaler.pkl")

# Example IDX tickers
IDX_TICKERS = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "ARTO.JK"]

def get_technical_indicators(df):
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['Stoch'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    bb = BollingerBands(close=df['Close'])
    df['BB_bbm'] = bb.bollinger_mavg()
    df['BB_bbh'] = bb.bollinger_hband()
    df['BB_bbl'] = bb.bollinger_lband()
    df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    return df.dropna()

def extract_features(df):
    return df[['RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike']].values[-1].reshape(1, -1)

def get_prediction_signals(stock):
    df = yf.download(stock, period="6mo", interval="1d")
    df = get_technical_indicators(df)
    X = scaler.transform(extract_features(df))
    rf_pred = rf_model.predict(X)[0]
    dnn_pred = np.argmax(dnn_model.predict(X), axis=1)[0]

    consensus = "🔴 Sell"
    if rf_pred == 1 and dnn_pred == 1:
        consensus = "🟢 Buy"
    elif rf_pred == 1 or dnn_pred == 1:
        consensus = "🟡 Neutral"

    return f"{stock}: {consensus} (RF={rf_pred}, DNN={dnn_pred})"
# Filtering dictionary
IDX_FILTER = {
    "perbankan_keuangan": {
        "tickers": ["BBRI.JK", "BMRI.JK", "BBCA.JK", "BBNI.JK", "BRIS.JK"],
        "valuasi": "murah",
        "ai_score": 8.5
    },
    "teknologi_digital": {
        "tickers": ["ARTO.JK"],
        "valuasi": "tidak terdefinisi",
        "ai_score": 6.5
    }
}

def filter_by_sector(sector_name):
    """Return tickers based on sector name from IDX_FILTER."""
    return IDX_FILTER.get(sector_name, {}).get("tickers", [])

def filter_by_valuasi(level="murah"):
    """Return tickers by valuasi level: 'murah', 'sedang', 'premium'."""
    return [ticker
            for info in IDX_FILTER.values()
            if info["valuasi"] == level
            for ticker in info["tickers"]]

def filter_by_ai_score(min_score=8.0):
    """Return tickers with AI predictive score >= min_score."""
    return [ticker
            for info in IDX_FILTER.values()
            if info["ai_score"] >= min_score
            for ticker in info["tickers"]]

# Contoh penggunaan (tidak dijalankan saat __main__)
# print(filter_by_sector("perbankan_keuangan"))
# print(filter_by_valuasi("murah"))
# print(filter_by_ai_score(8.0))

def run_scanner():
    print("📊 Daily Stock Recommendation (AI-Powered):")
    for stock in IDX_TICKERS:
        try:
            signal = get_prediction_signals(stock)
            print(signal)
        except Exception as e:
            print(f"{stock}: Error - {e}")

if __name__ == "__main__":
    run_scanner()
