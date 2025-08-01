import pandas as pd
import numpy as np
import talib
import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import os

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ===============================
# 1. Technical Indicators
# ===============================
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"])
    df["Stoch"] = slowk
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    df["BB_bbh"] = upper
    df["BB_bbm"] = middle
    df["BB_bbl"] = lower

    df["Volume_Spike"] = df["volume"] / df["volume"].rolling(20).mean()
    df["Volume_Spike"] = df["Volume_Spike"].replace([np.inf, -np.inf], 0).fillna(0)

    return df


# ===============================
# 2. Candlestick Pattern Detection
# ===============================
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    patterns = {
        "Hammer": talib.CDLHAMMER,
        "ShootingStar": talib.CDLSHOOTINGSTAR,
        "MorningStar": talib.CDLMORNINGSTAR,
        "EveningStar": talib.CDLEVENINGSTAR,
        "Doji": talib.CDLDOJI
    }

    for name, func in patterns.items():
        df[name] = func(df["open"], df["high"], df["low"], df["close"])

    return df


# ===============================
# 3. Dummy Macro Sentiment Score
# ===============================
def get_macro_sentiment(bi_rate: float, inflation: float, usd_idr: float) -> float:
    score = 1.0
    if bi_rate > 6.5: score -= 0.3
    if inflation > 4.0: score -= 0.3
    if usd_idr > 16000: score -= 0.4
    return max(0.0, min(score, 1.0))


# ===============================
# 4. Dummy Bandarmology Score
# ===============================
def calculate_bandarmology_score(df: pd.DataFrame) -> pd.DataFrame:
    buy_vol = df.get("top_broker_buy_volume", pd.Series(0))
    sell_vol = df.get("top_broker_sell_volume", pd.Series(1))
    net_buy = buy_vol - sell_vol
    df["Bandarmology_Score"] = (net_buy / (sell_vol + 1)).replace([np.inf, -np.inf], 0).fillna(0)
    return df


# ===============================
# 5. Load Model & Scaler
# ===============================
def load_model_and_scaler():
    try:
        model = joblib.load("models/random_forest_model.pkl")
        scaler = joblib.load("models/random_forest_scaler.pkl")
    except Exception:
        # fallback to hybrid keras if needed
        model = load_model("models/hybrid_lstm_model.h5")
        scaler = joblib.load("models/hybrid_lstm_scaler.pkl")
    return model, scaler


# ===============================
# 6. Prediction Function
# ===============================
def make_prediction(model, scaler, df_input: pd.DataFrame):
    FEATURES = [
        "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl",
        "Volume_Spike", "PER", "PBV", "Bandarmology_Score",
        "latest_close", "latest_volume", "macro_sentiment"
    ]

    X = df_input[FEATURES].fillna(0).copy()
    X_scaled = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0][1]
        pred = model.predict(X_scaled)[0]
    else:
        # Assume it's a keras model
        proba = model.predict(X_scaled, verbose=0)[0][0]
        pred = int(proba > 0.5)

    return proba, pred
