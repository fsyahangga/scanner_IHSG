import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from dotenv import load_dotenv
from datetime import datetime,timedelta
import ta
import yfinance as yf

# --- Load Env ---
load_dotenv()
TICKERS = os.getenv("FILTER_TICKER", "").split(",")

# Load historical dataset
historical_df = pd.read_csv("historical_idx_dataset.csv")
historical_df = historical_df[historical_df['ticker'].isin(TICKERS)]

# --- Fungsi ambil real-time ---
def get_realtime_data(tickers):
    all_rows = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2d", interval="1d", progress=False)
            df = df.reset_index()
            df["ticker"] = ticker
            df.rename(columns={
                "Date": "date",
                "Close": "Close",
                "Volume": "Volume"
            }, inplace=True)
            all_rows.append(df[["ticker", "date", "Close", "Volume"]])
        except Exception as e:
            print(f"❌ Gagal ambil data {ticker}: {e}")
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

def calculate_technical_indicators(df):
    df = df.copy()
    df = df.sort_values("date")  # Urut berdasarkan waktu

    # --- Momentum Indicators ---
    df["RSI"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
    df["Stoch"] = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"]
    ).stoch()
    macd = ta.trend.MACD(close=df["close"])
    df["MACD"] = macd.macd()

    # --- Volatility Indicator ---
    bb = ta.volatility.BollingerBands(close=df["close"])
    df["BB_bbm"] = bb.bollinger_mavg()
    df["BB_bbh"] = bb.bollinger_hband()
    df["BB_bbl"] = bb.bollinger_lband()

    # --- Trend Indicators ---
    df["EMA_12"] = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    df["EMA_26"] = ta.trend.EMAIndicator(close=df["close"], window=26).ema_indicator()
    df["ADX"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"]
    ).adx()

    # --- Volume Indicator ---
    df["Volume_Spike"] = df["volume"] / df["volume"].rolling(window=20).mean()

    return df

# --- Ambil data realtime ---
realtime_df = get_realtime_data(TICKERS)
# --- Gabungkan ke historis ---
combined_df = pd.merge(historical_df, realtime_df[["ticker", "date", "Close", "Volume"]], on="ticker", how="left")

# Ganti nilai fitur real-time (optional)
combined_df["latest_close"] = combined_df["Close"]
combined_df["latest_volume"] = combined_df["Volume"]

# Drop null (jika ada ticker gagal)
combined_df = combined_df.dropna(subset=["latest_close"])

# Hitung indikator teknikal berbasis OHLC untuk setiap ticker
df_list = []
for ticker in combined_df["ticker"].unique():
    df = combined_df[combined_df["ticker"] == ticker].copy()
    df = calculate_technical_indicators(df)
    df_list.append(df)

# Gabungkan kembali
final_df = pd.concat(df_list).dropna().reset_index(drop=True)

# Simpan ke historical_idx_dataset.csv
final_df.to_csv("historical_idx_dataset.csv", index=False)
print("✅ Dataset dengan indikator teknikal OHLC disimpan.")
print("✅ Dataset berhasil diperbarui dan disimpan ke historical_idx_dataset.csv.")


# --- Load Trained Models ---
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")

# --- Optional Scaler (jika digunakan saat training) ---
try:
    scaler = joblib.load("models/feature_scaler.pkl")
except:
    scaler = None

# --- Load Dataset ---
data = pd.read_csv("historical_idx_dataset.csv")
data = data[data["ticker"].isin(TICKERS)]

# --- Features ---
FEATURES = [
    "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl", "Volume_Spike",
    "PER", "PBV", "bandarmology_score",
    "latest_close", "latest_volume"
]

TARGET = "target"

# --- Hybrid Inference ---
def load_models_from_folds(file_path):
    models = []
    with open(file_path, "r") as f:
        for line in f:
            model_path = line.strip()
            models.append(joblib.load(model_path))
    return models

rf_models = load_models_from_folds("models/rf_model_folds.txt")
xgb_models = load_models_from_folds("models/xgb_model_folds.txt")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
scaler = joblib.load("models/feature_scaler.pkl")

def predict_with_confidence(X_raw):
    X_scaled = scaler.transform(X_raw)
    lstm_preds = lstm_model.predict(np.expand_dims(X_scaled, axis=1)).flatten()

    rf_preds = np.mean([m.predict_proba(X_scaled)[:, 1] for m in rf_models], axis=0)
    xgb_preds = np.mean([m.predict_proba(X_scaled)[:, 1] for m in xgb_models], axis=0)

    hybrid = (rf_preds + xgb_preds + lstm_preds) / 3
    return hybrid


# --- Filter Conditions ---
def pass_fundamental(row):
    return (row["PER"] < 15) and (row["PBV"] < 2)

def pass_bandarmology(row):
    return row["bandarmology_score"] >= 7

# --- Run Screening ---
signal_results = []

for ticker in data["ticker"].unique():
    df_ticker = data[data["ticker"] == ticker].copy()
    if df_ticker.empty:
        continue

    latest_row = df_ticker.tail(1).iloc[0]
    if not (pass_fundamental(latest_row) and pass_bandarmology(latest_row)):
        continue

    X = df_ticker[FEATURES].tail(1)
    confidence = predict_with_confidence(X)[0]

    if confidence > 0.7:
        signal_results.append({
            "ticker": ticker,
            "confidence": round(confidence, 4),
            "PER": latest_row["PER"],
            "PBV": latest_row["PBV"],
            "bandarmology_score": latest_row["bandarmology_score"],
            "date": datetime.today().strftime("%Y-%m-%d"),
            "signal": "BUY"
        })

# --- Save Results ---
df_signals = pd.DataFrame(signal_results)
df_signals.to_csv("buy_signals.csv", index=False)
print(f"✅ {len(df_signals)} sinyal BUY disimpan.")
