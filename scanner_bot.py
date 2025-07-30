import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from dotenv import load_dotenv
from datetime import datetime,timedelta
import ta
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator

# --- Load Env ---
load_dotenv()
TICKERS = os.getenv("FILTER_TICKER", "").split(",")

# --- Fungsi ambil real-time ---
def get_realtime_data(tickers):
    all_rows = []

    for ticker in tickers:
        try:
            ohlcv = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
            if isinstance(ohlcv.columns, pd.MultiIndex):
                ohlcv.columns = [col[0] if isinstance(col, tuple) else col for col in ohlcv.columns]
            if ohlcv.empty or "Close" not in ohlcv.columns:
                print(f"Data kosong/tidak valid: {ticker}")
                continue

            df = ohlcv.copy().reset_index()
            df.rename(columns={
                "Date": "date",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            # ✅ Perbaiki kolom jika MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.map(lambda x: x[0])

            # ✅ Ambil data 1D
            close_val = df["close"].values[-1]                
            df["ticker"] = ticker

            # Indikator teknikal
            df["RSI"] = RSIIndicator(close=close_val).rsi()

            macd = MACD(close=close_val)
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_hist"] = macd.macd_diff()

            df["EMA_20"] = EMAIndicator(close=close_val, window=20).ema_indicator()

            adx = ADXIndicator(high=df["High"], low=df["Low"], close=close_val, window=14)
            df["ADX"] = adx.adx()

            print(f"✅ Data berhasil diproses: {ticker}")
            all_rows.append(df[["ticker", "date", "close", "volume", "RSI", "MACD", "MACD_signal", "MACD_hist", "EMA_20", "ADX"]])

        except Exception as e:
            print(f"❌ isi data: {ohlcv}: {e}")
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

def clean_and_standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Standardisasi nama kolom (hilangkan spasi dan lowercase semua)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Deteksi kolom tanggal
    possible_date_cols = ["date", "datetime", "tanggal", "time"]
    date_col = next((col for col in possible_date_cols if col in df.columns), None)

    if not date_col:
        raise KeyError("❌ Kolom tanggal tidak ditemukan di dataset.")

    # Konversi kolom tanggal ke datetime.date
    df["date"] = pd.to_datetime(df[date_col]).dt.date

    # Hapus kolom tanggal lain jika bukan 'date'
    if date_col != "date":
        df.drop(columns=[date_col], inplace=True)

    # Drop duplikat berdasarkan ticker + date
    if "ticker" in df.columns:
        df.drop_duplicates(subset=["ticker", "date"], inplace=True)

    # Optional: Drop kolom tidak relevan (jika ada)
    drop_cols = [col for col in df.columns if col.startswith("unnamed") or col.strip() == ""]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Sortir berdasarkan ticker dan date
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    return df


# --- Load data historis dan realtime ---
historical_df = pd.read_csv("historical_idx_dataset.csv")

# 1. Ambil realtime data dari yfinance
realtime_df = get_realtime_data(TICKERS)  # hasilnya harus ada ['ticker', 'date', 'Close', 'Volume']
if realtime_df.empty:
    print("Data frame kosong. Tidak ada data berhasil diambil.")
    exit()
# 2. Untuk setiap ticker, hitung indikator teknikal
df_list = []
for ticker in realtime_df["ticker"].unique():
    df = realtime_df[realtime_df["ticker"] == ticker].copy()
    df = calculate_technical_indicators(df)  # hasilnya: + RSI, Stoch, BB_bbm, BB_bbh, BB_bbl
    df_list.append(df)

# 3. Gabungkan semua jadi satu DataFrame
new_data = pd.concat(df_list).dropna().reset_index(drop=True)

# 4. Tambahkan label `target` (opsional, tergantung definisi target kamu)
new_data["target"] = 0  # placeholder / bisa pakai strategi label lain

# 5. Gabungkan dengan historical
historical = pd.read_csv("historical_idx_dataset.csv")
final_df = pd.concat([historical, new_data], ignore_index=True)

# 6. Simpan kembali
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
    if X_scaled.ndim != 2:
        X_scaled = X_scaled.reshape(-1, X_raw.shape[1])  # jaga-jaga

    lstm_input = np.expand_dims(X_scaled, axis=1)  # shape = (n_samples, 1, n_features)
    lstm_preds = lstm_model.predict(lstm_input).flatten()

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

# ✅ Tambahkan kolom header meskipun kosong
if df_signals.empty:
    df_signals = pd.DataFrame(columns=["ticker", "confidence", "PER", "PBV", "bandarmology_score", "date", "signal"])
if df_signals.empty:
    print("⚠️ Tidak ada sinyal BUY hari ini.")
df_signals.to_csv("buy_signals.csv", index=False)
print(f"✅ {len(df_signals)} sinyal BUY disimpan.")

