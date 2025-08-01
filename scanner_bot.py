import pandas as pd
from utils import (
    calculate_technical_indicators,
    detect_candlestick_patterns,
    get_macro_sentiment,
    calculate_bandarmology_score,
    load_model_and_scaler,
    make_prediction
)

# Load data historis (kolom: date, ticker, open, high, low, close, volume, dll)
df = pd.read_csv("historical_idx_dataset.csv")

# Dapatkan list ticker unik
tickers = df["ticker"].unique()
results = []

# Dummy nilai makro (harus diganti otomatis nanti)
BI_RATE = 6.0
INFLATION = 3.2
USD_IDR = 15800
macro_score = get_macro_sentiment(BI_RATE, INFLATION, USD_IDR)

# Load model hybrid dan scaler
model, scaler = load_model_and_scaler()

# Loop setiap ticker
for ticker in tickers:
    df_ticker = df[df["ticker"] == ticker].copy()
    df_ticker.sort_values("date", inplace=True)

    # Tambahkan indikator teknikal
    df_ticker = calculate_technical_indicators(df_ticker)

    # Tambahkan pola candlestick
    df_ticker = detect_candlestick_patterns(df_ticker)

    # Tambahkan skor bandarmology dummy
    df_ticker["foreign_net_buy"] = df_ticker.get("foreign_net_buy", 0)
    df_ticker["top_broker_buy_volume"] = df_ticker.get("top_broker_buy_volume", 0)
    df_ticker["top_broker_sell_volume"] = df_ticker.get("top_broker_sell_volume", 0)
    df_ticker = calculate_bandarmology_score(df_ticker)

    # Ambil baris terakhir (terbaru)
    latest = df_ticker.iloc[-1:]

    # Tambahkan makro sentimen ke dataframe
    latest["macro_sentiment"] = macro_score

    # Prediksi
    proba, pred = make_prediction(model, scaler, latest)

    # Tentukan rekomendasi
    if proba > 0.65 and pred == 1:
        recommendation = "BUY"
    elif proba < 0.35 and pred == 0:
        recommendation = "SELL"
    else:
        recommendation = "NEUTRAL"

    results.append({
        "ticker": ticker,
        "prediction": int(pred),
        "confidence": round(float(proba), 4),
        "macro_sentiment": round(macro_score, 4),
        "bandarmology_score": round(float(latest["Bandarmology_Score"].values[0]), 4),
        "recommendation": recommendation
    })

# Simpan hasil ke CSV
result_df = pd.DataFrame(results)
result_df.to_csv("buy_signals.csv", index=False)

# Tampilkan hasil
print(result_df.sort_values("confidence", ascending=False))
