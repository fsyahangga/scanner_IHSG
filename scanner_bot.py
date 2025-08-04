# scanner_bot.py - Versi Stabil dan Terintegrasi
import pandas as pd
from utils import (
    calculate_indicators,
    scale_features,
    load_model_and_scaler,
    generate_recommendation,
    save_buy_signals
)
import numpy as np

# ------------------------------
# Load model dan scaler
# ------------------------------
model, scaler = load_model_and_scaler("rf")

# ------------------------------
# Load latest/realtime data
# ------------------------------
df = pd.read_csv("latest_realtime_data.csv")

if 'ticker' not in df.columns:
    raise ValueError("Kolom 'ticker' wajib ada di file input!")

# Pastikan ticker bersih
if df['ticker'].str.contains('.JK').any():
    df['ticker'] = df['ticker'].str.replace('.JK', '', regex=False)

# ------------------------------
# Hitung indikator teknikal
# ------------------------------
df = calculate_indicators(df)
df.dropna(inplace=True)

# ------------------------------
# Pastikan semua kolom wajib ada
# ------------------------------
model_features = [
    'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike',
    'PER', 'PBV', 'bandarmology_score', 'latest_close', 'latest_volume'
]
extra_columns = [
    'ticker', 'Foreign_Buy_Ratio', 'macro_sentiment', 'candlestick_pattern'
]

for col in model_features + extra_columns:
    if col not in df.columns:
        df[col] = 0

# Hitung Volume Spike jika belum ada
if 'Volume_Spike' not in df.columns or df['Volume_Spike'].isnull().all():
    df['Volume_Spike'] = (df['latest_volume'] > df['latest_volume'].rolling(5).mean()).astype(int)
else:
    df['Volume_Spike'] = df['Volume_Spike'].astype(int)

# ------------------------------
# Preprocessing dan prediksi
# ------------------------------
X = df[model_features]
X_scaled, _ = scale_features(X, method='standard', scaler=scaler)

# Prediksi
try:
    df['prediction'] = model.predict(X_scaled)
    df['probability'] = model.predict_proba(X_scaled)[:, 1]
except Exception as e:
    print(f"❌ Error prediksi: {e}")
    df['prediction'] = 0
    df['probability'] = 0.0

# Generate rekomendasi
try:
    df['recommendation'] = df['prediction'].apply(generate_recommendation)
except:
    df['recommendation'] = "NEUTRAL"

# ------------------------------
# Simpan hasil
# ------------------------------
output_df = df[extra_columns + model_features + ['probability', 'recommendation']]
save_buy_signals(output_df, output_path="buy_signals.csv")

# ------------------------------
# Tampilkan hasil akhir
# ------------------------------
print("\n✅ Selesai. Rekomendasi disimpan ke buy_signals.csv")
print(output_df[['ticker', 'RSI', 'Stoch', 'PER', 'PBV', 'bandarmology_score',
                 'Foreign_Buy_Ratio', 'macro_sentiment', 'candlestick_pattern', 'recommendation']])
