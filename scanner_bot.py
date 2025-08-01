import pandas as pd
from utils import (
    calculate_indicators,
    scale_features,
    load_model_and_scaler,
    generate_recommendation,
    save_buy_signals
)

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

# ------------------------------
# Hitung indikator teknikal
# ------------------------------
df = calculate_indicators(df)
df.dropna(inplace=True)

# ------------------------------
# Pilih fitur model dan kolom tambahan
# ------------------------------
model_features = ['RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike']
extra_columns = [
    'ticker', 'PER', 'PBV', 'bandarmology_score', 'latest_close', 'latest_volume',
    'Foreign_Buy_Ratio', 'macro_sentiment', 'candlestick_pattern'
]

# Cek dan isi missing columns jika perlu
for col in extra_columns:
    if col not in df.columns:
        df[col] = 0  # atau np.nan / default value

# ------------------------------
# Preprocessing dan prediksi
# ------------------------------
X = df[model_features].copy()
X['Volume_Spike'] = X['Volume_Spike'].astype(int)

X_scaled, _ = scale_features(X, method='standard', scaler=scaler)

df['prediction'] = model.predict(X_scaled)
df['probability'] = model.predict_proba(X_scaled)[:, 1]
df['recommendation'] = df['prediction'].apply(generate_recommendation)

# ------------------------------
# Output dan simpan
# ------------------------------
cols_output = extra_columns + model_features + ['probability', 'recommendation']
output_df = df[cols_output]

save_buy_signals(output_df, output_path="buy_signals.csv")

# ------------------------------
# Tampilkan hasil
# ------------------------------
print("✅ Selesai. Rekomendasi disimpan ke buy_signals.csv\n")
print(output_df[['ticker', 'RSI', 'Stoch', 'PER', 'PBV', 'bandarmology_score',
                 'Foreign_Buy_Ratio', 'macro_sentiment', 'candlestick_pattern', 'recommendation']])
