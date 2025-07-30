import pandas as pd
import joblib
from train_model import retrain_models

# Panggil retrain otomatis saat script dijalankan
retrain_models()

# Load ulang data dan model
df = pd.read_csv("historical_idx_dataset.csv")
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Pisahkan data numerik untuk prediksi
X = df.drop(columns=["ticker", "target"])
X_scaled = scaler.transform(X)

# Prediksi
predictions = model.predict(X_scaled)
df["prediction"] = predictions

# Filter hanya sinyal BUY
# Contoh data (gabungan teknikal, target, bandarmology)
data = {
    "ticker": ["BBRI.JK", "BMRI.JK", "BBNI.JK", "ARTO.JK", "HOKI.JK"],
    "RSI": [28.1, 34.5, 25.9, 29.8, 19.2],
    "Stoch": [15, 45, 18, 30, 12],
    "BB_bbm": [4800, 7200, 5400, 2500, 110],
    "BB_bbh": [5000, 7500, 5800, 2700, 130],
    "BB_bbl": [4600, 6900, 5000, 2300, 90],
    "Volume_Spike": [1, 0, 1, 1, 0],
    "target": [1, 0, 1, 1, 0],
    "Bandar_Accumulation": [True, False, True, True, False],
    "Foreign_Buy_Ratio": [0.75, 0.35, 0.60, 0.80, 0.20]
}

df = pd.DataFrame(data)

# Filter target = 1 dan validasi sinyal BUY
buy_signals = df[df["target"] == 1]
buy_signals_validated = buy_signals[
    (buy_signals["RSI"] < 30) &
    (buy_signals["Stoch"] < 20) &
    (buy_signals["BB_bbm"] - buy_signals["BB_bbl"] > 0) &
    ((buy_signals["BB_bbm"] - buy_signals["BB_bbl"]) > (buy_signals["BB_bbh"] - buy_signals["BB_bbm"])) &
    (buy_signals["Bandar_Accumulation"] == True) &
    (buy_signals["Foreign_Buy_Ratio"] > 0.5)
]

# Simpan hasil prediksi
buy_signals.to_csv("buy_signals.csv", index=False)
print("✅ Sinyal BUY disimpan ke 'buy_signals.csv'")
