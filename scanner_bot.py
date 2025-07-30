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
buy_signals = df[df["prediction"] == 1]

# Simpan hasil prediksi
buy_signals.to_csv("buy_signals.csv", index=False)
print("✅ Sinyal BUY disimpan ke 'buy_signals.csv'")
