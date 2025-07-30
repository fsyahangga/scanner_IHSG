import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
import joblib
from train_model import retrain_models

# Load dotenv config
from dotenv import load_dotenv
load_dotenv()

# ✅ Param: filter ticker tertentu (default: None)
FILTER_TICKER = os.getenv("FILTER_TICKER")  # Contoh: 'BBRI.JK,BBCA.JK'
if FILTER_TICKER:
    FILTER_TICKER = [x.strip() for x in FILTER_TICKER.split(',')]

# ✅ Path Dataset
DATASET_PATH = "historical_idx_dataset.csv"

# ✅ Retrain models (terintegrasi)
retrain_models()

# ✅ Load dataset
df = pd.read_csv(DATASET_PATH)

# ✅ Filter ticker (jika ada)
if FILTER_TICKER:
    df = df[df['ticker'].isin(FILTER_TICKER)]

# ✅ Simpan ticker dan target lalu hapus
tickers = df['ticker'].values
X = df.drop(['ticker', 'target'], axis=1)
y = df['target'].values

# ✅ Load model & scaler
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
nn_model = load_model("neural_network_model.h5")

# ✅ Preprocessing
X_scaled = scaler.transform(X)

# ✅ Predict dengan dua model
rf_preds = rf_model.predict(X_scaled)
nn_preds = (nn_model.predict(X_scaled) > 0.5).astype(int).flatten()

# ✅ Filter sinyal BUY yang disetujui dua model
buy_signals = df[(rf_preds == 1) & (nn_preds == 1)]
buy_signals = buy_signals.copy()
buy_signals['ticker'] = tickers[(rf_preds == 1) & (nn_preds == 1)]

# ✅ Simpan sinyal BUY
BUY_FILE = "buy_signals.csv"
buy_signals.to_csv(BUY_FILE, index=False)
print(f"✅ Sinyal BUY disimpan ke '{BUY_FILE}'")

# 🧹 Optional: Bersihkan memori
del df, X, y, rf_preds, nn_preds
