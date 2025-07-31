import pandas as pd
import numpy as np
import joblib
import ta
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Load dataset
data = pd.read_csv("historical_idx_dataset.csv")

# Fitur dan target
FEATURES = [
    "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl", "Volume_Spike",
    "PER", "PBV", "bandarmology_score",
    "latest_close", "latest_volume"
]
TARGET = "target"

X = data[FEATURES].copy()
y = data[TARGET]
os.makedirs("models", exist_ok=True)
# ------------------- StandardScaler  -------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# joblib.dump(scaler, "models/feature_scaler.pkl")
print("✅ StandardScaler trained and saved")

# ------------------- MinMaxScaler  -------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Simpan scaler untuk inference
# joblib.dump(scaler, "models/feature_scaler.pkl")
print("✅ MinMaxScaler trained and saved")

print("📁 Current working directory:", os.getcwd())
joblib.dump(StandardScaler().fit(X), "models/standard_scaler.pkl")
joblib.dump(MinMaxScaler().fit(X), "models/minmax_scaler.pkl")

# Stratified K-Fold CV (untuk klasifikasi imbang/tidak imbang)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === RANDOM FOREST ===
best_rf_auc = -np.inf
best_rf_model = None
rf_model_paths = []

print("\n🎯 Training Random Forest with CV")
for i, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=i)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)

    model_path = f"models/random_forest_fold{i+1}.pkl"
    print("✅ Selesai melatih model. Model terbaik:", model)
    print("📊 Skor terbaik:", auc)
    print("🔍 Model best_rf_model:", best_rf_model)

    joblib.dump(model, model_path)
    rf_model_paths.append(model_path)

    print(f"🟢 RF Fold {i+1} AUC: {auc:.4f}")
    if auc > best_rf_auc:
        best_rf_auc = auc
        best_rf_model = model

if best_rf_model is not None:
    joblib.dump(best_rf_model, "models/random_forest_model.pkl")
    print("✅ File random_forest_model.pkl berhasil disimpan")
else:
    print("❌ Model Random Forest tidak terbentuk!")

if os.path.exists("models/random_forest_model.pkl"):
    print("✅ File random_forest_model.pkl berhasil disimpan")
else:
    print("❌ Gagal menyimpan random_forest_model.pkl")


print(f"✅ RF Best AUC: {best_rf_auc:.4f}")

# === XGBOOST ===
best_xgb_auc = -np.inf
best_xgb_model = None
xgb_model_paths = []

print("\n🎯 Training XGBoost with CV")
for i, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=i)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)

    model_path = f"models/xgboost_fold{i+1}.pkl"
    joblib.dump(model, model_path)
    xgb_model_paths.append(model_path)

    print(f"🟠 XGB Fold {i+1} AUC: {auc:.4f}")
    if auc > best_xgb_auc:
        best_xgb_auc = auc
        best_xgb_model = model

joblib.dump(best_xgb_model, "models/xgboost_model.pkl")
print(f"✅ XGB Best AUC: {best_xgb_auc:.4f}")

# === LSTM ===
print("\n🎯 Training LSTM")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

lstm_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1]), activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))
lstm_model.save("models/lstm_model.h5")
print("✅ LSTM saved.")

# === Simpan path semua model fold ===
with open("models/rf_model_folds.txt", "w") as f:
    f.write("\n".join(rf_model_paths))

with open("models/xgb_model_folds.txt", "w") as f:
    f.write("\n".join(xgb_model_paths))

print("\n📊 Evaluasi Akhir:")
y_pred = best_rf_model.predict(X_scaled)
print("Random Forest Classification Report:")
print(classification_report(y, y_pred))

print("✅ Model retrained and saved successfully.")
print("📁 Cek isi folder models setelah training:", os.listdir("models"))
