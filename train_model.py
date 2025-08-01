import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =======================
# Configuration
# =======================
DATA_PATH = "historical_idx_dataset.csv"
FEATURES = [
    "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl", "Volume_Spike",
    "PER", "PBV", "bandarmology_score", "latest_close", "latest_volume"
]
TARGET = "target"
MODEL_DIR = "models"


# =======================
# Utility Functions
# =======================

def load_data(path):
    df = pd.read_csv(path)
    print(f"📥 Loaded dataset with shape: {df.shape}")
    return df


def preprocess_data(df, features, target):
    X = df[features].replace([np.inf, -np.inf], np.nan)
    
    print("🔍 Missing values before cleaning:")
    print(X.isna().sum())

    X = X.fillna(X.median())
    y = df[target]

    X = X.reset_index(drop=True)
    y = y.loc[X.index].reset_index(drop=True)

    if X.shape[0] == 0:
        raise ValueError("❌ Semua baris terhapus setelah cleaning! Dataset kosong.")

    return X, y


def train_and_save_scalers(X):
    os.makedirs(MODEL_DIR, exist_ok=True)

    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    joblib.dump(std_scaler.fit(X), os.path.join(MODEL_DIR, "standard_scaler.pkl"))
    print("✅ StandardScaler trained and saved")

    joblib.dump(minmax_scaler.fit(X), os.path.join(MODEL_DIR, "minmax_scaler.pkl"))
    print("✅ MinMaxScaler trained and saved")


def train_and_save_models(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_auc_scores = []
    xgb_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔄 Fold {fold + 1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict_proba(X_val)[:, 1]
        rf_auc = roc_auc_score(y_val, rf_pred)
        rf_auc_scores.append(rf_auc)
        print(f"🌲 RandomForest AUC: {rf_auc:.4f}")

        # XGBoost
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        xgb_auc_scores.append(xgb_auc)
        print(f"⚡ XGBoost AUC: {xgb_auc:.4f}")

    # Train final models on all data
    rf_final = RandomForestClassifier(random_state=42)
    rf_final.fit(X, y)
    joblib.dump(rf_final, os.path.join(MODEL_DIR, "rf_model.pkl"))
    print("✅ RandomForest model saved")

    xgb_final = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_final.fit(X, y)
    joblib.dump(xgb_final, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    print("✅ XGBoost model saved")

    # Train LSTM model
    X_lstm = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_lstm, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)

    model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    print("✅ LSTM model saved")


# =======================
# Main Pipeline
# =======================

if __name__ == "__main__":
    print(f"📁 Current working directory: {os.getcwd()}")
    
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df, FEATURES, TARGET)
    train_and_save_scalers(X)
    train_and_save_models(X, y)

    print("\n🎉 All models trained and saved successfully!")
