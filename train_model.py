import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import logging
from utils import ensure_dir_exists

# ===========================
# Configuration
# ===========================
DATA_PATH = "historical_idx_dataset.csv"
FEATURES = [
    "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl", "Volume_Spike",
    "PER", "PBV", "bandarmology_score", "latest_close", "latest_volume"
]
TARGET = "target"
MODEL_DIR = "models"

# ===========================
# Logging Setup
# ===========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================
# Load and Preprocess
# ===========================
def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"📥 Loaded dataset: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"❌ Failed to load dataset: {e}")
        raise


def preprocess_data(df, features, target):
    try:
        X = df[features].replace([np.inf, -np.inf], np.nan)
        logging.info("🔍 Missing values before fill:")
        logging.info(X.isna().sum())

        X.fillna(X.median(), inplace=True)
        y = df[target].copy()

        X = X.reset_index(drop=True)
        y = y.loc[X.index].reset_index(drop=True)

        if X.empty:
            raise ValueError("❌ All rows removed during cleaning. Empty dataset.")
        return X, y
    except Exception as e:
        logging.error(f"❌ Error during preprocessing: {e}")
        raise


# ===========================
# Scaler Functions
# ===========================
def train_and_save_scalers(X, model_dir):
    try:
        ensure_dir_exists(model_dir)
        std_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        joblib.dump(std_scaler.fit(X), os.path.join(model_dir, "standard_scaler.pkl"))
        joblib.dump(minmax_scaler.fit(X), os.path.join(model_dir, "minmax_scaler.pkl"))
        logging.info("✅ Scalers trained and saved.")
    except Exception as e:
        logging.error(f"❌ Failed to train scalers: {e}")
        raise


# ===========================
# Model Training Functions
# ===========================
def train_random_forest(X, y, model_dir):
    auc_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logging.info(f"🔁 RF Fold {fold+1}")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict_proba(X.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y.iloc[val_idx], pred)
        auc_scores.append(auc)
        logging.info(f"🌲 RF AUC: {auc:.4f}")

    rf_final = RandomForestClassifier(random_state=42)
    rf_final.fit(X, y)
    joblib.dump(rf_final, os.path.join(model_dir, "rf_model.pkl"))
    logging.info("✅ RF final model saved.")
    return np.mean(auc_scores)


def train_xgboost(X, y, model_dir):
    auc_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logging.info(f"🔁 XGB Fold {fold+1}")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = xgb.predict_proba(X.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y.iloc[val_idx], pred)
        auc_scores.append(auc)
        logging.info(f"⚡ XGB AUC: {auc:.4f}")

    xgb_final = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_final.fit(X, y)
    joblib.dump(xgb_final, os.path.join(model_dir, "xgb_model.pkl"))
    logging.info("✅ XGB final model saved.")
    return np.mean(auc_scores)


def train_lstm(X, y, model_dir):
    try:
        X_lstm = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
        model = Sequential([
            LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=False),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC"])

        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(X_lstm, y, epochs=50, batch_size=16, validation_split=0.2,
                  callbacks=[early_stop], verbose=1)

        model.save(os.path.join(model_dir, "lstm_model.h5"))
        logging.info("✅ LSTM model saved.")
    except Exception as e:
        logging.error(f"❌ Error training LSTM: {e}")
        raise


# ===========================
# Main Pipeline
# ===========================
def main():
    logging.info("🚀 Training pipeline started...")
    ensure_dir_exists(MODEL_DIR)

    df = load_data(DATA_PATH)
    X, y = preprocess_data(df, FEATURES, TARGET)

    train_and_save_scalers(X, MODEL_DIR)
    rf_auc = train_random_forest(X, y, MODEL_DIR)
    xgb_auc = train_xgboost(X, y, MODEL_DIR)
    train_lstm(X, y, MODEL_DIR)

    logging.info(f"\n🎯 Final AUC Scores:\n - RandomForest: {rf_auc:.4f}\n - XGBoost: {xgb_auc:.4f}")
    logging.info("🎉 All models trained and saved successfully!")


if __name__ == "__main__":
    main()
