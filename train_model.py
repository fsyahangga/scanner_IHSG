import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import (
    load_dataset,
    preprocess_features,
    save_scalers,
    train_model_rf,
    train_model_xgb,
    train_model_lstm,
    evaluate_model,
    save_model
)

# ===============================
# CONFIG
# ===============================
DATA_PATH = "historical_idx_dataset.csv"
MODEL_DIR = "models"
FEATURES = [
    "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl", "Volume_Spike",
    "PER", "PBV", "bandarmology_score", "latest_close", "latest_volume"
]
TARGET = "target"

# ===============================
# MAIN TRAINING PIPELINE
# ===============================
def main():
    print(f"📁 Current working directory: {os.getcwd()}")

    # Load & preprocess
    df = load_dataset(DATA_PATH)
    X, y = preprocess_features(df, FEATURES, TARGET)

    # Save scalers
    save_scalers(X, MODEL_DIR)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores, xgb_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold+1}/5")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train RF & XGB
        rf_model = train_model_rf(X_train, y_train)
        rf_score = evaluate_model(rf_model, X_val, y_val)
        rf_scores.append(rf_score)

        xgb_model = train_model_xgb(X_train, y_train)
        xgb_score = evaluate_model(xgb_model, X_val, y_val)
        xgb_scores.append(xgb_score)

    print("\n📊 Cross-Validation Summary:")
    print(f"🌲 Random Forest AUC: Mean={np.mean(rf_scores):.4f} | Std={np.std(rf_scores):.4f}")
    print(f"⚡ XGBoost AUC       : Mean={np.mean(xgb_scores):.4f} | Std={np.std(xgb_scores):.4f}")

    # Final training on full data
    rf_final = train_model_rf(X, y)
    save_model(rf_final, "rf_model.pkl", MODEL_DIR)

    xgb_final = train_model_xgb(X, y)
    save_model(xgb_final, "xgb_model.pkl", MODEL_DIR)

    lstm_final = train_model_lstm(X, y)
    save_model(lstm_final, "lstm_model.h5", MODEL_DIR)

    print("\n✅ All models trained and saved successfully!")


if __name__ == "__main__":
    main()
