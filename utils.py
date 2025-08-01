import pandas as pd
import numpy as np
import joblib
import os
import logging

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def log(message):
    logging.info(message)

def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        log(f"Gagal load model: {path} | Error: {e}")
        return None

def load_scaler(path):
    try:
        return joblib.load(path)
    except Exception as e:
        log(f"Gagal load scaler: {path} | Error: {e}")
        return None

def save_predictions(df, filename="buy_signals.csv"):
    try:
        df.to_csv(filename, index=False)
        log(f"📥 Data hasil prediksi disimpan ke {filename}")
    except Exception as e:
        log(f"❌ Gagal simpan hasil prediksi: {e}")

def scale_data(X, scaling_type="standard"):
    scaler_path = "models/standard_scaler.pkl" if scaling_type == "standard" else "models/minmax_scaler.pkl"
    scaler = load_scaler(scaler_path)
    return scaler.transform(X) if scaler else X

def extract_and_scale_features(df, feature_columns, scaling_type="standard"):
    try:
        X = df[feature_columns].copy()
        X_scaled = scale_data(X, scaling_type)
        return X_scaled
    except Exception as e:
        log(f"❌ Gagal ekstrak dan scaling fitur: {e}")
        return None

def predict_with_models(X, models):
    results = {}
    for name, model in models.items():
        try:
            prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
            results[name] = prob
        except Exception as e:
            log(f"❌ Gagal prediksi dengan model {name}: {e}")
            results[name] = np.zeros(len(X))
    return results

def majority_vote(predictions_dict, threshold=0.5):
    votes = pd.DataFrame(predictions_dict)
    votes["average"] = votes.mean(axis=1)
    conditions = [
        (votes["average"] >= threshold + 0.1),
        (votes["average"] <= threshold - 0.1)
    ]
    choices = ["BUY", "SELL"]
    votes["recommendation"] = np.select(conditions, choices, default="NEUTRAL")
    return votes["recommendation"]

def evaluate_model(model, X_val, y_val):
    try:
        preds = model.predict(X_val)
        report = classification_report(y_val, preds)
        matrix = confusion_matrix(y_val, preds)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]) if hasattr(model, "predict_proba") else "N/A"
        log(f"Classification Report:\n{report}")
        log(f"Confusion Matrix:\n{matrix}")
        log(f"AUC: {auc}")
    except Exception as e:
        log(f"❌ Gagal evaluasi model: {e}")

def load_data(path="historical_idx_dataset.csv"):
    try:
        return pd.read_csv(path)
    except Exception as e:
        log(f"❌ Gagal load data: {e}")
        return pd.DataFrame()

def clean_data(df):
    return df.dropna().copy() if not df.empty else df

def prepare_input_features(df, feature_columns, scaling_type="standard"):
    df = clean_data(df)
    return extract_and_scale_features(df, feature_columns, scaling_type) if not df.empty else None

def get_target_labels(df, target_column="target"):
    return df[target_column].values if target_column in df.columns else None

def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        log(f"✅ Model disimpan: {filename}")
    except Exception as e:
        log(f"❌ Gagal simpan model: {e}")
        
def detect_candlestick_pattern(df):
    # Contoh sederhana (bisa diganti dengan TA-Lib)
    patterns = []
    for _, row in df.iterrows():
        if row['RSI'] < 30 and row['Stoch'] < 20:
            patterns.append("Hammer")
        else:
            patterns.append("NoPattern")
    return patterns

def get_macro_sentiment():
    # Placeholder: bisa dari API atau input manual/data historis
    return "POSITIVE"  # atau bisa random / dari CSV makro
