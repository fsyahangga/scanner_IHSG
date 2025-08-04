import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import calculate_indicators, scale_features, save_model_and_scaler


def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded: {file_path}")
        return df
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return pd.DataFrame()


def ensure_columns(df: pd.DataFrame, required_cols: list, default_value=np.nan) -> pd.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Kolom {col} tidak ditemukan, menambahkan default.")
            df[col] = default_value
    return df


def preprocess_latest_df(latest_df: pd.DataFrame) -> pd.DataFrame:
    latest_df = calculate_indicators(latest_df)
    latest_df.dropna(inplace=True)

    expected_cols = [
        'ticker', 'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl',
        'volume', 'PER', 'PBV', 'bandarmology_score', 'close', 'target'
    ]
    latest_df = latest_df[expected_cols]

    latest_df.columns = [
        'ticker', 'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl',
        'latest_volume', 'PER', 'PBV', 'bandarmology_score', 'latest_close', 'target'
    ]
    return latest_df


def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # Hitung Volume Spike jika belum ada
    if 'Volume_Spike' not in df.columns:
        df['Volume_Spike'] = (df['latest_volume'] > df['latest_volume'].rolling(5).mean()).astype(int)

    feature_cols = [
        'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike',
        'PER', 'PBV', 'bandarmology_score', 'latest_close', 'latest_volume'
    ]

    df = df.dropna(subset=feature_cols + ['target'])
    df = df[df['target'].notna() & np.isfinite(df['target'])]

    X = df[feature_cols]
    y = df['target'].astype(int)
    return X, y


def train_evaluate_model(X_scaled, y) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []
    for train_idx, val_idx in kfold.split(X_scaled, y):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc)

    print(f"✅ AUC scores (5-fold): {auc_scores}")
    print(f"📊 Mean AUC: {np.mean(auc_scores):.4f}")
    return model


def main():
    historical_df = load_dataset("historical_idx_dataset.csv")
    latest_df = load_dataset("latest_realtime_data.csv")

    # Pastikan kolom wajib tersedia di real-time data
    required_cols = [
        'PER', 'PBV', 'bandarmology_score',
        'Foreign_Buy_Ratio', 'macro_sentiment', 'candlestick_pattern', 'target'
    ]
    latest_df = ensure_columns(latest_df, required_cols)

    latest_df = preprocess_latest_df(latest_df)

    # Gabungkan data historis dan real-time
    combined_df = pd.concat([historical_df, latest_df], ignore_index=True)

    # Siapkan fitur & target
    X, y = prepare_features(combined_df)

    # Scaling
    X_scaled, scaler = scale_features(X, method='standard')

    # Training + Evaluasi
    model = train_evaluate_model(X_scaled, y)

    # Save model dan scaler
    save_model_and_scaler(model, scaler, name_prefix="rf")
    print("💾 Model & scaler saved to: models/rf_model.pkl and rf_scaler.pkl")


if __name__ == "__main__":
    main()
