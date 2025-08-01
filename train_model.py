import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import calculate_indicators, scale_features, save_model_and_scaler

# ------------------------------
# Load Datasets
# ------------------------------
historical_df = pd.read_csv("historical_idx_dataset.csv")
latest_df = pd.read_csv("latest_realtime_data.csv")

# ------------------------------
# Preprocess latest real-time data
# ------------------------------
latest_df = calculate_indicators(latest_df)
latest_df.dropna(inplace=True)

# Pastikan struktur kolom latest_df sesuai historical
latest_df = latest_df[[
    'ticker', 'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl',
    'volume', 'PER', 'PBV', 'bandarmology_score', 'close', 'target'
]]
latest_df.columns = [
    'ticker', 'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl',
    'latest_volume', 'PER', 'PBV', 'bandarmology_score', 'latest_close', 'target'
]

# Gabungkan dua dataset
combined_df = pd.concat([historical_df, latest_df], ignore_index=True)

# ------------------------------
# Feature & Target Selection
# ------------------------------
features = [
    'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike',
    'PER', 'PBV', 'bandarmology_score', 'latest_close', 'latest_volume'
]

# Jika 'Volume_Spike' belum dihitung di latest_df, hitung berdasarkan threshold
if 'Volume_Spike' not in combined_df.columns:
    combined_df['Volume_Spike'] = (combined_df['latest_volume'] > combined_df['latest_volume'].rolling(5).mean()).astype(int)

X = combined_df[features].copy()
y = combined_df['target'].astype(int)

# ------------------------------
# Scaling
# ------------------------------
X_scaled, scaler = scale_features(X, method='standard')

# ------------------------------
# Model Training with Stratified K-Fold
# ------------------------------
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

# ------------------------------
# Save Model and Scaler
# ------------------------------
save_model_and_scaler(model, scaler, name_prefix="rf")
print("💾 Model & scaler saved to: models/rf_model.pkl and rf_scaler.pkl")
