import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from utils import load_latest_data, calculate_indicators

DATA_PATH = 'historical_idx_dataset.csv'
MODEL_DIR = 'models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODEL_DIR, 'random_forest.pkl'))
    return model

def train_xgboost(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgboost.pkl'))
    return model

def train_lstm(X, y):
    X_reshaped = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, X.shape[1]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_reshaped, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=0)
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    return model

def preprocess_data(df):
    # Pilih fitur dan target
    features = [
        'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl',
        'Volume_Spike', 'MACD', 'MACD_signal', 'EMA_12', 'EMA_26', 'ADX',
        'Foreign_Buy_Ratio', 'Bandar_Activity', 'Price_Change',
        'Stochastic_Signal', 'Candle_Pattern_Bullish', 'Candle_Pattern_Bearish',
        'candlestick_pattern'
    ]

    df = df.dropna(subset=features + ['target'])  # hindari NaN di fitur atau target

    if df.empty:
        raise ValueError("Data kosong setelah preprocessing. Cek apakah input dataset memiliki cukup data yang valid.")

    X = df[features]
    y = df['target']

    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()

    X_scaled_std = scaler_std.fit_transform(X)
    X_scaled_minmax = scaler_minmax.fit_transform(X)

    # Simpan scaler untuk penggunaan inference
    joblib.dump(scaler_std, "models/scaler_std.pkl")
    joblib.dump(scaler_minmax, "models/scaler_minmax.pkl")

    return X_scaled_std, y


def evaluate_model(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        scores.append(score)

    print(f'Model AUC: {np.mean(scores):.4f}')

def main():
    df_latest = load_latest_data()
    df = calculate_indicators(df_latest)

    X, y = preprocess_data(df)

    rf_model = train_random_forest(X, y)
    xgb_model = train_xgboost(X, y)
    lstm_model = train_lstm(X, y)

    evaluate_model(rf_model, X, y)
    evaluate_model(xgb_model, X, y)

if __name__ == '__main__':
    main()
