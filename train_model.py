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

from utils import calculate_indicators, load_latest_data

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
    df = df.dropna()
    X = df.drop(columns=['ticker', 'date', 'target'])
    y = df['target']

    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()
    
    X_scaled_std = scaler_std.fit_transform(X)
    X_scaled_minmax = scaler_minmax.fit_transform(X)

    joblib.dump(scaler_std, os.path.join(MODEL_DIR, 'scaler_std.pkl'))
    joblib.dump(scaler_minmax, os.path.join(MODEL_DIR, 'scaler_minmax.pkl'))

    return pd.DataFrame(X_scaled_std, columns=X.columns), y

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
    df = pd.read_csv(DATA_PATH)
    df = calculate_indicators(df)

    X, y = preprocess_data(df)

    rf_model = train_random_forest(X, y)
    xgb_model = train_xgboost(X, y)
    lstm_model = train_lstm(X, y)

    evaluate_model(rf_model, X, y)
    evaluate_model(xgb_model, X, y)

if __name__ == '__main__':
    main()
