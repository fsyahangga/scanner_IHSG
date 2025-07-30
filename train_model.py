import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

TICKERS = ["BBRI.JK", "BBNI.JK", "BMRI.JK", "BBCA.JK", "ARTO.JK"]
MODEL_DIR = "models"

def download_data(ticker):
    df = yf.download(ticker, period="1y")
    df.dropna(inplace=True)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

def build_rf_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def build_dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_save_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for ticker in TICKERS:
        print(f"Training model for {ticker}")
        df = download_data(ticker)
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Return']]
        y = df['Target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        rf = build_rf_model(X_train, y_train)
        joblib.dump(rf, f"{MODEL_DIR}/rf_{ticker}.pkl")
        joblib.dump(scaler, f"{MODEL_DIR}/scaler_{ticker}.pkl")

        dnn = build_dnn_model(X_train.shape[1])
        dnn.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
        dnn.save(f"{MODEL_DIR}/dnn_{ticker}.h5")

        print(f"Saved models for {ticker}")

if __name__ == "__main__":
    train_and_save_models()
