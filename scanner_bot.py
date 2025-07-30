# scanner.py

import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import yfinance as yf
from datetime import datetime

# Load models
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
dnn_model = load_model("deep_learning_model.h5")

# Target stock tickers to scan (could use IDX Tickers API)

tickers = [
    # Big Caps Perbankan & Keuangan
    "BBRI.JK", "BMRI.JK", "BBCA.JK", "BBNI.JK", "BRIS.JK",

    # Telekomunikasi & Infrastruktur
    "TLKM.JK", "ISAT.JK", "TOWR.JK", "TBIG.JK",

    # Konsumer & FMCG
    "MYOR.JK", "ICBP.JK", "INDF.JK", "UNVR.JK", "KLBF.JK", "CPIN.JK",

    # Energi, Mineral, Batu Bara
    "ITMG.JK", "PTBA.JK", "ADRO.JK", "ANTM.JK", "MDKA.JK", "INCO.JK", "INDY.JK", "MEDC.JK", "PGAS.JK", "ELSA.JK",

    # Industri, Komponen, Otomotif
    "ASII.JK", "UNTR.JK", "INKP.JK", "SMGR.JK",

    # Properti & Konstruksi
    "BSDE.JK", "SMRA.JK", "WSKT.JK", "WIKA.JK",

    # Teknologi & Digital Economy
    "GOTO.JK", "BUKA.JK",

    # Ritel & Konsumsi Tambahan
    "ACES.JK"
]


# Download today’s data
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(tickers, period="10d", interval="1d", group_by="ticker", threads=True)

signals = {}

for ticker in tickers:
    try:
        df = data[ticker].copy()
        df.dropna(inplace=True)
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(window=5).std()
        df.dropna(inplace=True)

        # Feature engineering
        features = df[["Close", "MA5", "MA10", "Return", "Volatility"]].tail(1).values
        features_scaled = scaler.transform(features)

        # Predict using RF and DNN
        rf_pred = rf_model.predict(features_scaled)[0]
        dnn_pred = np.argmax(dnn_model.predict(features_scaled), axis=1)[0]

        final_signal = "🔼 Buy" if rf_pred == 1 and dnn_pred == 1 else ("🔽 Sell" if rf_pred == 0 and dnn_pred == 0 else "🟡 Neutral")
        signals[ticker] = final_signal
    except Exception as e:
        signals[ticker] = f"Error: {e}"

# Display signals
print("=== Daily AI Stock Signal Recommendations ===")
for k, v in signals.items():
    print(f"{k}: {v}")

# Optional: integrate with Telegram Bot
# from telegram import Bot
# token = "YOUR_TOKEN"
# chat_id = "YOUR_CHAT_ID"
# bot = Bot(token)
# bot.send_message(chat_id=chat_id, text="\n".join([f"{k}: {v}" for k, v in signals.items()]))
