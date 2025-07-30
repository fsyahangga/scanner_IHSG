import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from train_model import retrain_models  # Import retraining function

# Load pre-trained models
rf_model = joblib.load("random_forest_model.pkl")
dnn_model = load_model("deep_learning_model.h5")
scaler = joblib.load("scaler.pkl")

# Example IDX tickers
IDX_TICKERS = [
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


# Filtering dictionary
IDX_FILTER = {
    "perbankan_keuangan": {
        "tickers": ["BBRI.JK", "BMRI.JK", "BBCA.JK", "BBNI.JK", "BRIS.JK"],
        "valuasi": "murah",
        "ai_score": 8.5,
        "teknikal": {
            "RSI_below_30": ["BBRI.JK"],
            "MACD_bullish": ["BMRI.JK", "BRIS.JK"],
            "Volume_spike": ["BBCA.JK"]
        }
    },
    "teknologi_digital": {
        "tickers": ["ARTO.JK", "GOTO.JK", "BUKA.JK"],
        "valuasi": "tinggi",
        "ai_score": 6.0,
        "teknikal": {
            "RSI_below_30": ["GOTO.JK"],
            "MACD_bullish": ["ARTO.JK"],
            "Volume_spike": ["BUKA.JK"]
        }
    },
    "barang_konsumsi": {
        "tickers": ["KLBF.JK", "UNVR.JK", "CPIN.JK", "ACES.JK"],
        "valuasi": "sedang",
        "ai_score": 7.5,
        "teknikal": {
            "RSI_below_30": [],
            "MACD_bullish": ["KLBF.JK", "CPIN.JK"],
            "Volume_spike": ["UNVR.JK"]
        }
    },
    "energi_batubara_migas": {
        "tickers": ["ADRO.JK", "ITMG.JK", "PTBA.JK", "INDY.JK", "MEDC.JK", "PGAS.JK", "ELSA.JK"],
        "valuasi": "murah",
        "ai_score": 8.0,
        "teknikal": {
            "RSI_below_30": ["PGAS.JK", "PTBA.JK"],
            "MACD_bullish": ["ADRO.JK", "MEDC.JK"],
            "Volume_spike": ["ITMG.JK"]
        }
    },
    "logam_tambang": {
        "tickers": ["ANTM.JK", "INCO.JK", "MDKA.JK"],
        "valuasi": "sedang",
        "ai_score": 8.2,
        "teknikal": {
            "RSI_below_30": ["MDKA.JK"],
            "MACD_bullish": ["INCO.JK"],
            "Volume_spike": ["ANTM.JK"]
        }
    },
    "konstruksi_infrastruktur": {
        "tickers": ["WIKA.JK", "WSKT.JK", "SMGR.JK", "TOWR.JK", "TBIG.JK"],
        "valuasi": "murah",
        "ai_score": 7.8,
        "teknikal": {
            "RSI_below_30": ["WIKA.JK", "WSKT.JK"],
            "MACD_bullish": ["SMGR.JK"],
            "Volume_spike": ["TOWR.JK"]
        }
    },
    "konglomerasi_otomotif": {
        "tickers": ["ASII.JK", "UNTR.JK"],
        "valuasi": "sedang",
        "ai_score": 8.3,
        "teknikal": {
            "RSI_below_30": [],
            "MACD_bullish": ["ASII.JK"],
            "Volume_spike": ["UNTR.JK"]
        }
    },
    "telekomunikasi": {
        "tickers": ["TLKM.JK"],
        "valuasi": "sedang",
        "ai_score": 8.4,
        "teknikal": {
            "RSI_below_30": [],
            "MACD_bullish": ["TLKM.JK"],
            "Volume_spike": []
        }
    },
    "pulp_kertas": {
        "tickers": ["INKP.JK"],
        "valuasi": "murah",
        "ai_score": 7.2,
        "teknikal": {
            "RSI_below_30": ["INKP.JK"],
            "MACD_bullish": [],
            "Volume_spike": []
        }
    }
}


def get_technical_indicators(df):
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['Stoch'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    bb = BollingerBands(close=df['Close'])
    df['BB_bbm'] = bb.bollinger_mavg()
    df['BB_bbh'] = bb.bollinger_hband()
    df['BB_bbl'] = bb.bollinger_lband()
    df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    return df.dropna()

def extract_features(df):
    return df[['RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike']].values[-1].reshape(1, -1)

def get_prediction_signals(stock):
    df = yf.download(stock, period="6mo", interval="1d")
    df = get_technical_indicators(df)
    X = scaler.transform(extract_features(df))
    rf_pred = rf_model.predict(X)[0]
    dnn_pred = np.argmax(dnn_model.predict(X), axis=1)[0]

    consensus = "🔴 Sell"
    if rf_pred == 1 and dnn_pred == 1:
        consensus = "🟢 Buy"
    elif rf_pred == 1 or dnn_pred == 1:
        consensus = "🟡 Neutral"

    return f"{stock}: {consensus} (RF={rf_pred}, DNN={dnn_pred})"

def filter_by_sector(sector_name):
    return IDX_FILTER.get(sector_name, {}).get("tickers", [])

def filter_by_valuasi(level="murah"):
    return [ticker for info in IDX_FILTER.values() if info["valuasi"] == level for ticker in info["tickers"]]

def filter_by_ai_score(min_score=8.0):
    return [ticker for info in IDX_FILTER.values() if info["ai_score"] >= min_score for ticker in info["tickers"]]

def run_scanner():
    print("📊 Daily Stock Recommendation (AI-Powered):")
    all_tickers = list(set(IDX_TICKERS + filter_by_ai_score(8.0)))
    for stock in all_tickers:
        try:
            signal = get_prediction_signals(stock)
            print(signal)
        except Exception as e:
            print(f"{stock}: Error - {e}")

def run_training():
    print("🔄 Retraining models with fresh data...")
    retrain_models()
    print("✅ Retraining complete.")

if __name__ == "__main__":
    run_training()
    run_scanner()
