import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import os

def load_latest_data(path='latest_realtime_data.csv') -> pd.DataFrame:
    """
    Memuat dan menghitung ulang indikator teknikal dari file latest_realtime_data.csv
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} tidak ditemukan.")

    df = pd.read_csv(path)

    # Pastikan kolom wajib ada untuk proses indikator
    expected_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib '{col}' tidak ada dalam {path}")

    # Hitung ulang indikator teknikal
    df = calculate_indicators(df)

    # Tambahkan fitur tambahan
    df['candlestick_pattern'] = detect_candlestick_pattern(df)
    df['macro_sentiment'] = get_macro_sentiment()

    # Loop untuk foreign flow & bandarmology
    df['Foreign_Buy_Ratio'] = df['ticker'].apply(lambda x: get_foreign_flow_data(f"{x}.JK")[1])
    df['bandarmology_score'] = df['ticker'].apply(lambda x: calculate_bandarmology_score(x))

    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung indikator teknikal menggunakan pustaka `ta` berdasarkan dataframe OHLCV.
    Input: df - DataFrame dengan kolom ['ticker','date','open','high','low','close','volume']
    Output: df - DataFrame dengan kolom tambahan indikator teknikal.
    """
    df = df.copy()
    
    # Format dan sort data
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['ticker', 'date'], inplace=True)
    df['ticker'] = df['ticker'].str.replace('.JK', '', regex=False)

    # Pastikan kolom yang diperlukan ada
    required_columns = ['ticker', 'RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in latest_realtime_data.csv: {missing_cols}")

    # Tambahkan kolom dummy jika tidak ada, untuk kesesuaian dengan training set
    for col in ['PER', 'PBV', 'bandarmology_score', 'Foreign_Buy_Ratio', 'macro_sentiment', 'candlestick_pattern', 'target']:
        if col not in df.columns:
            df[col] = 0

    result = []
    for ticker, group in df.groupby('ticker'):
        group = group.copy()

        # RSI
        group['RSI'] = RSIIndicator(close=group['close'], window=14).rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(high=group['high'], low=group['low'], close=group['close'], window=14, smooth_window=3)
        group['Stoch'] = stoch.stoch()

        # Bollinger Bands
        bb = BollingerBands(close=group['close'], window=20, window_dev=2)
        group['BB_bbh'] = bb.bollinger_hband()
        group['BB_bbm'] = bb.bollinger_mavg()
        group['BB_bbl'] = bb.bollinger_lband()

        # Volume Spike (boolean: volume > 1.5x MA20)
        group['vol_ma20'] = group['volume'].rolling(window=20).mean()
        group['Volume_Spike'] = (group['volume'] > 1.5 * group['vol_ma20']).astype(int)

        # Redundansi kolom
        group['latest_close'] = group['close']
        group['latest_volume'] = group['volume']

        result.append(group)

    df_final = pd.concat(result)
    df_final.drop(columns=['vol_ma20'], inplace=True)

    return df_final



# ---------------------------
# Fundamental Placeholder
# ---------------------------

def get_fundamental_metrics(ticker):
    # Simulasi ambil data PER, PBV
    return {
        "PER": np.random.uniform(5, 20),
        "PBV": np.random.uniform(0.5, 3),
    }

# ---------------------------
# Bandarmology Placeholder
# ---------------------------

def calculate_bandarmology_score(ticker):
    return round(np.random.uniform(0, 1), 2)

# ---------------------------
# Macro Sentiment (Static/Mocked)
# ---------------------------

def calculate_macro_sentiment(bi_rate, inflation, usd_idr, pmi, cpi):
    score = 0

    if bi_rate <= 6:
        score += 0.2
    else:
        score -= 0.2

    if inflation < 3.5:
        score += 0.2
    else:
        score -= 0.2

    if usd_idr < 15500:
        score += 0.2
    else:
        score -= 0.2

    if pmi >= 50:
        score += 0.2
    else:
        score -= 0.2

    if cpi <= 3:
        score += 0.2
    else:
        score -= 0.2

    return round(score, 2)

def calculate_foreign_flow(buy_foreign, sell_foreign):
    try:
        net = buy_foreign - sell_foreign
        ratio = buy_foreign / (buy_foreign + sell_foreign + 1e-9)
        return round(net), round(ratio, 2)
    except:
        return 0, 0.0

def get_macro_sentiment():
    # Dummy static values for now (replace later with API/scraper)
    return calculate_macro_sentiment(
        bi_rate=6.25,
        inflation=2.85,
        usd_idr=15480,
        pmi=51.2,
        cpi=2.9
    )

def get_foreign_flow_data(ticker):
    # Placeholder: ganti dengan hasil scraping dari RTI/EDB
    sample_data = {
        "BBRI.JK": (1_200_000_000, 600_000_000),
        "BBCA.JK": (800_000_000, 700_000_000),
        # Tambahkan ticker lainnya sesuai kebutuhan
    }
    return calculate_foreign_flow(*sample_data.get(ticker, (0, 0)))

# ---------------------------
# Candlestick Pattern Detection (Simple)
# ---------------------------

def detect_candlestick_pattern(df):
    patterns = []
    for _, row in df.iterrows():
        if row['RSI'] < 30 and row['Stoch'] < 20:
            patterns.append("Hammer")
        elif row['RSI'] > 70 and row['Stoch'] > 80:
            patterns.append("Shooting Star")
        else:
            patterns.append("NoPattern")
    return patterns

# ---------------------------
# Preprocessing Utilities
# ---------------------------

def scale_features(X, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# ---------------------------
# Save/Load Utility
# ---------------------------

def save_model_and_scaler(model, scaler, name_prefix):
    joblib.dump(model, f"models/{name_prefix}_model.pkl")
    joblib.dump(scaler, f"models/{name_prefix}_scaler.pkl")

def load_model_and_scaler(name_prefix):
    model = joblib.load(f"models/{name_prefix}_model.pkl")
    scaler = joblib.load(f"models/{name_prefix}_scaler.pkl")
    return model, scaler
