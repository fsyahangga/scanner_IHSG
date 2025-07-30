import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import telegram
import os

STOCKS = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "ARTO.JK"]
START_DATE = "2024-01-01"

def get_features(df):
    df = df.copy()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['Close']
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df.dropna(inplace=True)
    return df

def predict_signal(df):
    features = ['RSI', 'BB_Width', '%K', '%D']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['Target'].astype(int)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:-1], y[:-1])

    latest = scaler.transform([df[features].iloc[-1]])
    prediction = model.predict(latest)[0]
    return int(prediction)

def get_signal(ticker):
    df = yf.download(ticker, start=START_DATE, progress=False)
    df = get_features(df)

    latest = df.iloc[-1]
    prediction = predict_signal(df)

    rsi = round(latest['RSI'], 2)
    bbwidth = round(latest['BB_Width'], 4)
    k = round(latest['%K'], 2)
    d = round(latest['%D'], 2)

    status = "🟢 Buy" if prediction == 1 else "🔴 Sell"
    return f"{ticker}: {status} (RSI={rsi}, BB_Width={bbwidth}, %K={k}, ML={prediction})"

def send_telegram(message):
    try:
        token = os.environ['TELEGRAM_TOKEN']
        chat_id = os.environ['TELEGRAM_CHAT_ID']
        bot = telegram.Bot(token=token)
        bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def run_scanner():
    results = []
    for stock in STOCKS:
        try:
            signal = get_signal(stock)
            results.append(signal)
        except Exception as e:
            results.append(f"{stock}: Error - {str(e)}")

    output = "📈 Stock Signals " + datetime.now().strftime('%Y-%m-%d') + "\n" + "\n".join(results)
    print("Signal Result:\n", output)
    send_telegram(output)

if __name__ == "__main__":
    run_scanner()
