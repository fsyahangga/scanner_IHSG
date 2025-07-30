import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
import os
import requests

STOCKS = ['BBRI.JK', 'BBCA.JK', 'BMRI.JK', 'BBNI.JK', 'ARTO.JK']

def get_signal(stock):
    df = yf.download(stock, period="3mo", interval="1d")
    if df.empty or len(df) < 15:
        return f"{stock}: Data tidak cukup"

    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()  # Convert to Series if needed

    rsi = RSIIndicator(close=close_series, window=14).rsi()
    df['RSI'] = rsi

    last_rsi = df['RSI'].iloc[-1]
    if last_rsi < 30:
        return f"{stock}: 📉 OVERSOLD (RSI={last_rsi:.2f})"
    elif last_rsi > 70:
        return f"{stock}: 📈 OVERBOUGHT (RSI={last_rsi:.2f})"
    else:
        return f"{stock}: 🟡 Neutral (RSI={last_rsi:.2f})"

def send_telegram_message(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials not set.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print("Failed to send Telegram message:", response.text)

def run_scanner():
    signals = [get_signal(stock) for stock in STOCKS]
    result = "\n".join(signals)
    print("Signal Result:\n", result)
    send_telegram_message(f"*Daily RSI Signal 📊*\n\n{result}")

if __name__ == "__main__":
    run_scanner()
