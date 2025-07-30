# scanner_bot.py
import yfinance as yf
import pandas as pd
import datetime
import requests
import ta

# ============ CONFIG ==============
STOCKS = ['BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'CDIA.JK', 'COIN.JK']
TELEGRAM_TOKEN = 'TELEGRAM_TOKEN'
TELEGRAM_CHAT_ID = 'TELEGRAM_CHAT_ID'

# ========== FUNGSI UTAMA ==========
def get_signal(stock):
    df = yf.download(stock, period="3mo", interval="1d")
    if df.empty or len(df) < 30:
        return None

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal = ""
    if latest['RSI'] < 30:
        signal += f"\U0001F4C9 {stock} oversold (RSI={latest['RSI']:.2f})\n"
    if prev['MACD'] < prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
        signal += f"\u26A1 {stock} MACD Golden Cross\n"
    if latest['Close'] > latest['MA20'] > latest['MA50']:
        signal += f"\U0001F680 {stock} trend naik (MA20 > MA50)\n"

    return signal if signal else None

def send_telegram(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    requests.post(url, data=data)

def run_scanner():
    today = datetime.date.today().strftime("%Y-%m-%d")
    header = f"\U0001F4C8 Watchlist {today}:\n\n"
    signals = [get_signal(stock) for stock in STOCKS]
    results = [s for s in signals if s]
    if results:
        send_telegram(header + '\n'.join(results))
    else:
        send_telegram("\U0001F634 Tidak ada sinyal entry/exit yang kuat hari ini.")

if __name__ == "__main__":
    run_scanner()
