import yfinance as yf
import pandas as pd
import datetime
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

# ============ CONFIG ==============
STOCKS = ['BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'CDIA.JK', 'COIN.JK']
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'

# ========== FUNGSI UTAMA ==========
def get_signal(stock):
    df = yf.download(stock, period="3mo", interval="1d", auto_adjust=True)
    if df.empty or len(df) < 50:
        return None

    # Indikator RSI
    rsi = RSIIndicator(close=df['Close'], window=14).rsi()
    df['RSI'] = pd.Series(rsi.values.flatten(), index=df.index)

    # Indikator MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = pd.Series(macd.macd().values.flatten(), index=df.index)
    df['MACD_signal'] = pd.Series(macd.macd_signal().values.flatten(), index=df.index)

    # Indikator Moving Average
    ma20 = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    ma50 = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['MA20'] = pd.Series(ma20.values.flatten(), index=df.index)
    df['MA50'] = pd.Series(ma50.values.flatten(), index=df.index)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal = ""
    if latest['RSI'] < 30:
        signal += f"📉 {stock} oversold (RSI={latest['RSI']:.2f})\n"
    if prev['MACD'] < prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
        signal += f"⚡ {stock} MACD Golden Cross\n"
    if latest['Close'] > latest['MA20'] > latest['MA50']:
        signal += f"🚀 {stock} Uptrend (MA20 > MA50)\n"

    return signal if signal else None

def send_telegram(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    requests.post(url, data=data)

def run_scanner():
    today = datetime.date.today().strftime("%Y-%m-%d")
    header = f"📈 Watchlist {today}:\n\n"
    signals = [get_signal(stock) for stock in STOCKS]
    results = [s for s in signals if s]
    if results:
        send_telegram(header + '\n'.join(results))
    else:
        send_telegram(f"😴 Tidak ada sinyal entry/exit yang kuat hari ini.")

if __name__ == "__main__":
    run_scanner()
