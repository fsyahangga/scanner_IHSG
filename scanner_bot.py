import yfinance as yf
import pandas as pd
import talib
import datetime
import requests

# ============ CONFIG ==============
STOCKS = ['BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'CDIA.JK', 'COIN.JK']  # Watchlist kamu
TELEGRAM_TOKEN = 'TELEGRAM_TOKEN'
TELEGRAM_CHAT_ID = 'TELEGRAM_CHAT_ID'

# ========== FUNGSI UTAMA ==========
def get_signal(stock):
    df = yf.download(stock, period="3mo", interval="1d")
    if df.empty or len(df) < 30:
        return None

    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], macdsignal, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['Close'], timeperiod=50)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal = ""
    if latest['RSI'] < 30:
        signal += f"📉 {stock} oversold (RSI={latest['RSI']:.2f})\n"
    if prev['MACD'] < prev['MACD'] and latest['MACD'] > macdsignal[-1]:
        signal += f"⚡ {stock} MACD Golden Cross\n"
    if latest['Close'] > latest['MA20'] > latest['MA50']:
        signal += f"🚀 {stock} trend naik (MA20 > MA50)\n"

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
