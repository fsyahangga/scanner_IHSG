import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import requests
import os

# List saham
STOCKS = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "ARTO.JK"]

def get_technical_indicators(df):
    indicators = {}
    
    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14).rsi()
    indicators['RSI'] = rsi.iloc[-1]

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    indicators['BB_Upper'] = bb.bollinger_hband().iloc[-1]
    indicators['BB_Lower'] = bb.bollinger_lband().iloc[-1]
    indicators['BB_Price_Position'] = (
        "above_upper" if df['Close'].iloc[-1] > indicators['BB_Upper']
        else "below_lower" if df['Close'].iloc[-1] < indicators['BB_Lower']
        else "inside"
    )

    # Stochastic
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    indicators['Stoch_K'] = stoch.stoch().iloc[-1]
    indicators['Stoch_D'] = stoch.stoch_signal().iloc[-1]

    return indicators

def get_fundamentals(ticker):
    info = ticker.info
    return {
        "PER": info.get("trailingPE"),
        "PBV": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "ProfitMargin": info.get("profitMargins")
    }

def generate_signal(tech, fund):
    signal = "NEUTRAL"

    # Rule-based logic
    if tech['RSI'] < 30 and tech['BB_Price_Position'] == "below_lower" and tech['Stoch_K'] < 20:
        signal = "BUY ✅"
    elif tech['RSI'] > 70 and tech['BB_Price_Position'] == "above_upper" and tech['Stoch_K'] > 80:
        signal = "SELL ❌"
    elif fund["PER"] is not None and fund["PER"] < 10 and fund["ROE"] and fund["ROE"] > 0.15:
        signal = "BUY ✅ (Fundamental)"

    return signal

def run_scanner():
    report = []

    for symbol in STOCKS:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo", interval="1d")
        if df.empty or len(df) < 20:
            continue

        tech = get_technical_indicators(df)
        fund = get_fundamentals(ticker)
        signal = generate_signal(tech, fund)

        report.append(f"{symbol}: {signal} (RSI={tech['RSI']:.2f}, PER={fund['PER']}, ROE={fund['ROE']})")

    # Kirim ke Telegram
    send_telegram("\n".join(report))

def send_telegram(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": f"📈📊 *Daily Stock Scanner Result:*\n\n{message}",
        "parse_mode": "Markdown"
    }

    response = requests.post(url, data=data)
    if not response.ok:
        print(f"Failed to send Telegram message: {response.text}")

if __name__ == "__main__":
    run_scanner()
