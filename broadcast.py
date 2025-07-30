import pandas as pd
import requests
import os

# === Konfigurasi Token & Chat ID Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Load sinyal BUY
df = pd.read_csv("buy_signals.csv")

if df.empty:
    message = "📉 Tidak ada sinyal BUY hari ini."
else:
    message = "📈 *Sinyal BUY Hari Ini*:\n\n"
    for _, row in df.iterrows():
        message += f"• {row['ticker']} | RSI: {row['RSI']}, Volume Spike: {row['Volume_Spike']}\n"

# Kirim ke Telegram
payload = {
    "chat_id": TELEGRAM_CHAT_ID,
    "text": message,
    "parse_mode": "Markdown"
}

response = requests.post(TELEGRAM_API_URL, data=payload)

if response.status_code == 200:
    print("✅ Broadcast berhasil dikirim.")
else:
    print(f"❌ Gagal kirim broadcast. Status: {response.status_code}")
