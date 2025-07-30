import pandas as pd
import telegram
import os
from dotenv import load_dotenv

# Load token dari .env atau environment variable GitHub Actions
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Inisialisasi bot
bot = telegram.Bot(token=TELEGRAM_TOKEN)

def send_buy_signals():
    try:
        df = pd.read_csv("buy_signals.csv")

        if df.empty:
            message = "📉 Tidak ada sinyal BUY hari ini."
        else:
            message = "📈📈📊 Daily Stock Scanner Result:\n"
            for _, row in df.iterrows():
                message += f"• {row['ticker']} | RSI: {row['RSI']} | Stoch: {row['Stoch']} | Foreign: {row['Foreign_Buy_Ratio']:.2f}\n"

        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

    except Exception as e:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ Gagal kirim sinyal BUY: {e}")

if __name__ == "__main__":
    send_buy_signals()
