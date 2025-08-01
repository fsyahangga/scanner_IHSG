import pandas as pd
import telegram
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Inisialisasi bot Telegram
bot = telegram.Bot(token=TELEGRAM_TOKEN)

def send_all_signals():
    try:
        try:
            df = pd.read_csv("buy_signals.csv")
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()

        if df.empty:
            message = "📉 Tidak ada data hasil analisis hari ini."
        else:
            message = "📊 **Hasil Pemindaian Saham Hari Ini**\n\n"
            for _, row in df.iterrows():
                message += (
                    f"• `{row['ticker']}` | 📌 Rekomendasi: *{row['recommendation'].upper()}*\n"
                    f"  💹 RSI: {row['RSI']:.2f} | 🔍 Stoch: {row['Stoch']:.2f}\n"
                    f"  💼 PER: {row['PER']:.2f} | 📚 PBV: {row['PBV']:.2f}\n"
                    f"  🧠 Bandarmology: {row['Bandarmology_Score']:.2f} | 🌏 Macro: {row['macro_sentiment']:.2f}\n"
                    f"  💰 Close: {row['latest_close']:,} | Vol: {row['latest_volume']:,}\n"
                    f"  🔎 Confidence: {row['confidence']:.2%}\n\n"
                )

        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=telegram.ParseMode.MARKDOWN)

    except Exception as e:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ Gagal mengirim sinyal saham: {e}")

if __name__ == "__main__":
    send_all_signals()
