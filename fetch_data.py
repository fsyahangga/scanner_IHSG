import yfinance as yf
import pandas as pd
import datetime
import os
from utils import(
    get_foreign_flow_data,
    get_macro_sentiment
)
from dotenv import load_dotenv
load_dotenv()

# List ticker IDX yang ingin dipantau (ganti sesuai kebutuhan)
TICKERS = os.getenv("FILTER_TICKER")
if TICKERS:
    TICKERS = [ticker.strip() for ticker in TICKERS.split(",")]
else:
    raise ValueError("❌ FILTER_TICKER environment variable is not set.")

def fetch_yfinance_data(tickers):
    data_rows = []

    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="5d", interval="1d")

            if hist.empty:
                continue

            latest = hist.iloc[-1]
            info = yf_ticker.info
            macro_sentiment = get_macro_sentiment()
            net_foreign, foreign_ratio = get_foreign_flow_data(ticker)

            data_rows.append({
                'ticker': ticker.replace('.JK', ''),
                'latest_close': latest['Close'],
                'latest_volume': latest['Volume'],
                'PER': info.get('trailingPE', 0),
                'PBV': info.get('priceToBook', 0),
                'Foreign_Buy_Ratio': foreign_ratio,  # Dummy, nanti isi via RTI/EDA
                'bandarmology_score': 0,   # Dummy, nanti isi via RTI broker summary
                'macro_sentiment': macro_sentiment,    # Dummy, nanti isi via BI/inflasi
                'candlestick_pattern': '', # Akan dihitung ulang di scanner
            })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return pd.DataFrame(data_rows)


if __name__ == "__main__":
    df = fetch_yfinance_data(TICKERS)

    if not df.empty:
        df.to_csv("latest_realtime_data.csv", index=False)
        print("✅ latest_realtime_data.csv berhasil disimpan.")
    else:
        print("⚠️ Tidak ada data yang berhasil diambil.")
