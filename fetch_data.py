import yfinance as yf
import pandas as pd
import datetime
import os

# List ticker IDX yang ingin dipantau (ganti sesuai kebutuhan)
TICKERS = os.getenv("FILTER_TICKER")

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

            data_rows.append({
                'ticker': ticker.replace('.JK', ''),
                'latest_close': latest['Close'],
                'latest_volume': latest['Volume'],
                'PER': info.get('trailingPE', 0),
                'PBV': info.get('priceToBook', 0),
                'Foreign_Buy_Ratio': 0.5,  # Dummy, nanti isi via RTI/EDA
                'bandarmology_score': 0,   # Dummy, nanti isi via RTI broker summary
                'macro_sentiment': 0.0,    # Dummy, nanti isi via BI/inflasi
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
