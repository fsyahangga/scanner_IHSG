import pandas as pd
import os

from utils import (
    load_data,
    preprocess_features,
    load_models,
    predict_hybrid_model,
    calculate_technical_indicators,
    get_macro_sentiment,
    detect_candlestick_pattern,
    classify_signal
)

# ========== CONFIG ==========
DATA_PATH = "historical_idx_dataset.csv"
MODEL_DIR = "models"
OUTPUT_CSV = "buy_signals.csv"
FEATURES = [
    "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl", "Volume_Spike",
    "PER", "PBV", "bandarmology_score", "latest_close", "latest_volume"
]
TARGET = "target"

# ========== MAIN ==========
def main():
    # Load dataset & technical indicators
    df = load_data(DATA_PATH)
    df = calculate_technical_indicators(df)

    # Preprocess features
    X, _ = preprocess_features(df, FEATURES, target_column=None)

    # Load models
    rf_model, xgb_model, lstm_model = load_models(MODEL_DIR)

    # Predict
    df["pred_proba"] = predict_hybrid_model(rf_model, xgb_model, lstm_model, X)
    df["macro_sentiment"] = get_macro_sentiment()
    df["candlestick_pattern"] = detect_candlestick_pattern(df)
    df["signal"] = df["pred_proba"].apply(classify_signal)
    
    # Simpan hasil untuk broadcast
    columns_to_save = [
        "ticker", "RSI", "Stoch", "BB_bbm", "BB_bbh", "BB_bbl",
        "Volume_Spike", "PER", "PBV", "bandarmology_score",
        "latest_close", "latest_volume", "Foreign_Buy_Ratio",
        "macro_sentiment", "candlestick_pattern",
        "pred_proba", "signal"
    ]
    df[columns_to_save].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Scanner selesai. Output tersimpan di: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
