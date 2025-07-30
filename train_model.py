# ===== train_model.py =====
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def retrain_models():
    # Load dataset
    df = pd.read_csv("historical_idx_dataset.csv")

    # Pastikan semua kolom numerik kecuali target dan ticker
    features = df.drop(columns=["ticker", "target"])  # Drop kolom string
    target = df["target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save model & scaler
    joblib.dump(rf_model, "random_forest_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Model retrained and saved successfully.")

# Only run when executed directly
if __name__ == "__main__":
    retrain_models()
