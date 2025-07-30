# train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
DATA_PATH = "historical_idx_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Features & target
X = df.drop(columns=["Target"])
y = df["Target"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"RandomForest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")

# Save RandomForest model
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Prepare for Deep Learning
num_classes = len(np.unique(y))
y_train_c = to_categorical(y_train, num_classes=num_classes)
y_test_c = to_categorical(y_test, num_classes=num_classes)

# Train Deep Neural Network
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_c, epochs=50, batch_size=16, verbose=0)

# Evaluate and save
loss, acc = model.evaluate(X_test, y_test_c, verbose=0)
print(f"DNN Accuracy: {acc:.2f}")
model.save("deep_learning_model.h5")

# Optional: Export retrain function for automation
def retrain_models():
    df = pd.read_csv("historical_idx_dataset.csv")
    df.dropna(inplace=True)

    X = df[['RSI', 'Stoch', 'BB_bbm', 'BB_bbh', 'BB_bbl', 'Volume_Spike']]
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, "scaler.pkl")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    joblib.dump(rf, "random_forest_model.pkl")

    # DNN Model
    model = Sequential([
        Dense(64, input_shape=(X_scaled.shape[1],), activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=20, batch_size=16, verbose=0)
    model.save("deep_learning_model.h5")
