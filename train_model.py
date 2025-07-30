import pandas as pd
import yfinance as yf
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Download historical data (example: BBCA.JK)
ticker = 'BBCA.JK'
data = yf.download(ticker, period="5y")
data.dropna(inplace=True)

# Calculate features
data['Return'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = 100 - (100 / (1 + data['Return'].rolling(window=14).mean()))
data['VolSpike'] = data['Volume'] / data['Volume'].rolling(20).mean()
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)

# Features & target
features = ['Return', 'MA5', 'MA20', 'RSI', 'VolSpike']
X = data[features]
y = data['Target']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train RandomForest (shallow model)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
joblib.dump(rfc, 'random_forest_model.pkl')

# Train Deep Learning model (Keras)
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
model.save('deep_learning_model.h5')

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
