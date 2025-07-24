# model.py
import pandas as pd
import numpy as np
from indicators import compute_indicators
from signal_engine import generate_signals
from backtest import backtest
from utils import load_options_data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

options_data = load_options_data("options_data_2023.csv")

def main():          
    df = pd.read_csv("spot_with_signals_2023.csv", parse_dates=['datetime']) 

    df = compute_indicators(df)
    df = generate_signals(df, options_data)  
    print("Signal distribution:", df['composite_signal'].value_counts())
    # --- ML Model Training with LSTM ---
    feature_cols = ['rsi', 'wt1', 'wt2', 'ci', 'tci']  # Add more features as needed
    X = df[feature_cols].fillna(0).values 
    y = df['composite_signal'].fillna(0).values
    print(df['composite_signal'].value_counts())
    # Convert labels to categorical for classification
    y_cat = to_categorical(y + 1, num_classes=3)  # Assuming signals are -1, 0, 1

    # Reshape X for LSTM: [samples, timesteps, features]
    # Here, use a window of 10 timesteps
    window = 10
    X_seq = []
    y_seq = []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y_cat[i+window])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train/test split (80/20)
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]  
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:] 

    # Build LSTM model
    model = Sequential([
        LSTM(32, input_shape=(window, X.shape[1]), return_sequences=False),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Predict and evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1) - 1  # Convert back to -1, 0, 1
    y_true = np.argmax(y_test, axis=1) - 1

    print("LSTM Model Evaluation:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))

    # Optionally, add predictions to df for backtesting (align indices)
    df['lstm_pred_signal'] = 0
    for i in range(window, window + len(y_pred)):
        df.at[i, 'lstm_pred_signal'] = y_pred[i - window]

    trades = backtest(df, options_data)

    print("Backtesting completed.")
    print("Trades saved to trades.csv")
    print("Metrics saved to metrics.csv")
    print("Equity and drawdown charts saved as PNGs.")

if __name__ == "__main__":
    main()
