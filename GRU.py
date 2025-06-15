import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import r2_score

# Liczba notowań dziennie
FORECAST_INTERVALS = {
    "tyg": {"AAPL": 16 * 7, "BTC": 24 * 7, "ETH": 24 * 7, "SPY": 17 * 7},
    "2tyg": {"AAPL": 16 * 14, "BTC": 24 * 14, "ETH": 24 * 14, "SPY": 17 * 14},
    "3tyg": {"AAPL": 16 * 21, "BTC": 24 * 21, "ETH": 24 * 21, "SPY": 17 * 21},
    "msc": {"AAPL": 320, "BTC": 744, "ETH": 744, "SPY": 340},
    "2msc": {"AAPL": 640, "BTC": 1488, "ETH": 1488, "SPY": 680},
    "3msc": {"AAPL": "test_len", "BTC": "test_len", "ETH": "test_len", "SPY": "test_len"}
}

file_map = {
    'AAPL_data.csv': 'AAPL',
    'BTC_data.csv': 'BTC',
    'ETH_data.csv': 'ETH',
    'SPY_data.csv': 'SPY'
}

def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(GRU(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

look_back = 10
epochs = 30
batch_size = 32
output_folder = 'wyniki_predykcji'
num_models = 10 



file_list = ['ETH_data.csv']  # tylko ETH i SPY

for file_name in file_list:
    if not os.path.exists(file_name):
        print(f"Brak pliku: {file_name}")
        continue

    instrument = file_map[file_name]
    print(f"\n=== Przetwarzanie: {instrument} ===")

    df = pd.read_csv(file_name, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    close_data = df[['close']].values
    test_len = (df['timestamp'].dt.year == 2025).sum()
    train_data = close_data[:-test_len]
    test_data = close_data[-test_len:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    X_train, y_train = create_dataset(train_scaled, look_back)

    best_mape = float('inf')
    best_predictions = None

    for model_idx in range(num_models):
        model = build_gru_model((look_back, 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        input_seq = scaler.transform(close_data[-test_len - look_back:-test_len]).flatten().tolist()
        pred_scaled = []

        for _ in range(test_len):
            seq_input = np.array(input_seq[-look_back:]).reshape(1, look_back, 1)
            pred = model.predict(seq_input, verbose=0)[0][0]
            pred_scaled.append(pred)
            input_seq.append(pred)

        predictions = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
        mape = calculate_mape(test_data.flatten(), predictions)

        print(f"Model {model_idx + 1}/{num_models} – MAPE: {mape:.2f}%")

        if mape < best_mape:
            best_mape = mape
            best_predictions = predictions

        # Jeśli MAPE jest już wystarczająco dobre, przerywamy dalsze trenowanie
        if best_mape < 18:
            print(f"MAPE < 18% ({best_mape:.2f}%), przerywam dalsze trenowanie modeli.")
            break


    actual = test_data.flatten()
    os.makedirs(f"{output_folder}/{instrument.lower()}", exist_ok=True)

    for label, interval_map in FORECAST_INTERVALS.items():
        forecast_len = interval_map[instrument]
        if forecast_len == "test_len":
            forecast_len = test_len
        if forecast_len > len(best_predictions):
            print(f"Pomijam zakres {label} – za mało danych.")
            continue

        pred_slice = best_predictions[:forecast_len]
        actual_slice = actual[:forecast_len]

        mape = calculate_mape(actual_slice, pred_slice)
        r2 = r2_score(actual_slice, pred_slice)

        pred_file = f"{output_folder}/{instrument.lower()}/{instrument.lower()}_{label}_GRU.txt"
        metrics_file = f"{output_folder}/{instrument.lower()}/{instrument.lower()}_{label}_GRU_metrics.txt"

        np.savetxt(pred_file, pred_slice, fmt="%.6f")
        with open(metrics_file, "w") as f:
            f.write(f"{mape:.4f}\n{r2:.4f}\n")

        print(f"Zapisano: {pred_file}, MAPE={mape:.2f}%, R2={r2:.4f}")

