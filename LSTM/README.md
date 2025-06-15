LSTM Prediction Project
This project implements Long Short-Term Memory (LSTM) models for time-series forecasting of financial instruments (AAPL, BTC, ETH, SP500). The repository contains code for training, fine-tuning, and interactive prediction, along with model outputs and results.
Project Structure

LSTMCode/: Main directory containing source code and data.

data/: Stores input datasets (e.g., AAPL_data.csv, BTC_data.csv, ETH_data.csv, SPY_data.csv).
models/: Contains trained LSTM models for each instrument.
aapl/: Models for AAPL (e.g., aapl_seq100_LSTM.h5).
btc/: Models for BTC (e.g., btc_seq100_LSTM.h5).
eth/: Models for ETH (e.g., eth_seq100_LSTM.h5).
sp500/: Models for SP500 (e.g., sp500_seq100_LSTM.h5).


publications/: Reserved for publications that were used in some capacity to create the codes.
interactive.ipynb: Jupyter notebook for interactive model training and prediction with a user interface.
LSTM.ipynb: Main notebook for training LSTM models, generating predictions, and fine-tuning.


wyniki_predykcji/: Directory storing prediction results and metrics for each instrument.

aapl/: Prediction outputs and metrics (e.g., aapl_tyg_LSTM.txt, aapl_tyg_LSTM_metrics.txt).
btc/: Prediction outputs and metrics (e.g., btc_tyg_LSTM.txt, btc_tyg_LSTM_metrics.txt).
eth/: Prediction outputs and metrics (e.g., eth_tyg_LSTM.txt, eth_tyg_LSTM_metrics.txt).
sp500/: Prediction outputs and metrics (e.g., sp500_tyg_LSTM.txt, sp500_tyg_LSTM_metrics.txt).


wyniki_predykcji_every1h/: Contains hourly prediction results (prediction for every hour based on real data + the model) - from interactive.ipynb code.


Code Overview
LSTM.ipynb

Purpose: Trains LSTM models for forecasting closing prices of AAPL, BTC, ETH, and SP500 over different time horizons (1 week, 2 weeks, 1 month, 3 months).
Key Features:
Implements multiple LSTM architectures (stacked LSTM, bidirectional LSTM, LSTM with attention, LSTM + Transformer).
Uses MinMaxScaler for data normalization.
Supports sequence lengths (e.g., seq_len=100).
Saves models to .h5 files (e.g., wyniki_predykcji/aapl/model_seq100.h5).
Generates predictions and saves them as .txt files (e.g., aapl_tyg_LSTM.txt).
Computes metrics (MAPE, R²) and saves them (e.g., aapl_tyg_LSTM_metrics.txt).
Includes fine-tuning functionality for pre-trained models.


Dependencies: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Scikit-learn.

interactive.ipynb

Purpose: Provides an interactive interface for training or fine-tuning models and visualizing predictions.
Key Features:
Supports file uploads for custom datasets (CSV format with timestamp and price columns).
Allows selection of instruments, architectures, and training modes (train from scratch or fine-tune).
Uses a DatePicker for train/test split.
Visualizes predictions using Plotly.
Saves models optionally (.h5 format).


Dependencies: TensorFlow, Keras, Pandas, NumPy, Plotly, Scikit-learn, IPyWidgets, IPython.

Usage

Setup Environment:

Install required packages:pip install tensorflow pandas numpy matplotlib scikit-learn plotly ipywidgets


Ensure Python 3.7+ and Jupyter Notebook/JupyterLab are installed.


Running LSTM.ipynb:

Open LSTM.ipynb in Jupyter.
Ensure datasets are in LSTMCode/data/ (e.g., AAPL_data.csv).
Run cells to train models, generate predictions, and save results.
Fine-tune models by specifying model paths and sequence lengths.


Running interactive.ipynb:

Open interactive.ipynb in Jupyter.
Use the UI to upload a CSV file or select a default dataset.
Choose an instrument, architecture, and training mode.
Set the train/test split date and number of epochs.
Click "Uruchom trening i predykcję" to train and visualize results.


Viewing Results:

Check wyniki_predykcji/<instrument>/ for prediction outputs (e.g., aapl_tyg_LSTM.txt) and metrics (e.g., aapl_tyg_LSTM_metrics.txt).
Models are saved in LSTMCode/models/<instrument>/ (e.g., aapl_seq100_LSTM.h5).

Model Details

Naming Convention for Models:

Format: <instrument>_seq<length>_LSTM.h5 (e.g., aapl_seq100_LSTM.h5).
Stored in LSTMCode/models/<instrument>/.


Prediction Outputs:

Format: <instrument>_<period>_LSTM.txt (e.g., aapl_tyg_LSTM.txt for 1-week predictions).
Periods: tyg (1 week), 2tyg (2 weeks), msc (1 month), 3msc (3 months).


Metrics Files:

Format: <instrument>_<period>_LSTM_metrics.txt.
Contains MAPE, R², and model path.