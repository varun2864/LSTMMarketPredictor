# Stock Price Movement Prediction using LSTM

predicts the movement of stock prices using Long Short-Term Memory (LSTM) networks. Utilizes historical stock data to train an LSTM model and predict whether the price of a stock will go up or down in the future.

## Project Overview

implements an LSTM model to predict the movement of the NIFTY 50 index. It uses technical indicators and historical data to train the model.

**Steps involved:**

1. **Data Acquisition:** The project uses the `yfinance` library to download historical stock data for the NIFTY 50 index from Yahoo Finance.
2. **Data Preprocessing:** The data is cleaned and preprocessed, including handling missing values and creating new features like moving averages and ratios.
3. **Feature Engineering:** The project uses technical indicators such as RSI, MACD, Bollinger Bands and EMA to derive additional features for the model.
4. **Model Building:** An LSTM model is built using the `tensorflow.keras` library.
5. **Model Training:** The LSTM model is trained using the historical stock data.
6. **Model Evaluation:** The model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Requirements

- Python 3.7+
- Libraries: `yfinance`, `ta`, `scikeras`, `scikit-learn`, `tensorflow`, `pandas`, `numpy`

## Usage

1. Clone the repository.
2. Install the required libraries.
3. Run the Jupyter Notebook `marketpredictorlstm.ipynb`.

## Results

The LSTM model achieves an accuracy of 60% on the test data.

## Future Work

- Experimenting with different architectures and hyperparameters for the LSTM model.
- Incorporating more technical indicators and fundamental data.
- Develop a trading strategy based on the model's predictions.
