import ta
import yfinance as yf
n50 = yf.Ticker("^NSEI")

n50 = n50.history(period = "max")

del n50["Dividends"]
del n50["Stock Splits"]
n50 = n50[n50["Volume"]!=0]

n50["tomorrow"] = n50["Close"].shift(-1)
n50["target"] = (n50["tomorrow"]>n50["Close"]).astype(int)

n50 = n50.dropna()

n50

import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input

from scikeras.wrappers import KerasClassifier

scaler  = MinMaxScaler()
predictors_n50 = ["Volume", "High", "Close", "Open", "Low"]

horizons = [2, 5, 60, 260, 1000]
#new_predictors = []

for horizon in horizons:
  rolling_avgs = n50.rolling(horizon).mean()

  ratio_col = f"Close_Ratio_{horizon}"
  n50[ratio_col] = n50["Close"] / rolling_avgs["Close"]

  trend_col = f"Trend_{horizon}"
  n50[trend_col] = n50.shift(1).rolling(horizon).sum()["target"]

  predictors_n50 += [ratio_col, trend_col]

n50 = n50.dropna()
n50[predictors_n50] = scaler.fit_transform(n50[predictors_n50])

import numpy as np

def sequences(data, target, time_steps = 10):
  data_seq = []
  target_seq = []
  for i in range(len(data) - time_steps):
    data_seq.append(data.iloc[i:(i+time_steps)].values)
    target_seq.append(target.iloc[i+time_steps])
  return np.array(data_seq), np.array(target_seq)

def create_model(units = 150, dropout_rate = 0.1, batch_size = 128, epochs = 10):
  model = Sequential()
  model.add(Input(shape = (x_train.shape[1], x_train.shape[2])))
  model.add(LSTM(units, return_sequences = True))
  model.add(Dropout(dropout_rate))
  model.add(LSTM(units = units))
  model.add(Dropout(dropout_rate))
  model.add(Dense(units = 1, activation = "sigmoid"))
  model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
  return model

x, y = sequences(n50[predictors_n50], n50["target"], time_steps = 10)

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1, random_state=1)

model = KerasClassifier(model=create_model, units=200, dropout_rate=0.1, batch_size=128, epochs=10)
results = model.fit(x_train, y_train, validation_data=(x_val, y_val))

accuracy = model.score(x_test, y_test)
print(f"accuracy: {accuracy}")

predictions = model.predict(x_test)
precision = precision_score(y_test, predictions)
print(f"precision: {precision}")

prec = 0

for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
      prec = prec+1\

print(prec/len(y_test))

from sklearn.metrics import recall_score, f1_score

recall = recall_score(y_test, predictions)
print(f"Recall: {recall}")

f1 = f1_score(y_test, predictions)
print(f"F1 Score: {f1}")

param_grid = {
    'model__units': [50, 100, 150],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(x_train, y_train)

_, accuracy = grid_result.best_estimator_.model_.evaluate(x_test, y_test, verbose=0)

print(f"Best Accuracy: {accuracy}")
print(f"Best Parameters: {grid_result.best_params_}")

#output with indicators came to be 150 units, 0.1 dropout rate, 10 epochs, 128 batch size
