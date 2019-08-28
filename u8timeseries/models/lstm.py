from u8timeseries.models.autoregressive_model import AutoRegressiveModel
import numpy as np
from ..timeseries import TimeSeries

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class LSTMModel(AutoRegressiveModel):
    def __init__(self, lstm_hidden_sizes=[20], dense_hidden_sizes=[1], dropout=0.0,
                 recurrent_dropout=0.0,
                 epochs=10, batch_size=1, back_steps=3, forecast_steps=3, validation_split=0.0):
        super().__init__()
        print("NEW MODEL")
        # Stores training date information:
        self.training_series: TimeSeries = None
        self.lstm_hidden_sizes = lstm_hidden_sizes
        self.dense_hidden_sizes = dense_hidden_sizes
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.back_steps = back_steps
        self.forecast_steps = forecast_steps
        self.validation_split = validation_split
        # state
        self._fit_called = False

    def __split_dataset(self, series, back_steps):
        data_in, data_out = [], []
        for i in range(len(series) - back_steps - 1):
            data_in.append(series[i:(i + back_steps)])
            data_out.append(series[i + back_steps])
        data_in = np.array(data_in, dtype=np.float32)
        data_out = np.array(data_out, dtype=np.float32)
        data_in = data_in.reshape(data_in.shape[0], data_in.shape[1], 1)
        data_out = data_out.reshape(-1, 1)
        return data_in, data_out

    def fit(self, series: TimeSeries) -> None:
        super().fit(series)
        self.model = Sequential()
        for size in self.lstm_hidden_sizes[:-1]:
            self.model.add(LSTM(size, return_sequences=True, recurrent_dropout=self.recurrent_dropout))
        self.model.add(LSTM(self.lstm_hidden_sizes[-1]))
        for size in self.dense_hidden_sizes:
            self.model.add(Dropout(self.dropout))
            self.model.add(Dense(size, activation='relu'))
        self.model.compile(loss="mean_absolute_error", optimizer="adam")

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.training_series = series
        self.scaled_series = self.scaler.fit_transform(series.values().reshape(-1, 1))
        data_in, data_out = self.__split_dataset(self.scaled_series, self.back_steps)
        self.model.fit(data_in, data_out, epochs=self.epochs, batch_size=self.batch_size, shuffle=False,
                       validation_split=self.validation_split, verbose=0)
        self._fit_called = True

    def predict(self, n: int) -> TimeSeries:
        super().predict(n)
        in_values = self.scaled_series[-self.back_steps:]
        predictions = []
        for i in range(n):
            pred = self.model.predict(in_values.reshape(1, in_values.shape[0], in_values.shape[1]))
            in_values = np.concatenate([in_values, pred], axis=0)
            in_values = in_values[1:]
            predictions.append(pred)
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).reshape(-1)
        return self._build_forecast_series(predictions)


