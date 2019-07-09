from u8timeseries.models.autoregressive_model import AutoRegressiveModel
import numpy as np
from ..timeseries import TimeSeries

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class LSTM(AutoRegressiveModel):

    def __init__(self, lstm_hidden_sizes, dense_hidden_sizes, recurrent_dropout=1.0, epochs=10, batch_size=1,
                 timesteps=3):
        super().__init__()
        # Stores training date information:
        self.training_series: TimeSeries = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.timesteps = timesteps

        # create model
        self.model = Sequential()
        for size in lstm_hidden_sizes[:-1]:
            self.model.add(LSTM(size, return_sequences=True, recurrent_dropout=recurrent_dropout))
        self.model.add(LSTM(lstm_hidden_sizes[-1]))
        self.model.add(Dense(dense_hidden_sizes[-1]))
        self.model.compile(loss="mean_squared_error", optimizer="adam")

        # state
        self._fit_called = False

    def create_dataset(self, series, timesteps):
        data_in, data_out = [], []
        for i in range(len(series) - timesteps - 1):
            data_in.append(series[i:(i + timesteps)])
            data_out.append(series[i + timesteps])
        return np.array(data_in), np.array(data_out).reshape(-1, 1)

    def fit(self, series: TimeSeries) -> None:
        super().fit(series)
        self.training_series = series
        data_in, data_out = self.create_dataset(series.values(), self.timesteps)
        self.model.fit(data_in, data_out, epochs=self.epochs, batch_size=self.batch_size, shuffle=False)
        self._fit_called = True

    def predict(self, n: int) -> TimeSeries:
        super().predict(n)
        in_values = self.training_series.values()[:-self.timesteps]
        predictions = []
        for i in range(n):
            pred = self.model.predict(in_values)
            in_values = in_values[1:] + [pred]
            predictions.append(pred)

        return self._build_forecast_series(np.array(predictions))


