from .timeseries_model import TimeseriesModel
from statsmodels.tsa.arima_model import ARMA, ARIMA
from pyramid.arima import auto_arima

# Croston method implementation based on R forecast croston
class Croston(TimeseriesModel):

    def __init__(self, alpha=0.1):
        super(Croston, self).__init__()
        self.alpha = alpha
        self.model = None

    def __str__(self):
        return 'Croston({})'.format(self.alpha)

    def fit(self, df, target_column, time_column=None, stepduration_str=None):
        super(Croston, self).fit(df, target_column, time_column, stepduration_str)
        self.values = df[target_column].values

    def predict(self, n):

        forecast = self.model.forecast(steps=n)[0]
        return self._build_forecast_df(forecast)
