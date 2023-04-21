import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# https://facebook.github.io/prophet/docs/quick_start.html\
train_url = "train.csv"
test_url = "test.csv"

data_train = pd.read_csv(train_url, sep=';')
data_test = pd.read_csv(test_url, sep=';')
pd.set_option('display.max_columns', None)

data_train['ds'] = pd.to_datetime(data_train['<DATE>'], format="%Y%m%d")
data_test['ds'] = pd.to_datetime(data_test['<DATE>'], format="%Y%m%d")
data_train['y'] = data_train['<CLOSE>']

model = Prophet()
model.fit(data_train)

future = model.make_future_dataframe(periods=105)
future.tail()

forecast = model.predict(future)
forecast.set_index('ds', inplace=True)
data_test.set_index('ds', inplace=True)
print(forecast)
plt.plot(forecast.index, forecast['yhat'], label='Predicted')
plt.plot(data_test.index, data_test['<CLOSE>'], label='Real')
plt.legend()
plt.show()


