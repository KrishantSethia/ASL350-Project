import pandas as pd
from prophet import Prophet
from pandas import to_datetime
from pandas import DataFrame
df = pd.read_csv("D:/ASL PROJECT/temperature final/temperature_data.csv")
df.columns = ['ds', 'y']
df['ds'] = to_datetime(df['ds'])
model = Prophet()
model.fit(df)
future = []
for year in range(2011, 2016):
    for month in range(1, 13):
        date = '{}-{:02d}'.format(year, month)
        future.append([date])
future = DataFrame(future)
future.columns = ['ds']
future['ds'] = to_datetime(future['ds'])
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
forecast.to_csv(
    "D:/ASL PROJECT/temperature final/temperature_forecast_final.csv", index=False)
