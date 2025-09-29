import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Load and prepare data ---
df = pd.read_csv("data/cleaned/daily_data.csv")
features = ["date", "max_temp", "min_temp"]
df = df[features]

# create mean temperature
df["mean_temp"] = (df["max_temp"] + df["min_temp"]) / 2

# ensure datetime index
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# --- Keep only last 10 years ---
start_date = df.index.max() - pd.DateOffset(years=5)
df = df.asfreq("D")  # explicit daily frequency
ts = df.loc[start_date:, "mean_temp"]

# --- Stationarity check ---
result = adfuller(ts)
print("ADF p-value:", result[1])
if result[1] > 0.05:
    print("Series is non-stationary, differencing will be applied.")
else:
    print("Series is stationary.")

# --- ACF and PACF plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, lags=100, ax=axes[0])
plot_pacf(ts, lags=100, ax=axes[1])
#plt.show()

# --- Train/test split ---
train_size = int(len(ts) * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

# --- Fit daily SARIMA model ---
# order=(p,d,q), seasonal_order=(P,D,Q,s)
# s=365 for yearly seasonality
model = SARIMAX(train,
                order=(2,0,2),
                seasonal_order=(1,1,1,365),
                enforce_stationarity=False,
                enforce_invertibility=False)

print("Starting SARIMA fit...")
model_fit = model.fit(disp=True)
print("SARIMA fit complete.")

print(model_fit.summary())

# --- Forecast ---
forecast = model_fit.forecast(steps=len(test))

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label="Train", color="#203147")
plt.plot(test.index, test, label="Test", color="#01ef63")
plt.plot(test.index, forecast, label="SARIMA Forecast", color="orange")
plt.title("Temperature Forecast (Daily SARIMA)")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.legend()
plt.show()
