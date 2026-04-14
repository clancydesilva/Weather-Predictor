# the ARIMA and SARIMAX modelling takes a long while to run as its calculating averages over many many years
# here i will try a regression model where the target is the next day's temperature recorded
    # by creating a new feature and shifting the column values up by one

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned/daily_data.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# create mean temperature
df["mean_temp"] = (df["max_temp"] + df["min_temp"]) / 2

# the lowest temp recorded is probs gonna be like very early morning/night time, 
    # but people are out and about in the morning when the sun is out and that when the they want to know the temp
    # so im gonna use max_temp here, instead of mean as the lowest is likely to skew the actual useful temp
df["target_temp"] = df["max_temp"].shift(-1)

df.dropna(inplace=True) # drop the last row that has an NaN since we shifted up

# naming features and target
features = ["max_temp","sunshine_duration"]
X = df[features]
y = df["target_temp"]

# training test split
# need to make sure that we are using past data to predict future data
    # therefore will use a manual split
    # for a 8:2 split so test starts at 2016-01-01

split_date = "2016-01-01"
X_train = X.loc[:split_date]
X_test  = X.loc[split_date:]
y_train = y.loc[:split_date]
y_test  = y.loc[split_date:]

model = Ridge(alpha=0.1)
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("\nRidge Regression:")
print("MSE:", mean_squared_error(y_test,predictions))
print("R2 :", r2_score(y_test, predictions))

# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# y_pred_rf = rf_model.predict(X_test)

# print("\nRandom Forest Regression:")
# print("MSE:", mean_squared_error(y_test, y_pred_rf))
# print("R2 :", r2_score(y_test, y_pred_rf))

# plot to compare the 
plt.figure(figsize=(12,6))
plt.plot(y_train.index, y_train, label="Train", color="#203147")
plt.plot(y_test.index, y_test, label="Test", color="#01ef63")
plt.plot(y_test.index, predictions, label="Ridge Forecast", color="orange")
plt.title("Temperature Forecast (Daily Ridge)")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.legend()
#plt.show()

