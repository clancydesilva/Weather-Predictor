import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned/daily_data.csv")
features = df.columns

# check correlations of features to each other
corr = df.select_dtypes(include=['float64', 'int64']).corr()

# extract maxtp and mintp to relate to all other features
temps = corr[['max_temp', 'min_temp']]

plt.figure()
sns.heatmap(temps, annot=True)
# plot shows positive correlation between max and min temp with other temp related features
    # this porvides no actual help in predicting temperature as the other features are caused by the max/min temp
    # ARIMA is only choice to predict future temps by only using previous days temperature

# just in case remove other temp related features and check correlations for clarity
non_temp = []
for feature in features:
    if feature.find("temp")==-1:
        non_temp.append(feature)
    else:
        if feature=="max_temp" or feature=="min_temp":
            non_temp.append(feature)


df = df[non_temp]
corr = df.select_dtypes(include=['float64', 'int64']).corr()
temps = corr[['max_temp', 'min_temp']]

plt.figure()
sns.heatmap(temps, annot=True)
plt.show()

# even with checking non temp correlations, there doesnt seem to be any useful features we can use to predict
    # temperature, if there was we could try SARIMAX, but in this case ARIMA will probably be the best choice
    # to complete the task with less complications