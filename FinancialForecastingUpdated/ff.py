import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA  # Corrected ARIMA import

# Load the dataset
df = pd.read_csv("D:\\ipd_Project\\data files\\real-estate-sales-730-days-1.csv")

# Display the first few rows of the dataframe
print(df.head())

# Display the last few rows of the dataframe
print(df.tail())

# Cleaning up the data by renaming the columns
df.columns = ["Date", "Sales"]

# Automatically drop rows with any NaN values
df.dropna(inplace=True)

# Display the dataframe after dropping NaN values
print(df.tail())

# Convert Month into Datetime
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Display the dataframe with the new index
print(df.head())

# Summary statistics
print(df.describe())

# Plot the data
df.plot()
plt.show()

# Testing for Stationarity
def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary.")

adfuller_test(df['Sales'])

# First Differencing
df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)

# Seasonal Differencing
df['Seasonal First Difference'] = df['Sales'] - df['Sales'].shift(12)

# Display the first few rows after differencing
print(df.head(14))

# Test Dickey-Fuller Test on Seasonal First Difference
adfuller_test(df['Seasonal First Difference'].dropna())

# Plot Seasonal First Difference
df['Seasonal First Difference'].plot()
plt.show()

# Autocorrelation Plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()

# ACF and PACF plots
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
plot_acf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
plot_pacf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax2)
plt.show()

# ARIMA Model for Non-Seasonal Data
model = ARIMA(df['Sales'], order=(1,1,1))  # Corrected ARIMA model usage
model_fit = model.fit()

print(model_fit.summary())

# Forecasting with ARIMA
df['forecast'] = model_fit.predict(start=90, end=103, dynamic=True)
df[['Sales', 'forecast']].plot(figsize=(12,8))
plt.show()

# SARIMAX Model for Seasonal Data
model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1,1,1,12))
results = model.fit()

df['forecast'] = results.predict(start=90, end=103, dynamic=True)
df[['Sales', 'forecast']].plot(figsize=(12,8))
plt.show()

# Forecasting into the Future
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]

# Create a dataframe for future dates
future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
future_datest_df.tail()

# Concatenate the dataframes
future_df = pd.concat([df, future_datest_df])

# Forecast future values
#future_df['forecast'] = results.predict(start=len(df), end=len(future_df)-1, dynamic=True)
future_df['forecast'] = results.predict(start = len(df)-1, end = len(future_df)-1, dynamic= True)
# Plot the future forecast
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))
plt.show()