import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prepare_data(filepath, date_col, sales_col):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace from column names
    df = df[[date_col, sales_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    return df

def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(f'{label}: {value}')
    if result[1] <= 0.05:
        print("Data is stationary.")
        return True
    else:
        print("Data is non-stationary.")
        return False

def difference_data(df, sales_col):
    df['First Difference'] = df[sales_col] - df[sales_col].shift(1)
    df['Seasonal First Difference'] = df[sales_col] - df[sales_col].shift(12)
    return df

def plot_acf_pacf(df, col_name, lags):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    plot_acf(df[col_name].dropna(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    plot_pacf(df[col_name].dropna(), lags=lags, ax=ax2)
    plt.show()

def arima_model(df, sales_col, order):
    model = ARIMA(df[sales_col], order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def sarimax_model(df, sales_col, order, seasonal_order):
    model = sm.tsa.statespace.SARIMAX(df[sales_col], order=order, seasonal_order=seasonal_order)
    results = model.fit()
    print(results.summary())
    return results

def forecast_future(df, model, periods):
    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1, periods+1)]
    future_df = pd.DataFrame(index=future_dates, columns=df.columns)
    future_df = pd.concat([df, future_df])
    future_df['forecast'] = model.predict(start=len(df), end=len(future_df)-1, dynamic=True)
    return future_df

def forecast_plot(df, model, sales_col, start, end):
    df['forecast'] = model.predict(start=start, end=end, dynamic=True)
    combined_df = pd.concat([df[sales_col], df['forecast']], axis=1)
    combined_df.columns = ['Historical Sales', 'Forecast']
    combined_df['Forecast'].iloc[:start] = np.nan  # Ensure forecast starts after historical data
    combined_df.plot(figsize=(12, 8))
    plt.title('Sales and Forecast')
    plt.show()
    return combined_df

def plot_future_forecast(df, model, sales_col, periods):
    future_df = forecast_future(df, model, periods)
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df[sales_col], label='Historical Sales')
    plt.plot(future_df.index, future_df['forecast'], label='Future Forecast', linestyle='--')
    plt.title('Future Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

def calculate_accuracy(df, model, sales_col, start, end):
    forecast_df = forecast_plot(df, model, sales_col, start, end)
    actual = df[sales_col].iloc[start:end+1]
    forecasted = forecast_df['Forecast'].iloc[start:end+1]

    # Calculate metrics
    mae = mean_absolute_error(actual, forecasted)
    mse = mean_squared_error(actual, forecasted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, forecasted)

    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Calculate and print accuracy percentage
    accuracy = 100 - (rmse / actual.mean() * 100)
    print(f"Forecast Accuracy: {accuracy:.2f}%")

def main():
    # User inputs
    filepath = input("Enter the path to your CSV file: ")
    date_col = input("Enter the name of the Date column: ")
    sales_col = input("Enter the name of the Sales column: ")

    # Load and prepare data
    df = load_and_prepare_data(filepath, date_col, sales_col)
    
    # Display the data
    print(df.head())
    print(df.tail())

    # Summary statistics
    print(df.describe())

    # Plot the data
    df[sales_col].plot(title="Sales Over Time")
    plt.show()

    # Stationarity test
    is_stationary = adfuller_test(df[sales_col])

    # Differencing the data if not stationary
    if not is_stationary:
        df = difference_data(df, sales_col)
        diff_col = 'Seasonal First Difference'
    else:
        diff_col = sales_col

    # User input for ACF and PACF lags
    lags = int(input("Enter the number of lags for ACF and PACF plots: "))

    # Plot ACF and PACF
    plot_acf_pacf(df, diff_col, lags)

    # User-friendly input for ARIMA order
    trend_strength = int(input("On a scale of 0-2, how strong is the trend in your data? (0: None, 1: Mild, 2: Strong): "))
    seasonality_strength = int(input("On a scale of 0-2, how strong is the seasonality in your data? (0: None, 1: Mild, 2: Strong): "))
    noise_level = int(input("On a scale of 0-2, how much random noise do you see in your data? (0: None, 1: Some, 2: A lot): "))

    # Convert user input into ARIMA order
    arima_order = (trend_strength, 1, noise_level)

    # User input for SARIMAX seasonal order
    seasonal_period = int(input("Enter the length of the seasonal cycle in months (e.g., 12 for yearly seasonality): "))
    sarimax_seasonal_order = (seasonality_strength, 1, noise_level, seasonal_period)

    # User input for forecasting periods
    periods = int(input("Enter the number of periods for forecasting: "))

    # ARIMA Model
    arima_results = arima_model(df, sales_col, arima_order)
    # Display ARIMA forecast plot
    forecast_plot(df, arima_results, sales_col, start=len(df)-12, end=len(df)-1)

    # SARIMAX Model
    sarimax_results = sarimax_model(df, sales_col, arima_order, sarimax_seasonal_order)
    # Display SARIMAX forecast plot
    forecast_plot(df, sarimax_results, sales_col, start=len(df)-12, end=len(df)-1)

    # Plot future forecast
    plot_future_forecast(df, sarimax_results, sales_col, periods)

    # Calculate and print accuracy
    calculate_accuracy(df, sarimax_results, sales_col, start=len(df)-12, end=len(df)-1)

if __name__ == "__main__":
    main()
