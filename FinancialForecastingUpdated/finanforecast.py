import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings


def preprocess_data(file_path, date_col, sales_col):
    # Load data based on file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, parse_dates=[date_col], index_col=date_col)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, parse_dates=[date_col], index_col=date_col)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep='\t', parse_dates=[date_col], index_col=date_col)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, Excel, or TXT file.")

    # Ensure sales column is numeric and handle money symbols
    df[sales_col] = df[sales_col].replace({r'[^\d.]': ''}, regex=True).astype(float)

    # Sorting the data by date
    df = df.sort_index()

    return df


def forecast_sales(df, sales_col, seasonal_length=12, trend_strength=1, seasonality_strength=1, randomness_strength=1,
                   months_to_predict=24):
    # Suppress warnings from ARIMA fitting
    warnings.filterwarnings("ignore")

    try:
        # ARIMA model fitting
        model = sm.tsa.ARIMA(df[sales_col], order=(1, 1, 1))
        arima_results = model.fit()

        # Forecast future values
        forecast = arima_results.get_forecast(steps=months_to_predict)
        forecast_index = pd.date_range(start=df.index[-1], periods=months_to_predict + 1, freq='MS')[1:]
        forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['forecast'])

        # Combine actual sales and forecast for plotting
        combined_df = pd.concat([df[sales_col], forecast_df], axis=1)

        # Plotting the forecast
        plt.figure(figsize=(10, 6))
        plt.plot(combined_df.index, combined_df[sales_col], label='Actual Sales')
        plt.plot(combined_df.index, combined_df['forecast'], label='Forecast', linestyle='--')
        plt.title('Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        return combined_df, arima_results

    except Exception as e:
        print(f"An error occurred while forecasting: {e}")
        return None, None


def main():
    # Request necessary inputs
    data_path = input("Please provide the path to your sales data file (CSV, Excel, or TXT): ")
    date_col = input("What is the name of the column with dates? ")
    sales_col = input("What is the name of the column with sales data? ")
    months_to_predict = int(input("How many months into the future would you like to forecast? "))

    # Load and preprocess the data
    df = preprocess_data(data_path, date_col, sales_col)

    # Perform sales forecasting
    forecasted_df, arima_results = forecast_sales(df, sales_col, months_to_predict=months_to_predict)

    if forecasted_df is not None and arima_results is not None:
        # Output some key statistics and plots
        print("\nSummary of the ARIMA model:")
        print(arima_results.summary())

        print("\nDescriptive statistics of the forecasted sales:")
        print(forecasted_df.describe())
    else:
        print("Forecasting could not be completed. Please check the input data or parameters.")


if __name__ == "__main__":
    main()
