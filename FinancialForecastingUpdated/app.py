import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset
import io
import base64
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from flask import Flask, render_template, request
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Import ACF and PACF plotting functions
from pandas.tseries.offsets import DateOffset
import io
import base64
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Helper functions

def load_and_prepare_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='ISO-8859-1', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df

def parse_dates(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df.set_index(date_col, inplace=True)
    return df

def adfuller_test(sales):
    result = adfuller(sales)
    if result[1] <= 0.05:
        return True  # Data is stationary
    else:
        return False  # Data is non-stationary

def plot_to_img_tag(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return f'data:image/png;base64,{plot_url}'

def generate_plots(df, sales_col, periods=12):
    # Prepare data for modeling and plots
    df[sales_col] = df[sales_col].astype(float)
    df.dropna(subset=[sales_col], inplace=True)

    # Forecast using SARIMAX model
    sarimax_model = SARIMAX(df[sales_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarimax_fit = sarimax_model.fit(disp=False)

    # Future forecast
    future_df = forecast_future(df, sarimax_fit, periods)

    # Create subplots for different types of plots
    fig, axs = plt.subplots(6, 1, figsize=(14, 18))

    # Prediction Accuracy Plot
    forecast_values = sarimax_fit.predict(start=1, end=len(df), dynamic=False)
    axs[0].plot(df.index, df[sales_col], label='Actual Sales')
    axs[0].plot(df.index, forecast_values, label='SARIMAX Forecast', linestyle='-.')
    axs[0].set_title('Prediction Accuracy Plot')
    axs[0].legend()

    # Future Sales Forecast Plot
    axs[1].plot(df.index, df[sales_col], label='Actual Sales')
    axs[1].plot(future_df['Date'], future_df['Predicted Sales'], label='Forecasted Sales', linestyle='--')
    axs[1].set_title('Future Sales Forecast')
    axs[1].legend()

    # Time Series Plot
    axs[2].plot(df.index, df[sales_col], label='Actual Sales')
    axs[2].set_title('Time Series of Sales Data')
    axs[2].legend()

    # ACF Plot
    plot_acf(df[sales_col], ax=axs[3], lags=40)
    axs[3].set_title('Autocorrelation Function (ACF)')

    # PACF Plot
    plot_pacf(df[sales_col], ax=axs[4], lags=40)
    axs[4].set_title('Partial Autocorrelation Function (PACF)')

    # Autocorrelation Plot
    pd.plotting.autocorrelation_plot(df[sales_col], ax=axs[5])
    axs[5].set_title('Autocorrelation Plot')

    plt.tight_layout()

    # Convert plots to base64 for displaying on the web page
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return plot_url, future_df

# Flask route for rendering the results page
@app.route('/select_columns', methods=['GET', 'POST'])
def select_columns():
    df = pd.read_csv('uploaded_data.csv')  # Loading the saved CSV file
    columns = df.columns.tolist()

    if request.method == 'POST':
        date_col = request.form['date_col']
        sales_col = request.form['sales_col']
        periods = int(request.form['months_to_predict'])

        # Convert the date column to datetime and set it as index
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index(date_col, inplace=True)

        # Generate plots and forecast data
        plot_url, future_df = generate_plots(df, sales_col, periods)

        # Render the results page with plots and forecast table
        return render_template('plots.html', plot_url=plot_url, future_df=future_df.to_html(classes='data', index=False))

    return render_template('select_columns.html', columns=columns)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df.to_csv('uploaded_data.csv', index=False)
            return redirect(url_for('select_columns'))
    return render_template('upload.html')


def forecast_future(df, model, periods):
    future_dates = pd.date_range(df.index[-1] + DateOffset(months=1), periods=periods, freq='M')
    future_forecast = model.get_forecast(steps=periods).predicted_mean
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Sales': future_forecast
    })
    return future_df


if __name__ == "__main__":
    app.run(debug=True)
