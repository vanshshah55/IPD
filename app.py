from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file."

        if request.form.get('financial_forecasting'):
            return financial_forecasting_arima(df)
        else:
            return "File uploaded successfully, but no analysis was selected."
    except Exception as e:
        return f"An error occurred: {e}"

def financial_forecasting_arima(df):
    try:
        # Ensure there are no duplicate indices
        df = df.reset_index(drop=True)
        
        # Check if 'Revenue' column exists
        if 'Revenue' not in df.columns:
            return "The required 'Revenue' column is missing from the dataset."

        # Preparing the data for ARIMA
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
        df = df.dropna(subset=['Revenue'])

        # Train-Test split
        train_size = int(0.8 * len(df))
        train = df['Revenue'][:train_size]
        test = df['Revenue'][train_size:]

        # Fitting the ARIMA model
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        # Making predictions
        predictions = model_fit.forecast(steps=len(test))

        # Accuracy calculation
        accuracy = 100 - np.mean(np.abs((test.values - predictions) / test.values)) * 100

        # Prepare the results
        results = pd.DataFrame({'Actual Revenue': test.values, 'Predicted Revenue': predictions})
        results['Error'] = results['Actual Revenue'] - results['Predicted Revenue']

        # Plotting the future predictions
        model_full = ARIMA(df['Revenue'], order=(5, 1, 0))
        model_full_fit = model_full.fit()
        future_steps = 12  # Forecasting for 12 months into the future
        future_forecast = model_full_fit.forecast(steps=future_steps)

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Revenue'], label='Historical Revenue')
        plt.plot(range(len(df), len(df) + future_steps), future_forecast, label='Future Forecast', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Revenue')
        plt.title('Revenue Forecast')
        plt.legend()

        # Saving the plot to a string
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('results.html', results=results.to_html(classes='table table-striped'), accuracy=accuracy, plot_url=plot_url)
    except Exception as e:
        return f"An error occurred during financial forecasting: {e}"

if __name__ == '__main__':
    app.run(debug=True)
