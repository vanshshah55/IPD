from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from PyPDF2 import PdfReader
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to process Excel files
def process_excel(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to process PDF files
def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    # For simplicity, we'll just return the text here
    # Further processing could be done depending on the PDF structure
    return pd.DataFrame({'PDF Content': [text]})

# Function to process CSV files
def process_csv(file_path):
    df = pd.read_csv(file_path)
    return df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the file based on its extension
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = process_excel(file_path)
            elif filename.endswith('.pdf'):
                df = process_pdf(file_path)
            elif filename.endswith('.csv'):
                df = process_csv(file_path)
            else:
                return 'Unsupported file format'

            # For demonstration, return the first few rows of the DataFrame
            return df.head().to_html()

    return '''
    <!doctype html>
    <title>Upload File</title>
    <h1>Upload Excel, CSV, or PDF File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
