import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc

# Initialize the Dash app with Bootstrap for better styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Universal Financial Dashboard"

# Layout of the Dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Universal Financial Dashboard", className="text-center my-4 text-primary"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a CSV or XLSX File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-bottom': '20px',
                    'backgroundColor': '#f8f9fa',
                    'color': '#343a40'
                },
                multiple=False
            ),
            html.Div(id='upload-status')
        ], width=12)
    ]),
    
    # Hidden div to store the uploaded data
    dcc.Store(id='stored-data'),
    
    # Dynamic inputs for selecting columns after data upload
    dbc.Row([
        dbc.Col([
            html.Label("Select Date Column:", className="font-weight-bold"),
            dcc.Dropdown(id='date-column', placeholder="Select Date Column", className="mb-2")
        ], width=4),
        dbc.Col([
            html.Label("Select Category Column:", className="font-weight-bold"),
            dcc.Dropdown(id='category-column', placeholder="Select Category Column", className="mb-2")
        ], width=4),
        dbc.Col([
            html.Label("Select Value Column:", className="font-weight-bold"),
            dcc.Dropdown(id='value-column', placeholder="Select Value Column", className="mb-2")
        ], width=4),
    ], className="mb-4"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Select Date Range:", className="font-weight-bold"),
            dcc.DatePickerRange(
                id='date-range',
                start_date_placeholder_text="Start Period",
                end_date_placeholder_text="End Period",
                display_format='YYYY-MM-DD',
                className="mb-2"
            )
        ], width=6),
        dbc.Col([
            html.Label("Select Categories:", className="font-weight-bold"),
            dcc.Dropdown(id='category-filter', multi=True, placeholder="Filter by Categories", className="mb-2")
        ], width=6),
    ], className="mb-4"),
    
    # Slider and Radio Items for additional control
    dbc.Row([
        dbc.Col([
            html.Label("Select Maximum Value:", className="font-weight-bold"),
            dcc.Slider(id='value-slider', min=0, max=100, step=1, value=50,
                       marks={i: str(i) for i in range(0, 101, 10)}, className="mb-4")
        ], width=6),
        dbc.Col([
            html.Label("Select Chart Type:", className="font-weight-bold"),
            dcc.RadioItems(
                id='chart-type-radio',
                options=[
                    {'label': 'Line Chart', 'value': 'line'},
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Bubble Chart', 'value': 'bubble'}
                ],
                value='line',
                inline=True,
                className="mb-4"
            )
        ], width=6),
    ], className="mb-4"),
    
    # Visualizations
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-chart')
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='bar-chart')
        ], width=6),
        dbc.Col([
            dcc.Graph(id='pie-chart')
        ], width=6),
    ]),
    
    # Data Table and Additional Metrics
    dbc.Row([
        dbc.Col([
            html.H4("Data Table", className="text-primary"),
            dash_table.DataTable(
                id='data-table',
                columns=[],
                data=[],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px'
                },
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_current=0,
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='additional-metrics', className='mt-4')
        ], width=12)
    ])
    
], fluid=True)

# Callback to parse uploaded data and store it
@app.callback(
    Output('stored-data', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_upload(contents, filename):
    if contents is None:
        return dash.no_update, ""
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            # Handle CSV files
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.xlsx'):
            # Handle Excel files
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return dash.no_update, dbc.Alert("Unsupported file type. Please upload a CSV or XLSX file.", color="danger")
    except Exception as e:
        return dash.no_update, dbc.Alert(f"Error processing file: {e}", color="danger")
    
    return df.to_json(date_format='iso', orient='split'), dbc.Alert(f"Successfully uploaded {filename}", color="success")

# Callback to populate dropdowns based on uploaded data
@app.callback(
    Output('date-column', 'options'),
    Output('category-column', 'options'),
    Output('value-column', 'options'),
    Input('stored-data', 'data')
)
def update_dropdowns(jsonified_data):
    if jsonified_data is None:
        return [], [], []
    df = pd.read_json(jsonified_data, orient='split')
    options = [{'label': col, 'value': col} for col in df.columns]
    return options, options, options

# Callback to populate category filter based on selected category column
@app.callback(
    Output('category-filter', 'options'),
    Input('category-column', 'value'),
    State('stored-data', 'data')
)
def update_category_filter(category_col, jsonified_data):
    if jsonified_data is None or category_col is None:
        return []
    df = pd.read_json(jsonified_data, orient='split')
    categories = df[category_col].unique()
    return [{'label': cat, 'value': cat} for cat in categories]

# Callback to update the date range picker based on selected date column
@app.callback(
    Output('date-range', 'min_date_allowed'),
    Output('date-range', 'max_date_allowed'),
    Output('date-range', 'start_date'),
    Output('date-range', 'end_date'),
    Input('date-column', 'value'),
    State('stored-data', 'data')
)
def update_date_picker(date_col, jsonified_data):
    if jsonified_data is None or date_col is None:
        return None, None, None, None
    df = pd.read_json(jsonified_data, orient='split')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    return min_date, max_date, min_date, max_date

# Main callback to update all visualizations and table
@app.callback(
    Output('line-chart', 'figure'),
    Output('bar-chart', 'figure'),
    Output('pie-chart', 'figure'),
    Output('data-table', 'columns'),
    Output('data-table', 'data'),
    Output('additional-metrics', 'children'),
    Input('stored-data', 'data'),
    Input('date-column', 'value'),
    Input('category-column', 'value'),
    Input('value-column', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('category-filter', 'value'),
    Input('value-slider', 'value'),
    Input('chart-type-radio', 'value')
)
def update_visuals(jsonified_data, date_col, category_col, value_col, start_date, end_date, selected_categories, value_slider, chart_type):
    if not jsonified_data or not date_col or not category_col or not value_col:
        empty_fig = px.scatter(title="No Data Available")
        return empty_fig, empty_fig, empty_fig, [], [], "", 
    
    df = pd.read_json(jsonified_data, orient='split')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    df_filtered = df.loc[mask]
    
    if selected_categories:
        df_filtered = df_filtered[df_filtered[category_col].isin(selected_categories)]
    
    # Filter based on slider value
    df_filtered = df_filtered[df_filtered[value_col] <= value_slider]

    # Prepare the data for the various visualizations
    df_time = df_filtered.groupby(date_col)[value_col].sum().reset_index()
    df_category = df_filtered.groupby(category_col)[value_col].sum().reset_index()
    df_pie = df_filtered.groupby(category_col)[value_col].sum().reset_index()
    
    # Initialize figures
    line_fig = px.line(df_time, x=date_col, y=value_col, title=f"Trend of {value_col} Over Time")
    bar_fig = px.bar(df_category, x=category_col, y=value_col, title=f"Total {value_col} by {category_col}")
    pie_fig = px.pie(df_pie, names=category_col, values=value_col, title=f"Distribution of {value_col} by {category_col}")
    
    # Generate figures based on selected chart type
    if chart_type == 'line':
        return line_fig, bar_fig, pie_fig, generate_table(df_filtered), df_filtered.to_dict('records'), generate_additional_metrics(df_filtered, value_col)
    elif chart_type == 'bar':
        return px.line(df_time, x=date_col, y=value_col, title=f"Trend of {value_col} Over Time"), bar_fig, pie_fig, generate_table(df_filtered), df_filtered.to_dict('records'), generate_additional_metrics(df_filtered, value_col)
    elif chart_type == 'pie':
        return px.line(df_time, x=date_col, y=value_col, title=f"Trend of {value_col} Over Time"), px.bar(df_category, x=category_col, y=value_col, title=f"Total {value_col} by {category_col}"), pie_fig, generate_table(df_filtered), df_filtered.to_dict('records'), generate_additional_metrics(df_filtered, value_col)
    elif chart_type == 'bubble':
        bubble_fig = px.scatter(df_filtered, x=date_col, y=value_col, size=value_col, color=category_col, title=f"Bubble Chart of {value_col} by {category_col}")
        return bubble_fig, bar_fig, pie_fig, generate_table(df_filtered), df_filtered.to_dict('records'), generate_additional_metrics(df_filtered, value_col)
    
    return px.scatter(title="No Data Available"), px.scatter(title="No Data Available"), px.scatter(title="No Data Available"), [], [], generate_additional_metrics(df_filtered, value_col)

def generate_table(df_filtered):
    return [{"name": col, "id": col} for col in df_filtered.columns]

def generate_additional_metrics(df_filtered, value_col):
    total_value = df_filtered[value_col].sum()
    avg_value = df_filtered[value_col].mean()
    min_value = df_filtered[value_col].min()
    max_value = df_filtered[value_col].max()
    count_value = df_filtered[value_col].count()

    additional_metrics = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Total"),
            dbc.CardBody(html.H5(f"{total_value:,.2f}", className="card-title"))
        ], color="primary", inverse=True)),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Average"),
            dbc.CardBody(html.H5(f"{avg_value:,.2f}", className="card-title"))
        ], color="success", inverse=True)),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Minimum"),
            dbc.CardBody(html.H5(f"{min_value:,.2f}", className="card-title"))
        ], color="danger", inverse=True)),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Maximum"),
            dbc.CardBody(html.H5(f"{max_value:,.2f}", className="card-title"))
        ], color="warning", inverse=True)),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Count"),
            dbc.CardBody(html.H5(f"{count_value}", className="card-title"))
        ], color="info", inverse=True)),
    ], className="mt-4")
    
    return additional_metrics

if __name__ == '__main__':
    app.run_server(debug=True)
