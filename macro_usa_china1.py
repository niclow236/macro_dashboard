# IMPORTS
import io
import base64
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table
import webbrowser
import threading

# CONFIGURATION
start_date = '2018-01-01'
country_features = {
    'USA': [
        'GDP_YoY', 'IP_YoY', 'Unemployment', 'Core_CPI_YoY', 'PMI',
        'Yield_Spread', 'Credit_Spread', 'Retail_Sales_MoM', 'Capacity_Utilization',
        'Core_PCE_YoY', 'Services_PMI', 'SmallBiz_Sentiment',
        'Home_Sales', 'Jobless_Claims_4WMA'
    ],
    'China': [
        'GDP_YoY', 'Industrial_Value_Added_YoY', 'Unemployment', 'Core_CPI_YoY', 'PMI',
        'Retail_Sales_YoY', 'Service_Production_Index'
    ]
}

feature_signs = {
    'GDP_YoY': 1,
    'IP_YoY': 1,
    'Industrial_Value_Added_YoY': 1,
    'Unemployment': -1,
    'Core_CPI_YoY': -1,
    'PMI': 1,
    'Yield_Spread': 1,
    'Credit_Spread': -1,
    'Retail_Sales_MoM': 1,
    'Retail_Sales_YoY': 1,
    'Capacity_Utilization': 1,
    'Core_PCE_YoY': -1,
    'Services_PMI': 1,
    'Service_Production_Index': 1,
    'SmallBiz_Sentiment': 1,
    'Home_Sales': 1,
    'Jobless_Claims_4WMA': -1
}

# GLOBALS
raw_data = None
macro_df_model = None
macro_df_model_with_future = None
future_df = None
latest_stage = ""
latest_stage_date = ""
future_stage = ""
future_date = None
table_df = None

# DATA PROCESSING FUNCTION
def process_macro_data(raw_data, features):
    global macro_df_model, macro_df_model_with_future, future_df, latest_stage, latest_stage_date, future_stage, future_date, table_df

    data = {col: raw_data[[col]].dropna() for col in raw_data.columns}
    gdp_df = data['GDP_YoY'].resample('ME').interpolate(method='linear')
    other_data = {k: v.resample('ME').last() for k, v in data.items() if k != 'GDP_YoY'}
    monthly_data = {**other_data, 'GDP_YoY': gdp_df}
    macro_df = pd.concat(monthly_data.values(), axis=1)
    if '10Y' in raw_data.columns and '2Y' in raw_data.columns:
        macro_df['Yield_Spread'] = macro_df['10Y'] - macro_df['2Y']
    if 'HY_OAS' in raw_data.columns:
        macro_df['Credit_Spread'] = macro_df['HY_OAS']
    macro_df.drop(columns=[col for col in ['10Y', '2Y', 'HY_OAS'] if col in macro_df.columns], inplace=True)

    
    for col in raw_data.columns:
        if col in features:
            macro_df[col] = macro_df[col].rolling(window=3, min_periods=1).mean()
    macro_df_model = macro_df.dropna(subset=features).copy()

    # Random Forest for weights
    X = macro_df_model[features]
    if 'Raw_Score' not in macro_df_model:
        macro_df_model['Raw_Score'] = X.mean(axis=1)
    y = pd.qcut(macro_df_model['Raw_Score'], q=4, labels=False)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    importances = pd.Series(rf_model.feature_importances_, index=features)
    normalized_importances = importances / importances.max()
    weights = {f: round(normalized_importances[f] * feature_signs[f], 4) for f in features}

    def compute_macro_score(df):
        def robust_z(x, median, mad):
            return (x - median) / (mad * 1.4826 + 1e-8)

        def logistic(x):
            return 2 / (1 + np.exp(-x)) - 1

        scores = []
        for _, row in df.iterrows():
            score = 0
            for f in features:
                if pd.notna(row[f]):
                    median = macro_df_model[f].median()
                    mad = np.median(np.abs(macro_df_model[f] - median))
                    rz = robust_z(row[f], median, mad)
                    score += weights[f] * logistic(rz)
            scores.append(score)

        df['Raw_Score'] = scores

        # Fit KMeans on historical data to determine regime clusters
        kmeans = KMeans(n_clusters=4, random_state=42)
        historical_scores = macro_df_model[['Raw_Score']].dropna()
        kmeans.fit(historical_scores)

        # Predict cluster labels
        df['Stage_Label'] = kmeans.predict(df[['Raw_Score']])

        # Order cluster labels by average score (lowest = Recession, highest = Expansion)
        centroids = kmeans.cluster_centers_.flatten()
        ordered = np.argsort(centroids)
        label_map = {old: new for new, old in enumerate(ordered)}
        df['Stage_Label'] = df['Stage_Label'].map(label_map)
        df['Stage'] = df['Stage_Label'].map({0: 'Recession', 1: 'Slowdown', 2: 'Recovery', 3: 'Expansion'})
        return df

    macro_df_model = compute_macro_score(macro_df_model)
    latest_date = macro_df_model.index.max()
    future_date = (latest_date + pd.DateOffset(months=1)).replace(day=1)

    # Forecast
    future_features = {}
    for f in features:
        df_feat = macro_df_model[[f]].dropna().copy()
        df_feat['t'] = np.arange(len(df_feat)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(df_feat[['t']], df_feat[f])
        pred = model.predict(pd.DataFrame({'t': [len(df_feat)]}))[0]
        future_features[f] = pred

    future_df = pd.DataFrame([future_features], index=[future_date])
    future_df = compute_macro_score(future_df)
    future_stage = future_df['Stage'].values[0]

    latest_valid = macro_df_model.dropna(subset=['Stage']).iloc[-1]
    latest_stage = latest_valid['Stage']
    latest_stage_date = latest_valid.name.strftime('%Y-%m')

    macro_df_model_with_future = pd.concat([macro_df_model, future_df])
    
    # Table setup
    table_df = macro_df_model_with_future[['Stage']].reset_index()
    table_df.columns = ['Date', 'Economic Stage']
    table_df = table_df.sort_values('Date')
    last_12_hist = table_df[table_df['Date'] < future_date].tail(12)
    predicted_row = table_df[table_df['Date'] == future_date]
    table_df = pd.concat([last_12_hist, predicted_row])
    table_df['Date_str'] = table_df['Date'].dt.strftime('%Y-%m')

# APP SETUP
app = Dash(__name__)
app.title = "Economic Stage Dashboard"

upload_success = False

app.layout = html.Div(
    style={'fontFamily': 'Segoe UI, Helvetica, Arial, sans-serif'},
    children=[
        html.H1(id='dashboard-title', style={'textAlign': 'center', 'paddingTop': '20px'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['📤', html.A('Select Excel file with data')]),
            style={
                'width': '60%', 'margin': 'auto', 'padding': '10px',
                'textAlign': 'center', 'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '10px', 'backgroundColor': '#f9f9f9', 'cursor': 'pointer'
            },
            multiple=False
        ),
        html.Div(id='upload-status', style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
        dcc.Store(id='upload-confirm', data=False),
        dcc.Store(id='uploaded-data-store'),
        html.Div([
            html.Label("Select Country:", style={'fontWeight': 'bold', 'textAlign': 'center', 'display': 'block', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='country-selector',
                options=[
                    {'label': 'USA', 'value': 'USA'},
                    {'label': 'China', 'value': 'China'}
                ],
                value='USA',
                style={'width': '30%', 'margin': 'auto', 'marginBottom': '20px'}
            )
        ]),
        html.Div([
            html.Label("Select Feature to Visualize:", style={'fontWeight': 'bold', 'textAlign': 'center', 'display': 'block', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='feature-selector',
                options=[],  # Dynamically filled based on selected country
                style={'width': '40%', 'margin': 'auto', 'marginBottom': '20px'}
            )
        ], style={'paddingBottom': '20px'}),
        html.Div(id='summary-cards-container'),
        html.Div([
            dcc.Graph(id='feature-graph'),
            dcc.Graph(id='stage-timeline'),
            dcc.Graph(id='score-graph')
        ], style={'padding': '0 5%'}),
        html.H4("Stage Classification - Last 12 Months + Prediction", style={'textAlign': 'center', 'marginTop': '30px'}),
        html.Div([
            dash_table.DataTable(
                id='stage-table',
                columns=[
                    {"name": "Date", "id": "Date_str"},
                    {"name": "Economic Stage", "id": "Economic Stage"}
                ],
                style_table={'width': '60%', 'margin': 'auto'},
                style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': '14px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'}
            )
        ])
    ]
)

@app.callback(
    Output('feature-selector', 'options'),
    Input('country-selector', 'value')
)
def update_feature_dropdown(country):
    return [{'label': f, 'value': f} for f in country_features[country]]

@app.callback(
    Output('feature-selector', 'value'),
    Input('country-selector', 'value')
)
def reset_feature_selector(country):
    return country_features[country][0]

@app.callback(
    [Output('upload-status', 'children'),
     Output('uploaded-data-store', 'data'),
     Output('upload-confirm', 'data')],
    Input('upload-data', 'contents')
)
def load_uploaded_excel(contents):
    if contents is None:
        return "📂 No file uploaded yet.", {}, False
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    buffer = io.BytesIO(decoded)
    xl = pd.read_excel(buffer, sheet_name=None, index_col=0, parse_dates=True)
    store = {k: v.to_json(date_format='iso') for k, v in xl.items()}
    return f"✅ Successfully uploaded file with sheets: {', '.join(store.keys())}", store, True

@app.callback(
    [Output('feature-graph', 'figure'),
     Output('stage-timeline', 'figure'),
     Output('score-graph', 'figure'),
     Output('stage-table', 'data'),
     Output('summary-cards-container', 'children')],
    Output('stage-table', 'style_data_conditional'),
    [Input('feature-selector', 'value'),
     Input('country-selector', 'value'),
     Input('uploaded-data-store', 'data'),
     Input('upload-confirm', 'data')],
    prevent_initial_call=False
)

def update_graphs(selected_feature, country, uploaded_data, confirm):
    if not confirm or uploaded_data is None or country not in uploaded_data:
        return {}, {}, {}, [], html.Div(), []

    df = pd.read_json(io.StringIO(uploaded_data[country]), convert_dates=True)
    process_macro_data(df, country_features[country])
        
    historical_only = macro_df_model_with_future[macro_df_model_with_future.index < future_date]
    fig_feature = px.line(historical_only, x=historical_only.index, y=selected_feature, title=f"{selected_feature} Over Time")
    fig_feature.update_layout(xaxis_title="Date")

    stage_df = macro_df_model_with_future[['Stage_Label']].reset_index()
    stage_df.columns = ['Date', 'Stage_Label']
    predicted_point = stage_df[stage_df['Date'] == future_date]
    historical_stage_df = stage_df[stage_df['Date'] < future_date]

    fig_timeline = px.line(
        historical_stage_df, x='Date', y='Stage_Label', title='Economic Stage Timeline',
        markers=True, labels={'Stage_Label': 'Stage'}
    )
    fig_timeline.update_layout(xaxis_title="Date", yaxis_title="Stage")
    fig_timeline.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=['Recession', 'Slowdown', 'Recovery', 'Expansion'])

    if not historical_stage_df.empty and not predicted_point.empty:
        fig_timeline.add_scatter(
            x=[historical_stage_df.iloc[-1]['Date'], predicted_point.iloc[0]['Date']],
            y=[historical_stage_df.iloc[-1]['Stage_Label'], predicted_point.iloc[0]['Stage_Label']],
            mode='lines+markers+text',
            text=['Current', 'Predicted'],
            textposition=['top center', 'bottom center'],
            marker=dict(size=10, color=['#636EFA', 'red']),
            line=dict(dash='solid', color='red'), name='Predicted Score'
        )

    fig_score = px.line(
        macro_df_model_with_future, x=macro_df_model_with_future.index, y='Raw_Score', title='Raw Score Over Time'
    )
    fig_score.update_layout(xaxis_title="Date")
    fig_score.add_scatter(
        x=[future_date], y=[future_df['Raw_Score'].values[0]],
        mode='markers+text',
        marker=dict(color='red', size=10),
        text=["Predicted"], textposition='top center', name='Predicted Score'
    )

    summary_cards = html.Div([
        html.Div([
            html.Div([
                html.H4(f"Current Stage (as of {latest_stage_date})", style={'marginBottom': '5px', 'marginTop': '45px', 'color': '#fff'}),
                html.H3(latest_stage, style={'margin': '0', 'color': '#fff'})
            ], style={
                'backgroundColor': (
                    '#d32f2f' if latest_stage == 'Recession' else
                    '#FFA500' if latest_stage == 'Slowdown' else
                    '#388e3c' if latest_stage == 'Recovery' else
                    '#1976d2'
                ),
                'padding': '15px 25px', 'borderRadius': '10px', 'textAlign': 'center',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginRight': '20px'
            }),
            html.Div([
                html.H4(f"Predicted Stage for {future_date.strftime('%Y-%m')}", style={'marginBottom': '5px', 'marginTop': '45px', 'color': '#fff'}),
                html.H3(future_stage, style={'margin': '0', 'color': '#fff'})
            ], style={
                'backgroundColor': (
                    '#d32f2f' if future_stage == 'Recession' else
                    '#FFA500' if future_stage == 'Slowdown' else
                    '#388e3c' if future_stage == 'Recovery' else
                    '#1976d2'
                ),
                'padding': '15px 25px', 'borderRadius': '10px', 'textAlign': 'center',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
            }),
            html.Div([
                    html.H5("Legend", style={'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Div("Recession", style={'backgroundColor': '#d32f2f', 'color': 'white', 'padding': '5px 10px', 'marginBottom': '5px'}),
                    html.Div("Slowdown", style={'backgroundColor': '#FFA500', 'color': 'black', 'padding': '5px 10px', 'marginBottom': '5px'}),
                    html.Div("Recovery", style={'backgroundColor': '#388e3c', 'color': 'white', 'padding': '5px 10px', 'marginBottom': '5px'}),
                    html.Div("Expansion", style={'backgroundColor': '#1976d2', 'color': 'white', 'padding': '5px 10px'})
                ], style={
                    'paddingLeft': '30px',
                    'fontFamily': 'Arial',
                    'fontSize': '14px',
                    'alignSelf': 'center'
                })
        ], style={'display': 'flex', 'justifyContent': 'center', 'paddingTop': '20px'})
    ])

    return fig_feature, fig_timeline, fig_score, table_df.to_dict('records'), summary_cards, [
        {
            'if': {'filter_query': f'{{Date_str}} = "{future_date.strftime("%Y-%m")}"'},
            'color': 'red',
            'fontWeight': 'bold'
        }
    ]

@app.callback(
    Output('dashboard-title', 'children'),
    Input('country-selector', 'value')
)
def update_title(country):
    return f"{country} Macroeconomic Dashboard"

if __name__ == '__main__':
    def open_browser():
        try:
            webbrowser.get(using='windows-default').open_new("http://127.0.0.1:8050/")
        except:
            webbrowser.open_new("http://127.0.0.1:8050/")
    threading.Timer(1, open_browser).start()
    app.run(debug=False, use_reloader=False)
server = app.server

