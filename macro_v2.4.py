# IMPORTS
import os
import io
import base64
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table, State
import webbrowser
import threading

# CONFIGURATION
start_date = '2018-01-01'

country_features = {
    'USA': [
        'GDP_YoY', 'IP_YoY', 'Unemployment', 'Core_CPI_YoY', 'PPI_YoY', 'Manufacturing_PMI',
        'Yield_Spread', 'Credit_Spread', 'Retail_Sales_MoM', 'Capacity_Utilization',
        'Core_PCE_YoY', 'Services_PMI', 'SmallBiz_Sentiment',
        'Home_Sales', 'Jobless_Claims_4WMA'
    ],
    'China': [
        'GDP_YoY', 'Industrial_Value_Added_YoY', 'Unemployment', 'Core_CPI_YoY', 'Manufacturing_PMI',
        'Retail_Sales_YoY', 'Service_Production_Index', 'PPI_YoY', 'Exports_YoY',
        'Money_Supply_M2_YoY', 'SHIBOR_1M', 'Real_Estate_Climate_Index_YoY'
    ],
    'Japan': [
        'GDP_YoY', 'Unemployment', 'Core_CPI_YoY', 'PPI_YoY', 'Retail_Sales_YoY', 'Exports_YoY',
        'Yield_Spread', 'Money_Supply_M2_YoY', 'Tankan_Business_Conditions_LE_Mfg',
        'Industrial_Production_YoY', 'Consumer_Confidence', 'Business_Confidence_All_Industry'
    ]
}

feature_signs = {
    'GDP_YoY': 1,
    'IP_YoY': 1,
    'Industrial_Value_Added_YoY': 1,
    'Unemployment': -1,
    'Core_CPI_YoY': -1,
    'PPI_YoY': -1,
    'Manufacturing_PMI': 1,
    'Yield_Spread': 1,
    'Credit_Spread': -1,
    'Retail_Sales_MoM': 1,
    'Retail_Sales_YoY': 1,
    'Exports_YoY': 1,
    'Capacity_Utilization': 1,
    'Core_PCE_YoY': -1,
    'Services_PMI': 1,
    'Service_Production_Index': 1,
    'SmallBiz_Sentiment': 1,
    'Home_Sales': 1,
    'Jobless_Claims_4WMA': -1,
    'Real_Estate_Climate_Index_YoY': 1,
    'Money_Supply_M2_YoY': 1,
    'SHIBOR_1M': -1,
    'Tankan_Business_Conditions_LE_Mfg': 1,
    'Industrial_Production_YoY': 1,
    'Consumer_Confidence': 1,
    'Business_Confidence_All_Industry': 1
}

stage_thresholds = {
    'USA': [-1.25, -0.75, -0.25],   
    'China': [-1.25, -0.25, 0.75],
    'Japan': [-1.0, 0, 1.0]
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
forecast_conf_intervals = None

# DATA PROCESSING FUNCTION
def process_macro_data(raw_data, features, country, n_months=1, ci_alpha=0.05):
    global macro_df_model, macro_df_model_with_future, future_df
    global latest_stage, latest_stage_date, future_stage
    global future_date, forecast_conf_intervals

    # --- Data Initialization ---
    forecast_conf_intervals = pd.DataFrame(columns=['Date', 'Score', 'Lower', 'Upper', 'Lower_Stage', 'Upper_Stage'])

    # Preprocess raw data, interpolate GDP, and resample to monthly frequency, calculating yield and credit spreads
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

    # 3M Rolling Average for smoothing
    for col in raw_data.columns:
        if col in features:
            macro_df[col] = macro_df[col].rolling(window=3, min_periods=1).mean()

    min_required = int(len(features) * 0.6)
    macro_df_model = macro_df[macro_df[features].notna().sum(axis=1) >= min_required].copy()

    # --- Feature Engineering for indicator weights ---
    # Prepare data
    X = macro_df_model[features].dropna()
    macro_df_model = macro_df_model.loc[X.index]
    y = pd.qcut(X.mean(axis=1), q=4, labels=False)

    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X, y)

    ridge_model = RidgeClassifier(alpha=1.0, random_state=42)
    ridge_model.fit(X, y)

    # Logistic regression for feature importance
    logit = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)
    logit.fit(X, y)

    # Permutation importance on RF
    perm = permutation_importance(rf_model, X, y, n_repeats=10, random_state=42)
    perm_imp = pd.Series(perm.importances_mean, index=features)

    # Normalize all importances
    rf_imp = pd.Series(rf_model.feature_importances_, index=features)
    gb_imp = pd.Series(gb_model.feature_importances_, index=features)
    ridge_imp = pd.Series(np.abs(ridge_model.coef_[0]), index=features)
    logit_imp = pd.Series(np.abs(logit.coef_[0]), index=features)

    imp_df = pd.DataFrame({
        'RF': rf_imp / rf_imp.max(),
        'GB': gb_imp / gb_imp.max(),
        'Ridge': ridge_imp / ridge_imp.max(),
        'Perm': perm_imp / perm_imp.max(),
        'Logit': logit_imp / logit_imp.max()
    })

    # Ensemble average
    avg_imp = imp_df.mean(axis=1)
    importances = avg_imp.sort_values(ascending=False)

    # Final directional weights
    weights = {f: round(avg_imp[f] * feature_signs[f], 4) for f in features}

    # Helper function to compute macroeconomic score and classify stage
    def compute_macro_score(df, country):
        # apply logistic transformation and robust z-score to normalize features to ensure comparability
        def robust_z(x, median, mad):
            return (x - median) / (mad * 1.4826 + 1e-8)

        def logistic(x):
            return 2 / (1 + np.exp(-x)) - 1

        # Calculate raw scores
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

        # Classify stages based on thresholds depending on country
        th = stage_thresholds[country]
        def classify_stage(s):
            if s < th[0]: return 0
            elif s < th[1]: return 1
            elif s < th[2]: return 2
            else: return 3
        df['Stage_Label'] = df['Raw_Score'].apply(classify_stage)
        df['Stage'] = df['Stage_Label'].map({0: 'Recession', 1: 'Slowdown', 2: 'Recovery', 3: 'Expansion'})
        return df

    # Create the econommic scoring model dataframe
    macro_df_model = compute_macro_score(macro_df_model, country)

    # --- Forecasting of Economic Stage (Future) ---
    # Set the latest historical date and the target forecast date
    latest_date = macro_df_model.index.max()
    future_date = (latest_date + pd.DateOffset(months=1)).replace(day=1)

    # Use recent 36 months
    var_df = macro_df_model[features].dropna()
    if len(var_df) > 36:
        var_df = var_df.tail(36)

    # Standardize (Z-score)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(var_df)
    scaled_df = pd.DataFrame(scaled_values, index=var_df.index, columns=features)

    # Fit VAR on standardized data
    model = VAR(scaled_df)
    max_lags_possible = int(len(scaled_df) / (len(features) + 1)) - 1
    max_lags_to_use = min(max_lags_possible, 12)
    if max_lags_to_use < 1:
        raise ValueError("Not enough data points for VAR forecasting.")
    results = model.fit(maxlags=max_lags_to_use, ic='aic')

    # Forecast N months ahead
    lag_order = results.k_ar
    forecast_input = scaled_df.values[-lag_order:]
    forecast_scaled_all, lower_bounds_scaled, upper_bounds_scaled = results.forecast_interval(
        y=forecast_input, steps=n_months, alpha=ci_alpha or 0.05
    )
    forecast_unscaled_all = scaler.inverse_transform(forecast_scaled_all)
    lower_unscaled = scaler.inverse_transform(lower_bounds_scaled)
    upper_unscaled = scaler.inverse_transform(upper_bounds_scaled)

    future_rows = []
    for i in range(n_months):
        forecast_month = (latest_date + pd.DateOffset(months=i+1)).replace(day=1)
        raw_forecast = dict(zip(features, forecast_unscaled_all[i]))
        latest_actual = macro_df_model.iloc[-1][features]
        blended = {
            f: 0.2 * latest_actual[f] + 0.8 * raw_forecast[f] for f in features
        }
        row = pd.DataFrame([blended], index=[forecast_month])
        scored = compute_macro_score(row, country)

        # Blend lower and upper bounds with latest actual 
        blended_lower = {
            f: 0.2 * latest_actual[f] + 0.8 * lower_unscaled[i][j] for j, f in enumerate(features)
        }
        blended_upper = {
            f: 0.2 * latest_actual[f] + 0.8 * upper_unscaled[i][j] for j, f in enumerate(features)
        }

        # Score blended lower bound
        lower_row = pd.DataFrame([blended_lower], index=[forecast_month])
        lower_scored = compute_macro_score(lower_row, country)
        raw_lower = lower_scored['Raw_Score'].values[0]
        lower_stage = lower_scored['Stage'].values[0]

        # Score blended upper bound
        upper_row = pd.DataFrame([blended_upper], index=[forecast_month])
        upper_scored = compute_macro_score(upper_row, country)
        raw_upper = upper_scored['Raw_Score'].values[0]
        upper_stage = upper_scored['Stage'].values[0]

        # Match stages with correct scores before sorting
        score_stage_pairs = sorted(
            [(raw_lower, lower_stage), (raw_upper, upper_stage)],
            key=lambda x: x[0]
        )
        (lower_score, lower_stage), (upper_score, upper_stage) = score_stage_pairs

        # Append to CI DataFrame
        forecast_conf_intervals.loc[len(forecast_conf_intervals)] = {
            'Date': forecast_month,
            'Score': scored['Raw_Score'].values[0],
            'Lower': lower_score,
            'Upper': upper_score,
            'Lower_Stage': lower_stage,
            'Upper_Stage': upper_stage
        }

        # Rolling quantiles for future point classification
        recent_window = macro_df_model.tail(36) if len(macro_df_model) > 36 else macro_df_model
        p25, p50, p75 = recent_window['Raw_Score'].quantile([0.25, 0.5, 0.75])
        scored['Stage_Label'] = scored['Raw_Score'].apply(
            lambda s: 0 if s < p25 else 1 if s < p50 else 2 if s < p75 else 3
        )
        scored['Stage'] = scored['Stage_Label'].map({
            0: 'Recession', 1: 'Slowdown', 2: 'Recovery', 3: 'Expansion'
        })
        future_rows.append(scored)

    # Classify future stages based on thresholds
    future_df = pd.concat(future_rows)
    future_stage = future_df['Stage'].iloc[-1]
    th = stage_thresholds[country]
    def classify_future_stage(s):
        if s < th[0]: return 0
        elif s < th[1]: return 1
        elif s < th[2]: return 2
        else: return 3
    future_df['Stage_Label'] = future_df['Raw_Score'].apply(classify_future_stage)
    future_df['Stage'] = future_df['Stage_Label'].map({0: 'Recession', 1: 'Slowdown', 2: 'Recovery', 3: 'Expansion'})
    future_stage = future_df['Stage'].values[0]

    # Identify latest valid historical stage
    latest_valid = macro_df_model.dropna(subset=['Stage']).iloc[-1]
    latest_stage = latest_valid['Stage']
    latest_stage_date = latest_valid.name.strftime('%Y-%m')

    # Concatenate into final model with future row
    macro_df_model_with_future = pd.concat([macro_df_model, future_df])

    return importances.sort_values(ascending=False)

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
            children=html.Div(['📤 Drag and drop or ', html.A('select a macro Excel file')]),
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
        dcc.Store(id='processed-data'),
        html.Div([
            html.Label("Select Country:", style={'fontWeight': 'bold', 'textAlign': 'center', 'display': 'block', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='country-selector',
                options=[
                    {'label': 'USA', 'value': 'USA'},
                    {'label': 'China', 'value': 'China'},
                    {'label': 'Japan', 'value': 'Japan'}
                ],
                value='USA',
                style={'width': '30%', 'margin': 'auto', 'marginBottom': '20px'}
            )
        ]),
        html.Div(id='summary-cards-container'),
        html.Div([
            html.Label("Forecast Period (months):", style={
                'fontWeight': 'bold', 'textAlign': 'center', 'display': 'block',
                'marginBottom': '10px'
            }),
            dcc.Slider(
                id='forecast-horizon-slider',
                min=1,
                max=24,
                step=1,
                value=1,
                marks={i: f'{i}M' for i in range(1, 25) if i % 3 == 0},
                tooltip={'always_visible': True},
                updatemode='drag', 
                included=False
            )
        ], style={
            'width': '50%',
            'margin': 'auto',
            'marginBottom': '30px',
            'padding': '20px',
            'backgroundColor': '#f7f9fc'
        }),
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='scenario-toggle',
                    options=[
                        {'label': 'Baseline Prediction', 'value': 'baseline'},
                        {'label': 'Optimistic (Upper Bound)', 'value': 'optimistic'},
                        {'label': 'Pessimistic (Lower Bound)', 'value': 'pessimistic'}
                    ],
                    value='baseline',
                    labelStyle={'display': 'inline-block', 'marginRight': '15px'},
                    style={'textAlign': 'center', 'marginBottom': '8px', 'marginTop': '8px'}
                ),
                html.Div([
                    html.Label("Confidence Interval Level", style={'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '30px', 'marginBottom': '15px', 'display': 'block'}),
                    dcc.Slider(id='ci-slider', min=0, max=5, step=None,
                        marks={
                            0: {'label': 'OFF'},
                            1: {'label': '80%'},
                            2: {'label': '90%'},
                            3: {'label': '95%'},
                            4: {'label': '97.5%'},
                            5: {'label': '99%'}
                        },
                        value=3, included=False, tooltip={'always_visible': True}
                    )
                ], style={'textAlign': 'center', 'marginBottom': '20px'})
            ]),
            dcc.Graph(id='score-graph', style={'width': '100%', 'marginBottom': '0px'}),
            html.Div([
                html.H5("Economic Stage Legend", style={'fontWeight': 'bold', 'display': 'block', 'margin': '0 0 10px 0', 'fontSize': '16px'}),
                html.Span("Recession", style={'backgroundColor': 'rgba(211, 47, 47, 0.45)', 'color': '#d32f2f', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #d32f2f', 'borderRadius': '5px'}),
                html.Span("Slowdown", style={'backgroundColor': 'rgba(255, 165, 0, 0.45)', 'color': '#FFA500', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #FFA500', 'borderRadius': '5px'}),
                html.Span("Recovery", style={'backgroundColor': 'rgba(56, 142, 60, 0.45)', 'color': '#388e3c', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #388e3c', 'borderRadius': '5px'}),
                html.Span("Expansion", style={'backgroundColor': 'rgba(25, 118, 210, 0.45)', 'color': '#1976d2', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #1976d2', 'borderRadius': '5px'})
            ], style={
                'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '0px', 'marginBottom': '0px', 'paddingTop': '0px'
            }),
            dcc.Graph(id='contribution-chart', style={'marginTop': '20px', 'marginBottom': '100px'}),
            html.Div([
                html.Label("Select Feature to Visualize:", style={'fontWeight': 'bold', 'textAlign': 'center', 'display': 'block', 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='feature-selector',
                    options=[],  
                    style={'width': '40%', 'margin': 'auto', 'marginBottom': '1px'}
                ),
                dcc.Graph(id='feature-graph', style={'marginTop': '0px'})
            ], style={'paddingBottom': '20px'}),
            dcc.Graph(id='importance-graph')
        ], style={'padding': '0 5%'}),
        html.Div([
            html.Label("Filter Stage Table by Date Range:", style={
                'fontWeight': 'bold',
                'display': 'block',
                'textAlign': 'center',
                'marginBottom': '8px'
            }),
            html.Div([
                dcc.DatePickerRange(
                    id='date-range-picker',
                    display_format='YYYY-MM',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    style={
                        'display': 'inline-block',
                        'padding': '10px',
                        'borderRadius': '8px',
                        'backgroundColor': '#ffffff',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)'
                    }
                )
            ], style={'textAlign': 'center'})
        ], style={'marginTop': '30px', 'marginBottom': '30px'}),
        html.H4("Economic Stage Table", style={'textAlign': 'center', 'marginTop': '30px'}),
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
    return f"✅ Uploaded file with sheets: {', '.join(store.keys())}", store, True

@app.callback(
    [Output('feature-graph', 'figure'),
    Output('score-graph', 'figure'),
    Output('importance-graph', 'figure'),
    Output('stage-table', 'data'),
    Output('summary-cards-container', 'children'),
    Output('stage-table', 'style_data_conditional'),
    Output('processed-data', 'data')],
    [Input('feature-selector', 'value'),
     Input('country-selector', 'value'),
     Input('uploaded-data-store', 'data'),
     Input('upload-confirm', 'data'),
     Input('forecast-horizon-slider', 'value'),
     Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input('ci-slider', 'value'),
    Input('scenario-toggle', 'value')],
    prevent_initial_call=False
)
def update_graphs(selected_feature, country, uploaded_data, confirm, forecast_months, start_date, end_date, ci_slider_val, scenario_toggle):
    if not confirm or uploaded_data is None or country not in uploaded_data:
        return {}, {}, {}, [], html.Div(), [], ""
    
    ci_map = {0: None, 1: 0.20, 2: 0.10, 3: 0.05, 4: 0.025, 5: 0.01}
    ci_alpha = ci_map[ci_slider_val]

    df = pd.read_json(io.StringIO(uploaded_data[country]), convert_dates=True)
    feature_importance_series = process_macro_data(df, country_features[country], country, forecast_months, ci_alpha)

    # === Dashboard Table Setup ===
    table_df = macro_df_model_with_future[['Stage']].reset_index()
    table_df.columns = ['Date', 'Economic Stage']
    table_df = table_df.sort_values('Date')

    # Apply user-specified date range filter
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_table_df = table_df[(table_df['Date'] >= start_date) & (table_df['Date'] <= end_date)].copy()
    else:
        hist_only = table_df[table_df['Date'] < future_df.index[0]]
        last_12_hist = hist_only.tail(12)
        predicted_row = future_df[['Stage']].reset_index()
        predicted_row.columns = ['Date', 'Economic Stage']
        filtered_table_df = pd.concat([last_12_hist, predicted_row])

    filtered_table_df['Date_str'] = filtered_table_df['Date'].dt.strftime('%Y-%m')
        
    historical_only = macro_df_model_with_future[macro_df_model_with_future.index < future_date]
    fig_feature = px.line(historical_only, x=historical_only.index, y=selected_feature, title=f"{selected_feature} Over Time")
    fig_feature.update_layout(xaxis_title="Date")
    fig_feature.update_traces(hovertemplate="Date: %{x|%Y %b}<br>" + selected_feature + ": %{y}<extra></extra>")

    stage_df = macro_df_model_with_future[['Stage_Label']].reset_index()
    stage_df.columns = ['Date', 'Stage_Label']

    # Identify last historical point (current)
    last_hist_date = macro_df_model.index[-1]

    # Create the feature importance bar chart
    fig_importance = px.bar(
        feature_importance_series,
        x=feature_importance_series.values,
        y=feature_importance_series.index,
        orientation='h',
        title='Feature Importance (Random Forest)',
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig_importance.update_layout(
        yaxis_title="Feature",
        xaxis_title="Importance",
        yaxis={'categoryorder': 'total ascending'}
    )
    fig_importance.update_traces(
        hovertemplate="Feature: %{y}<br>Importance: %{x}<extra></extra>"
    )

    # Raw Score Line Plot with hover on Stage
    fig_score = px.line(
        macro_df_model,
        x=macro_df_model.index,
        y='Raw_Score',
        title='Economic Stage Over Time',
        custom_data=['Stage']
    )
    fig_score.update_traces(
        hovertemplate='Date: %{x|%Y %b}<br>Raw Score: %{y}<br>Stage: %{customdata[0]}<extra></extra>'
    )

    fig_score.update_layout(
        yaxis_title="Raw Score",
        xaxis_title="Date",
        width=None,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        margin=dict(b=10)
    )

    last_hist_score = macro_df_model['Raw_Score'].iloc[-1]

    # Current (last historical) point
    fig_score.add_trace(go.Scatter(
        x=[last_hist_date],
        y=[last_hist_score],
        mode='markers',
        marker=dict(color='#636EFA', size=6),
        name='Current Score',
        customdata=[macro_df_model['Stage'].iloc[-1]],
        hovertemplate='Date: %{x|%Y %b}<br>Raw Score: %{y}<br>Stage: %{customdata}<extra></extra>'
    ))

    # Select scenario forecast
    if scenario_toggle == 'baseline':
        y_forecast = forecast_conf_intervals['Score']
    elif scenario_toggle == 'optimistic':
        y_forecast = forecast_conf_intervals['Upper']
    else: 
        y_forecast = forecast_conf_intervals['Lower']

    fig_score.add_trace(go.Scatter(
        x=forecast_conf_intervals['Date'],
        y=y_forecast,
        mode='markers',
        marker=dict(color='red', size=6),
        name='Predicted Score',
        customdata=future_df['Stage'],
        hovertemplate='Date: %{x|%Y %b}<br>Raw Score: %{y}<br>Stage: %{customdata}<extra></extra>'
    ))

    fig_score.add_trace(go.Scatter(
        x=[last_hist_date] + list(forecast_conf_intervals['Date']),
        y=[last_hist_score] + list(y_forecast),
        mode='lines',
        line=dict(color='red', dash='dot'),
        showlegend=False
    ))

    # Confidence Interval (CI) bands
    if ci_alpha is not None:
        if len(forecast_conf_intervals) == 1:
            d = forecast_conf_intervals['Date'].iloc[0]
            epsilon = pd.Timedelta(hours=12)
            left = d - epsilon
            right = d + epsilon
            upper = forecast_conf_intervals['Upper'].iloc[0]
            lower = forecast_conf_intervals['Lower'].iloc[0]
            upper_stage = forecast_conf_intervals['Upper_Stage'].iloc[0]
            lower_stage = forecast_conf_intervals['Lower_Stage'].iloc[0]

            fig_score.add_trace(go.Scatter(
                x=[left, right],
                y=[upper, upper],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.6)', dash='dot'),
                name='Upper Bound',
                customdata=[upper_stage] * 2,
                hovertemplate='Date: %{x|%Y %b}<br>Raw Score: %{y}<br>Stage: %{customdata}<br>Bound: Upper<extra></extra>',
                showlegend=False
            ))
            fig_score.add_trace(go.Scatter(
                x=[left, right],
                y=[lower, lower],
                mode='lines',
                fill='tonexty',
                line=dict(color='rgba(255,0,0,0.6)', dash='dot'),
                name='Lower Bound',
                customdata=[lower_stage] * 2,
                hovertemplate='Date: %{x|%Y %b}<br>Raw Score: %{y}<br>Stage: %{customdata}<br>Bound: Lower<extra></extra>',
                showlegend=False
            ))
        else:
            fig_score.add_trace(go.Scatter(
                x=forecast_conf_intervals['Date'],
                y=forecast_conf_intervals['Upper'],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.6)', dash='dot'),
                name='Upper Bound',
                customdata=forecast_conf_intervals['Upper_Stage'],
                hovertemplate='Date: %{x|%Y %b}<br>Raw Score: %{y}<br>Stage: %{customdata}<br>Bound: Upper<extra></extra>',
                showlegend=False
            ))
            fig_score.add_trace(go.Scatter(
                x=forecast_conf_intervals['Date'],
                y=forecast_conf_intervals['Lower'],
                mode='lines',
                fill='tonexty',
                line=dict(color='rgba(255,0,0,0.6)', dash='dot'),
                name='Lower Bound',
                customdata=forecast_conf_intervals['Lower_Stage'],
                hovertemplate='Date: %{x|%Y %b}<br>Raw Score: %{y}<br>Stage: %{customdata}<br>Bound: Lower<extra></extra>',
                showlegend=False
            ))

    # Stage background shading
    th = stage_thresholds[country]
    y_padding = 0.5
    y_min = macro_df_model_with_future['Raw_Score'].min() - y_padding
    y_max = macro_df_model_with_future['Raw_Score'].max() + y_padding

    stage_bands = [
        {'name': 'Recession', 'y0': y_min, 'y1': th[0], 'color': 'rgba(211, 47, 47, 0.45)'},  # #d32f2f
        {'name': 'Slowdown', 'y0': th[0], 'y1': th[1], 'color': 'rgba(255, 165, 0, 0.45)'},   # #FFA500
        {'name': 'Recovery', 'y0': th[1], 'y1': th[2], 'color': 'rgba(56, 142, 60, 0.45)'},   # #388e3c
        {'name': 'Expansion', 'y0': th[2], 'y1': y_max, 'color': 'rgba(25, 118, 210, 0.45)'}  # #1976d2
    ]

    for band in stage_bands:
        fig_score.add_shape(
            type='rect',
            xref='paper', yref='y',
            x0=0, x1=1,
            y0=band['y0'], y1=band['y1'],
            fillcolor=band['color'],
            layer='below',
            line_width=0
        )

    # Shared card style generator
    card_style = lambda bg: {
        'backgroundColor': bg,
        'padding': '15px',
        'borderRadius': '10px',
        'textAlign': 'center',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'minWidth': '220px',
        'minHeight': '140px',
        'color': '#fff'
    }

    # --- Current Stage ---
    current_stage = latest_stage
    current_score = round(macro_df_model['Raw_Score'].iloc[-1], 4)
    current_color = (
        '#d32f2f' if current_stage == 'Recession' else
        '#FFA500' if current_stage == 'Slowdown' else
        '#388e3c' if current_stage == 'Recovery' else
        '#1976d2'
    )
    current_card = html.Div([
        html.H4(f"Current Stage (as of {latest_stage_date})"),
        html.H3(current_stage, style={'margin': '0'}),
        html.P(f"Raw Score: {current_score}", style={'marginTop': '10px'}),
    ], style=card_style(current_color))

    # --- Next Forecast Stage ---
    next_date = future_df.index[0]
    next_stage = future_df.loc[next_date, 'Stage']
    next_score = round(future_df.loc[next_date, 'Raw_Score'], 4)
    ci_row = forecast_conf_intervals[forecast_conf_intervals['Date'] == next_date]
    ci_lower = round(ci_row['Lower'].values[0], 4)
    ci_upper = round(ci_row['Upper'].values[0], 4)
    next_color = (
        '#d32f2f' if next_stage == 'Recession' else
        '#FFA500' if next_stage == 'Slowdown' else
        '#388e3c' if next_stage == 'Recovery' else
        '#1976d2'
    )

    # Direction logic
    delta = next_score - current_score
    if delta > 0.05:
        direction_text = "Improving"
        direction_color = "#00e676"
    elif delta < -0.05:
        direction_text = "Worsening"
        direction_color = "#ff1744"
    else:
        direction_text = "Stable"
        direction_color = "#ffc400"

    next_card = html.Div([
        html.H4(f"Predicted Stage ({next_date.strftime('%Y-%m')})", style={'color': '#fff'}),
        html.H3(next_stage, style={'margin': '0', 'color': '#fff'}),
        html.P(f"Forecasted Score: {next_score}", style={'color': '#fff', 'marginTop': '10px'}),
        html.P(f"Forecasted Upper Bound: {ci_upper}", style={'color': '#fff', 'marginTop': '10px'}),
        html.P(f"Forecasted Lower Bound: {ci_lower}", style={'color': '#fff', 'marginTop': '10px'}),
        html.Div([
            html.Span(
                "▲" if direction_text == "Improving" else "▼" if direction_text == "Worsening" else "■",
                style={
                    'color': direction_color,
                    'fontWeight': 'bold',
                    'marginRight': '6px',
                    'fontSize': '16px'
                }
            ),
            html.Span(direction_text, style={'color': '#fff', 'fontSize': '14px'})
        ], style={
            'marginTop': '10px',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center'
        })
    ], style=card_style(next_color))

    # --- Legend Box ---
    legend_box = html.Div([
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

    # --- Combine all ---
    summary_cards = html.Div([
        html.Div([
            current_card,
            next_card,
            legend_box
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'flexWrap': 'wrap',
            'gap': '20px',
            'padding': '20px',
            'marginBottom': '20px',
        })
    ])


    highlight_conditional = [
        {
            'if': {'filter_query': f'{{Date_str}} = "{d.strftime("%Y-%m")}"'},
            'color': 'red',
            'fontWeight': 'bold'
        }
        for d in future_df.index
    ]

    return (
        fig_feature,
        fig_score,
        fig_importance,
        filtered_table_df.to_dict('records'),
        summary_cards,
        highlight_conditional,
        macro_df_model_with_future.to_json(date_format='iso')
    )

@app.callback(
    Output('dashboard-title', 'children'),
    Input('country-selector', 'value')
)
def update_title(country):
    return f"{country} Macroeconomic Dashboard"

@app.callback(
    Output('contribution-chart', 'figure'),
    [Input('score-graph', 'clickData'),
     Input('country-selector', 'value')],
    State('processed-data', 'data')
)
def update_contribution_chart(clickData, country, processed_data):
    if processed_data is None or clickData is None:
        return go.Figure()

    selected_date = pd.to_datetime(clickData['points'][0]['x'])
    features = country_features[country]

    df = pd.read_json(io.StringIO(processed_data), convert_dates=True)

    # Guard against mismatched data (e.g., country just changed)
    if not all(f in df.columns for f in features):
        return go.Figure()

    smoothed = df[features].rolling(window=3, min_periods=1).mean()
    if selected_date not in smoothed.index:
        return go.Figure()

    # Use the same logic as compute_macro_score
    def robust_z(x, median, mad):
        return (x - median) / (mad * 1.4826 + 1e-8)
    def logistic(x):
        return 2 / (1 + np.exp(-x)) - 1

    row = smoothed.loc[selected_date]

    # Reuse feature importance logic here
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    X = df[features].dropna()
    y = pd.qcut(X.mean(axis=1), q=4, labels=False)
    rf_model.fit(X, y)

    perm = permutation_importance(rf_model, X, y, n_repeats=10, random_state=42)
    perm_imp = pd.Series(perm.importances_mean, index=features)
    rf_imp = pd.Series(rf_model.feature_importances_, index=features)

    ridge = RidgeClassifier(alpha=1.0, random_state=42)
    ridge.fit(X, y)
    ridge_imp = pd.Series(np.abs(ridge.coef_[0]), index=features)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    gb_imp = pd.Series(gb.feature_importances_, index=features)

    logit = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)
    logit.fit(X, y)
    logit_imp = pd.Series(np.abs(logit.coef_[0]), index=features)

    imp_df = pd.DataFrame({
        'RF': rf_imp / rf_imp.max(),
        'GB': gb_imp / gb_imp.max(),
        'Ridge': ridge_imp / ridge_imp.max(),
        'Perm': perm_imp / perm_imp.max(),
        'Logit': logit_imp / logit_imp.max()
    })

    avg_imp = imp_df.mean(axis=1)
    weights_series = avg_imp * pd.Series({f: feature_signs[f] for f in features})

    contributions = {}
    for f in features:
        if pd.notna(row[f]):
            median = macro_df_model[f].median()
            mad = np.median(np.abs(macro_df_model[f] - median))
            rz = robust_z(row[f], median, mad)
            contrib = weights_series[f] * logistic(rz)
            contributions[f] = contrib
    contributions = pd.Series(contributions).sort_values()

    fig = go.Figure(go.Bar(
        x=contributions.values,
        y=contributions.index,
        orientation='h',
        marker_color=['green' if v > 0 else 'red' for v in contributions.values]
    ))

    fig.update_layout(
        title=f"Raw Score Breakdown — {selected_date.strftime('%Y-%m')}",
        xaxis_title="Contribution to Raw Score",
        yaxis_title="Feature",
        height=450,
        margin=dict(t=40, l=100, r=20, b=40)
    )

    return fig

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)



