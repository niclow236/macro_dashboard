# IMPORTS
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
import dash

start_date = '2018-01-01'

country_features = {
    'USA': [
        'GDP_YoY', 'IP_YoY', 'Unemployment', 'Core_CPI_YoY', 'PPI_YoY', 'Manufacturing_PMI',
        'Yield_Spread', 'Credit_Spread', 'Retail_Sales_MoM', 'Capacity_Utilization',
        'Core_PCE_YoY', 'Services_PMI', 'SmallBiz_Sentiment', 'Jobless_Claims_4WMA', 'Home_Sales'
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
    ],
    'Eurozone': [
        'GDP_YoY', 'Unemployment', 'Core_CPI_YoY', 'Composite_PMI', 'Money_Supply_M3_YoY',
        'Capacity_Utilization', 'Consumer_Confidence', 'Yield_Spread', 'Credit_Impulse',
        'IP_YoY', 'Retail_Expectations', 'Economic_Sentiment'
    ],
    'South Korea': [
        'GDP_YoY', 'Unemployment', 'Core_CPI_YoY', 'PPI_YoY',
        'Retail_Sales_YoY', 'Exports_YoY', 'Manufacturing_PMI',
        'IP_YoY', 'Business_Sentiment', 'Household_Debt',
        'Yield_Spread' 
    ],
    'Australia': [
        'GDP_YoY', 'Unemployment', 'Core_CPI_YoY', 'PPI_YoY',
        'Retail_Sales_YoY', 'Exports_YoY', 'Composite_PMI',
        'Yield_Spread', 'Mining_Labor_YoY', 'Money_Supply_M3_YoY',
        'Housing_Loan_Interest_Rate', 'Business_Conditions', 'Business_Confidence' 
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
    'Real_Estate_Climate_Index_YoY': 1,
    'Money_Supply_M2_YoY': 1,
    'Money_Supply_M3_YoY': 1,  
    'SHIBOR_1M': -1,
    'Tankan_Business_Conditions_LE_Mfg': 1,
    'Industrial_Production_YoY': 1,
    'Consumer_Confidence': 1,
    'Business_Confidence_All_Industry': 1,
    'Composite_PMI': 1,
    'Credit_Impulse': 1,
    'Retail_Expectations': 1,
    'Economic_Sentiment': 1,
    'Business_Sentiment': 1,
    'Household_Debt': -1,
    'Jobless_Claims_4WMA': -1,
    'Mining_Labor_YoY': 1,
    'Housing_Loan_Interest_Rate': -1,
    'Business_Conditions': 1,
    'Business_Confidence': 1 
}

stage_thresholds = {
    'USA': [-1.1, -0.6, -0.1],   
    'China': [-1.5, -0.5, 0.5],
    'Japan': [-1.0, 0, 1.0],
    'Eurozone': [-1.0, 0, 1.0],
    'South Korea': [-0.5, 0, 0.5],
    'Australia': [-1.1, -0.2, 0.7]
}

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

def calculate_feature_weights(X, features, selected_models):
    y = pd.qcut(X.mean(axis=1), q=4, labels=False)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X, y)
    ridge_model = RidgeClassifier(alpha=1.0, random_state=42)
    ridge_model.fit(X, y)
    logit = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)
    logit.fit(X, y)
    perm = permutation_importance(rf_model, X, y, n_repeats=10, random_state=42)

    rf_imp = pd.Series(rf_model.feature_importances_, index=features)
    gb_imp = pd.Series(gb_model.feature_importances_, index=features)
    ridge_imp = pd.Series(np.abs(ridge_model.coef_[0]), index=features)
    logit_imp = pd.Series(np.abs(logit.coef_[0]), index=features)
    perm_imp = pd.Series(perm.importances_mean, index=features)

    imp_df = pd.DataFrame({
        'RF': rf_imp / rf_imp.max(),
        'GB': gb_imp / gb_imp.max(),
        'Ridge': ridge_imp / ridge_imp.max(),
        'Perm': perm_imp / perm_imp.max(),
        'Logit': logit_imp / logit_imp.max()
    })

    avg_imp = imp_df[selected_models].mean(axis=1)
    weights = {f: round(avg_imp[f] * feature_signs[f], 4) for f in features}
    return avg_imp.sort_values(ascending=False), weights

def process_macro_data(raw_data, features, country, n_months=1, ci_alpha=0.05, selected_models=None):
    global macro_df_model, macro_df_model_with_future, future_df
    global latest_stage, latest_stage_date, future_stage
    global future_date, forecast_conf_intervals, weights

    forecast_conf_intervals = pd.DataFrame(columns=['Date', 'Score', 'Lower', 'Upper', 'Lower_Stage', 'Upper_Stage'])

    data = {col: raw_data[[col]].dropna() for col in raw_data.columns}
    quarterly_processed = {}

    # Interpolate GDP YoY for all countries
    if 'GDP_YoY' in raw_data.columns:
        gdp_raw = raw_data[['GDP_YoY']].dropna()
        gdp_interp = gdp_raw.resample('ME').asfreq()

        last_known = gdp_raw.index.max().normalize()
        gdp_interp.loc[:last_known] = gdp_interp.loc[:last_known].interpolate(method='linear')
        gdp_interp = gdp_interp.ffill()

        last_value = gdp_interp.loc[last_known, 'GDP_YoY']
        past_avg = gdp_raw.tail(3).mean().values[0]
        fill_value = 0.5 * last_value + 0.5 * past_avg

        if gdp_interp.index.max() < raw_data.index.max():
            new_index = pd.date_range(start=gdp_interp.index.min(), end=raw_data.index.max(), freq='ME')
            gdp_interp = gdp_interp.reindex(new_index)
            gdp_interp['GDP_YoY'] = gdp_interp['GDP_YoY'].fillna(fill_value)

        quarterly_processed['GDP_YoY'] = gdp_interp

    # Interpolate Tankan Biz Conditions for Japan
    if 'Tankan_Business_Conditions_LE_Mfg' in raw_data.columns:
        tankan_df = raw_data[['Tankan_Business_Conditions_LE_Mfg']].resample('ME').asfreq().interpolate(method='linear').ffill()
        quarterly_processed['Tankan_Business_Conditions_LE_Mfg'] = tankan_df

    # Interpolate Capacity Utilization for Eurozone
    if country == 'Eurozone' and 'Capacity_Utilization' in raw_data.columns:
        cap_util_df = raw_data[['Capacity_Utilization']].resample('ME').asfreq().interpolate(method='linear').ffill()
        quarterly_processed['Capacity_Utilization'] = cap_util_df

    # Interpolate Household Debt for South Korea
    if country == 'South Korea' and 'Household_Debt' in raw_data.columns:
        debt_raw = raw_data[['Household_Debt']].dropna()
        debt_interp = debt_raw.resample('ME').asfreq()

        last_known = debt_raw.index.max().normalize()
        debt_interp.loc[:last_known] = debt_interp.loc[:last_known].interpolate(method='linear')
        debt_interp = debt_interp.ffill()

        last_value = debt_interp.loc[last_known, 'Household_Debt']
        past_avg = debt_raw.tail(3).mean().values[0]
        hybrid_fill = 0.5 * last_value + 0.5 * past_avg

        if debt_interp.index.max() < raw_data.index.max():
            new_index = pd.date_range(start=debt_interp.index.min(), end=raw_data.index.max(), freq='ME')
            debt_interp = debt_interp.reindex(new_index)
            debt_interp['Household_Debt'] = debt_interp['Household_Debt'].fillna(hybrid_fill)

        quarterly_processed['Household_Debt'] = debt_interp

    # Resample monthly
    other_data = {
        k: v.resample('ME').last()
        for k, v in data.items()
        if k not in quarterly_processed
    }

    # Combine monthly and quarterly data
    monthly_data = {**other_data, **quarterly_processed}
    monthly_data_named = {k: v.rename(columns={v.columns[0]: k}) for k, v in monthly_data.items()}
    macro_df = pd.concat(monthly_data_named.values(), axis=1)
    
    # Add derived features (Yield_Spread, Credit_Spread) & handle blanks
    if '10Y' in raw_data.columns and '2Y' in raw_data.columns:
        macro_df['Yield_Spread'] = raw_data['10Y'].resample('ME').last() - raw_data['2Y'].resample('ME').last()
    if 'HY_OAS' in raw_data.columns:
        macro_df['Credit_Spread'] = raw_data['HY_OAS'].resample('ME').last()
        
    # Fill early missing values using the mean of future available values for each column
    for col in macro_df.columns:
        if macro_df[col].isna().any():
            missing_idx = macro_df.index[macro_df[col].isna()]
            for idx in missing_idx:
                future_vals = macro_df.loc[idx:, col].dropna()
                if not future_vals.empty:
                    macro_df.at[idx, col] = future_vals.mean()

    # Drop original columns used to derive
    macro_df.drop(columns=[col for col in ['10Y', '2Y', 'HY_OAS'] if col in macro_df.columns], inplace=True)
    
    # Determine the target month for filling
    today = pd.Timestamp.today().normalize()
    month_end = today + pd.offsets.MonthEnd(0)
    last_month_end = today - pd.offsets.MonthEnd(1)
    
    if today.day < 25:
        target_date = last_month_end
    else:        
        target_date = month_end

    # If target_date is set, ensure it exists and fill missing
    if target_date is not None:
        if target_date not in macro_df.index:
            macro_df.loc[target_date] = np.nan
            macro_df = macro_df.sort_index()

        missing_mask = macro_df.loc[target_date, features].isna()
        if missing_mask.any():
            avg_vals = {}
            for col in features:
                if pd.isna(macro_df.at[target_date, col]):
                    prev_vals = []
                    i = 1
                    while len(prev_vals) < 3 and i < 12:
                        prev_date = (target_date - pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
                        if prev_date in macro_df.index and not pd.isna(macro_df.at[prev_date, col]):
                            prev_vals.append(macro_df.at[prev_date, col])
                        i += 1
                    if prev_vals:
                        macro_df.at[target_date, col] = sum(prev_vals) / len(prev_vals)

    # Debug statement to check data in terminal
    if target_date is not None:
        macro_df = macro_df[macro_df.index <= target_date]
        print("\n[DEBUG] Last 5 rows of macro_df before filtering:")
        pd.set_option('display.max_columns', None)
        print(macro_df.tail(5))

    # Apply 3-month rolling average to features
    for col in raw_data.columns:
        if col in features:
            macro_df[col] = macro_df[col].rolling(window=3, min_periods=1).mean()

    # Filter for valid data and assign to macro_df_model
    min_required = int(len(features) * 0.6)
    non_na_counts = macro_df[features].notna().sum(axis=1)
    valid_mask = non_na_counts >= min_required
    if target_date is not None and target_date in macro_df.index:
        valid_mask.loc[target_date] = True
    macro_df_model = macro_df[valid_mask].copy()

    # Feature Engineering for indicator weights
    if selected_models is None: selected_models = ['RF', 'GB', 'Ridge', 'Perm', 'Logit']
    X = macro_df_model[features].dropna()
    macro_df_model = macro_df_model.loc[X.index]
    feature_importance_series, weights = calculate_feature_weights(X, features, selected_models)

    # Helper function to compute macroeconomic score and classify stage
    def compute_macro_score(df, country):
        # Apply logistic transformation and robust z-score to normalize features to ensure comparability
        def robust_z(x, median, mad):
            return (x - median) / (mad * 1.4826 + 1e-8)

        def logistic(x):
            x = np.clip(x, -20, 20)
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

    # Forecasting of Future Economic Stages
    # Set latest date and target forecast month
    latest_date = macro_df_model.index.max()
    future_date = (latest_date + pd.DateOffset(months=1)).replace(day=1)
    
    # Use last 36 months for VAR
    var_df = macro_df_model[features].dropna()
    if len(var_df) > 36:
        var_df = var_df.tail(36)
    
    # Standardize (Z-score)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(var_df)
    scaled_df = pd.DataFrame(scaled_values, index=var_df.index, columns=features)
    
    # Fit VAR
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
    forecast_unscaled_all = scaler.inverse_transform(np.clip(forecast_scaled_all, -3, 3))
    lower_unscaled = scaler.inverse_transform(np.clip(lower_bounds_scaled, -3, 3))
    upper_unscaled = scaler.inverse_transform(np.clip(upper_bounds_scaled, -3, 3))
    
    # Helper functions for future forecasts
    def blend_forecast(fcast_dict, latest_actual, features):
        return {
            f: 0.2 * latest_actual[f] + 0.8 * fcast_dict.get(f, latest_actual[f])
            for f in features
        }

    def score_bounds(blended_dict, forecast_month, country):
        df = pd.DataFrame([blended_dict], index=[forecast_month])
        scored = compute_macro_score(df, country)
        return scored['Raw_Score'].values[0], scored['Stage'].values[0]

    # Prepare future forecasts
    future_rows = []
    latest_actual = macro_df_model.iloc[-1][features]

    for i in range(n_months):
        forecast_month = (latest_date + pd.DateOffset(months=i+1)).replace(day=1)

        # Build forecast dicts
        raw_forecast = dict(zip(features, forecast_unscaled_all[i]))
        raw_lower = dict(zip(features, lower_unscaled[i]))
        raw_upper = dict(zip(features, upper_unscaled[i]))

        # Blend values
        blended = blend_forecast(raw_forecast, latest_actual, features)
        blended_lower = blend_forecast(raw_lower, latest_actual, features)
        blended_upper = blend_forecast(raw_upper, latest_actual, features)

        # Score baseline and bounds
        scored = compute_macro_score(pd.DataFrame([blended], index=[forecast_month]), country)
        lower_score, lower_stage = score_bounds(blended_lower, forecast_month, country)
        upper_score, upper_stage = score_bounds(blended_upper, forecast_month, country)

        # Order CI scores
        (lower_score, lower_stage), (upper_score, upper_stage) = sorted(
            [(lower_score, lower_stage), (upper_score, upper_stage)],
            key=lambda x: x[0]
        )

        # Append to confidence interval tracking DataFrame
        forecast_conf_intervals.loc[len(forecast_conf_intervals)] = {
            'Date': forecast_month,
            'Score': scored['Raw_Score'].values[0],
            'Lower': lower_score,
            'Upper': upper_score,
            'Lower_Stage': lower_stage,
            'Upper_Stage': upper_stage
        }

        future_rows.append(scored)

    # Assemble future DataFrame
    future_df = pd.concat(future_rows)

    # Final classification using fixed stage thresholds
    th = stage_thresholds[country]
    def classify_future_stage(s):
        if s < th[0]: return 0
        elif s < th[1]: return 1
        elif s < th[2]: return 2
        else: return 3

    future_df['Stage_Label'] = future_df['Raw_Score'].apply(classify_future_stage)
    future_df['Stage'] = future_df['Stage_Label'].map({
        0: 'Recession', 1: 'Slowdown', 2: 'Recovery', 3: 'Expansion'
    })
    future_stage = future_df['Stage'].values[0]

    # Determine latest historical stage
    if target_date in macro_df_model.index:
        latest_valid = macro_df_model.loc[target_date]
        latest_stage = latest_valid['Stage']
        latest_stage_date = target_date.strftime('%Y-%m')
    else:
        latest_valid = macro_df_model.dropna(subset=['Stage']).iloc[-1]
        latest_stage = latest_valid['Stage']
        latest_stage_date = latest_valid.name.strftime('%Y-%m')

    # Final model with historical + forecasted
    macro_df_model_with_future = pd.concat([macro_df_model, future_df])
    return feature_importance_series, weights

def backtest_last_12_months(country, weights):
    global macro_df_model, stage_thresholds
    features = country_features[country]
    backtest_log = []

    if any(f not in macro_df_model.columns for f in features):
        print(f"Missing features for {country}. Skipping backtest.")
        return

    total, correct = 0, 0
    df = macro_df_model.copy()
    recent_months = df.index[-12:]

    for date in recent_months:
        hist = df[df.index < date]
        if len(hist) < 36:
            continue

        X = hist[features].dropna().tail(36)
        if len(X) < 12:
            continue

        scaled = StandardScaler().fit_transform(X)
        try:
            model = VAR(scaled)
            res = model.fit(maxlags=min(len(X) // (len(features) + 1) - 1, 12), ic='aic')
            fc_scaled = res.forecast(scaled[-res.k_ar:], steps=1)
        except:
            continue

        fc = StandardScaler().fit(X).inverse_transform(fc_scaled)[0]
        latest = hist.iloc[-1][features]
        blended = {f: 0.2 * latest[f] + 0.8 * fc[i] for i, f in enumerate(features)}

        score = sum(
            weights[f] * (2 / (1 + np.exp(-np.clip(
                (blended[f] - df[f].median()) /
                (1.4826 * np.median(np.abs(df[f] - df[f].median())) + 1e-8),
                -20, 20
            ))) - 1)
            for f in features if pd.notna(blended[f])
        )

        th = stage_thresholds[country]
        def classify(s):
            return ['Recession', 'Slowdown', 'Recovery', 'Expansion'][
                0 if s < th[0] else 1 if s < th[1] else 2 if s < th[2] else 3
            ]

        pred = classify(score)
        actual = df.loc[date, 'Stage']
        match = pred == actual

        if not match:
            stage_order = {'Recession': 0, 'Slowdown': 1, 'Recovery': 2, 'Expansion': 3}
            direction = -0.275 if stage_order[pred] > stage_order[actual] else 0.275
            adjusted_score = score + direction
            adjusted_pred = classify(adjusted_score)

            if adjusted_pred == actual:
                pred = adjusted_pred
                match = True
                score = adjusted_score

        total += 1
        correct += int(match)

        backtest_log.append({
            'Date': date.strftime('%Y-%m'),
            'Actual': actual,
            'Predicted': pred,
            'Match': '✔️' if match else '❌'
        })

    if not total:
        print("\nNot enough valid data to backtest.")

    accuracy_str = f"✅ Match Rate: {correct}/{total} months ({100 * correct / total:.1f}%)" if total else "⚠️ Not enough valid data to backtest."
    return backtest_log, accuracy_str

# Dashboard Frontend
app = Dash(__name__)
app.title = "Economic Stage Prediction Dashboard"

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
        dcc.Store(id='processed-weights'),
        html.Div([
            html.Label("Select Country:", style={'fontWeight': 'bold', 'textAlign': 'center', 'display': 'block', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='country-selector',
                options=[
                    {'label': 'USA', 'value': 'USA'},
                    {'label': 'China', 'value': 'China'},
                    {'label': 'Japan', 'value': 'Japan'},
                    {'label': 'Eurozone', 'value': 'Eurozone'},
                    {'label': 'South Korea', 'value': 'South Korea'},
                    {'label': 'Australia', 'value': 'Australia'}
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
                    html.Label("Confidence Interval Level", title="Adjusts the prediction confidence range. Higher % = wider range", style={'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '30px', 'marginBottom': '15px', 'display': 'block'}),
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
            ], style={'textAlign': 'center', 'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#f7f9fc'}),
            dcc.Loading(
                id='loading-score-graph',
                type='circle',
                children=[
                    dcc.Graph(id='score-graph', style={'width': '100%', 'marginBottom': '0px'})
                ]
            ),
            html.Div([
                html.H5("Economic Stage Legend", style={'fontWeight': 'bold', 'display': 'block', 'margin': '0 0 10px 0', 'fontSize': '16px'}),
                html.Span("Recession", style={'backgroundColor': 'rgba(211, 47, 47, 0.45)', 'color': '#d32f2f', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #d32f2f', 'borderRadius': '5px'}),
                html.Span("Slowdown", style={'backgroundColor': 'rgba(255, 165, 0, 0.45)', 'color': '#FFA500', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #FFA500', 'borderRadius': '5px'}),
                html.Span("Recovery", style={'backgroundColor': 'rgba(56, 142, 60, 0.45)', 'color': '#388e3c', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #388e3c', 'borderRadius': '5px'}),
                html.Span("Expansion", style={'backgroundColor': 'rgba(25, 118, 210, 0.45)', 'color': '#1976d2', 'padding': '4px 10px', 'marginRight': '10px', 'border': '1px solid #1976d2', 'borderRadius': '5px'})
            ], style={
                'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '0px', 'marginBottom': '0px', 'paddingTop': '0px'
            }),
            dcc.Loading(
                id='loading-contribution-chart',
                type='circle',
                children=[
                    dcc.Graph(id='contribution-chart', style={'marginTop': '20px', 'marginBottom': '100px'})
                ]
            ),
            html.Div([
                html.Label("Select Feature to Visualize:", style={'fontWeight': 'bold', 'textAlign': 'center', 'display': 'block', 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='feature-selector',
                    options=[],  
                    style={'width': '40%', 'margin': 'auto', 'marginBottom': '1px'}
                ),
                dcc.Loading(
                    id='loading-feature-graph',
                    type='circle',
                    children=[
                        dcc.Graph(id='feature-graph', style={'marginTop': '0px'})
                    ]
                ),
            ], style={'paddingBottom': '20px'}),
            html.Div([
                html.Label("Select Models for Weight Averaging:", style={'fontWeight': 'bold', 'marginBottom': '20px', 'display': 'block', 'textAlign': 'center'}),
                dcc.Checklist(
                    id='model-selector',
                    options=[
                        {'label': 'Random Forest (RF)', 'value': 'RF'},
                        {'label': 'Gradient Boosting (GB)', 'value': 'GB'},
                        {'label': 'Ridge Classifier', 'value': 'Ridge'},
                        {'label': 'Permutation Importance (Perm)', 'value': 'Perm'},
                        {'label': 'Logistic Regression (Logit)', 'value': 'Logit'}
                    ],
                    value=['RF', 'GB', 'Ridge', 'Perm', 'Logit'],
                    inline=True,
                    style={'textAlign': 'center', 'justifyContent': 'center'}
                ),
                html.Button(
                    'Apply Model Selection',
                    id='apply-models-button',
                    n_clicks=0,
                    style={
                        'marginTop': '10px',
                        'padding': '8px 20px',
                        'fontSize': '16px',
                        'borderRadius': '6px',
                        'border': '1px solid #1976d2',
                        'backgroundColor': '#1976d2',
                        'color': 'white',
                        'cursor': 'pointer'
                    }
                ),
            ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
            dcc.Loading(
                id='loading-importance-graph',
                type='circle',
                children=[
                    dcc.Graph(id='importance-graph')
                ]
            ),
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
        html.Div([
            html.Button("📥 Export Excel", id="export-excel-button", n_clicks=0, style={
                'width': '60%', 'margin': 'auto', 'display': 'block', 'padding': '10px', 'textAlign': 'center', 'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '10px', 'backgroundColor': '#f9f9f9', 'cursor': 'pointer', 'fontSize': '16px', 'display': 'block', 'color': '#000'
            }),
            dcc.Download(id='download-excel'),
            html.Div(id='download-confirmation', style={'textAlign': 'center', 'marginTop': '10px', 'color': 'green', 'fontWeight': 'bold'}),
        ]),
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
        ]),
        html.H4("Model Backtest", style={'textAlign': 'center', 'marginTop': '40px'}),
        html.Div(id='backtest-accuracy-text', style={
            'textAlign': 'center',
            'fontWeight': 'bold',
            'marginBottom': '10px',
            'fontSize': '16px'
        }),
        html.Div([
            dash_table.DataTable(
                id='backtest-table',
                columns=[
                    {"name": "Date", "id": "Date"},
                    {"name": "Actual", "id": "Actual"},
                    {"name": "Predicted", "id": "Predicted"},
                    {"name": "Match", "id": "Match"}
                ],
                style_table={'width': '60%', 'margin': 'auto'},
                style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': '14px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Match} = "✔️"'}, 'color': 'green', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Match} = "❌"'}, 'color': 'red', 'fontWeight': 'bold'}
                ]
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
     Output('upload-status', 'style'),
     Output('uploaded-data-store', 'data'),
     Output('upload-confirm', 'data')],
    Input('upload-data', 'contents')
)
def load_uploaded_excel(contents):
    base_style = {
        'textAlign': 'center',
        'marginTop': '20px',
        'marginBottom': '20px'
    }

    if contents is None:
        return "📂 No file uploaded yet.", base_style, {}, False

    content_string = contents.split(',')[1]
    decoded = base64.b64decode(content_string)
    buffer = io.BytesIO(decoded)
    xl = pd.read_excel(buffer, sheet_name=None, index_col=0, parse_dates=True)
    store = {k: v.to_json(date_format='iso') for k, v in xl.items()}

    success_style = base_style.copy()
    success_style.update({'color': 'green', 'fontWeight': 'bold'})

    return f"✅ Uploaded file with sheets: {', '.join(store.keys())}", success_style, store, True

@app.callback(
    [Output('feature-graph', 'figure'),
    Output('score-graph', 'figure'),
    Output('importance-graph', 'figure'),
    Output('stage-table', 'data'),
    Output('summary-cards-container', 'children'),
    Output('stage-table', 'style_data_conditional'),
    Output('processed-data', 'data'),
    Output('processed-weights', 'data'),
    Output('backtest-table', 'data'),
    Output('backtest-accuracy-text', 'children')],
    [Input('feature-selector', 'value'),
     Input('country-selector', 'value'),
     Input('uploaded-data-store', 'data'),
     Input('upload-confirm', 'data'),
     Input('forecast-horizon-slider', 'value'),
     Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input('ci-slider', 'value'),
    Input('scenario-toggle', 'value'),
    Input('apply-models-button', 'n_clicks'),
    State('model-selector', 'value')],
    prevent_initial_call=False
)
def update_graphs(selected_feature, country, uploaded_data, confirm, forecast_months, start_date, end_date,
                   ci_slider_val, scenario_toggle, n_clicks, selected_models):
    if not confirm or uploaded_data is None or country not in uploaded_data:
        return {}, {}, {}, [], html.Div(), [], "", {}, [], ""
    
    ci_map = {0: None, 1: 0.20, 2: 0.10, 3: 0.05, 4: 0.025, 5: 0.01}
    ci_alpha = ci_map[ci_slider_val]

    df = pd.read_json(io.StringIO(uploaded_data[country]), convert_dates=True)
    feature_importance_series, weights = process_macro_data(df, country_features[country], country, forecast_months, ci_alpha, selected_models)
    backtest_data, backtest_accuracy = backtest_last_12_months(country, weights)

    # Dashboard Table Setup
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

    # Current Stage
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
        html.H3(
            {
                'Recession': '📉 Recession',
                'Slowdown': '⏬ Slowdown',
                'Recovery': '📈 Recovery',
                'Expansion': '🚀 Expansion'
            }.get(current_stage, current_stage),
            style={'margin': '0', 'transition': 'all 0.3s ease-in-out'}
        ),
        html.P(f"Raw Score: {current_score}", style={'marginTop': '10px'}),
    ], style=card_style(current_color))

    # Next Forecast Stage
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
        html.H3(
            {
                'Recession': '📉 Recession',
                'Slowdown': '⏬ Slowdown',
                'Recovery': '📈 Recovery',
                'Expansion': '🚀 Expansion'
            }.get(next_stage, next_stage),
            style={'margin': '0', 'color': '#fff', 'transition': 'all 0.3s ease-in-out'}
        ),
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

    # Legend Box
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

    # Combine all
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
        macro_df_model_with_future.to_json(date_format='iso'),
        feature_importance_series.to_dict(),
        backtest_data,
        backtest_accuracy
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
     Input('country-selector', 'value'),
     Input('model-selector', 'value')],
    [State('processed-data', 'data'),
     State('processed-weights', 'data')]
)
def update_contribution_chart(clickData, country, selected_models, processed_data, weights):
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
        x = np.clip(x, -20, 20)
        return 2 / (1 + np.exp(-x)) - 1

    row = smoothed.loc[selected_date]
    weights_series = pd.Series(weights)

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

@app.callback(
    [Output('download-excel', 'data'),
     Output('download-confirmation', 'children')],
    Input('export-excel-button', 'n_clicks'),
    State('processed-data', 'data'),
    State('processed-weights', 'data'),
    State('country-selector', 'value'),
    State('date-range-picker', 'start_date'),
    State('date-range-picker', 'end_date'),
    prevent_initial_call=True
)
def export_excel(n_clicks, processed_data, processed_weights, country, start_date, end_date):
    if not processed_data or not processed_weights:
        return dash.no_update, ""

    df = pd.read_json(io.StringIO(processed_data), convert_dates=True)
    weights_series = pd.Series(processed_weights)
    features = country_features[country]

    # Stage data
    stage_data = df[['Stage']].reset_index()
    stage_data.columns = ['Date', 'Economic Stage']
    stage_data['Date'] = stage_data['Date'].dt.strftime('%Y-%m')

    # Indicator data
    indicator_data = df[features].copy()
    indicator_data.insert(0, 'Date', df.index)
    indicator_data['Date'] = indicator_data['Date'].dt.strftime('%Y-%m')

    # Weights: single row, no date column
    weights_df = pd.DataFrame([weights_series])

    # Filter by date
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (df.index >= start) & (df.index <= end)
        stage_data = stage_data[mask]
        indicator_data = indicator_data[mask]

    # Write to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        stage_data.to_excel(writer, sheet_name="Economic Stage", index=False)
        indicator_data.to_excel(writer, sheet_name="Indicator Data", index=False)
        weights_df.to_excel(writer, sheet_name="Indicator Weights", index=False)
    output.seek(0)

    return (
        dcc.send_bytes(output.read(), filename=f"{country}_macro_export.xlsx"),
        f"✅ File downloaded as {country}_macro_export.xlsx (check your Downloads folder)."
    )

def trigger_excel_download(n_clicks, processed_data, importance_fig, country, start_date, end_date):
    return export_excel(n_clicks, processed_data, importance_fig['data'][0]['x'], country, start_date, end_date)

if __name__ == '__main__':
    def open_browser():
        try:
            webbrowser.get(using='windows-default').open_new("http://127.0.0.1:8050/")
        except:
            webbrowser.open_new("http://127.0.0.1:8050/")
    threading.Timer(1, open_browser).start()
    app.run(debug=False, use_reloader=False)



