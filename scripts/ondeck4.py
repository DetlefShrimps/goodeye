import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import requests
import logging
import json
import threading
from datetime import datetime, timedelta
from pybaseball import statcast, cache
import schedule
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

cache.enable()
sys.setrecursionlimit(3000)

def get_real_time_statcast_data():
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    data = statcast(start_dt=yesterday, end_dt=today)
    return data

def save_data():
    data = get_real_time_statcast_data()
    data.to_csv('/home/jesse/g00d3y3/data/baseball_savant.csv', index=False)
    data.to_excel('/home/jesse/g00d3y3/data/baseball_savant.xlsx', index=False)
    print(f"Data saved at {datetime.now()}")

def job():
    save_data()

def start_schedule():
    schedule.every(15).seconds.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

class BovadaScraper:
    def __init__(self):
        self.host = "https://www.bovada.lv"
        self.mlb_path = "/services/sports/event/v2/events/A/description/baseball/mlb"
        self.max_retries = 5
        self.retry_delay = 10
        self.headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Referer": "https://www.bovada.lv/"
        }

    def fetch_data(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            logging.info(f"Fetched data from {url}")
            if response.text:
                return response.json()
            else:
                logging.error(f"Empty response from {url}")
                return None
        except requests.RequestException as e:
            logging.error(f"Error fetching data from {url}: {e}")
            return None

    def parse_json(self, data):
        try:
            logging.debug("JSON data structure:\\n" + json.dumps(data, indent=2))
            events = []
            for event in data[0]['events']:
                event_data = {
                    'id': event['id'],
                    'description': event['description'],
                    'start_time': event['startTime'],
                    'competitors': [{'id': comp['id'], 'name': comp['name'], 'home': comp['home']} for comp in event['competitors']],
                    'markets': []
                }
                for group in event['displayGroups']:
                    for market in group['markets']:
                        market_data = {
                            'market_key': market['description'],
                            'outcomes': [{'id': outcome['id'], 'description': outcome['description'], 'price': outcome['price']['decimal']} for outcome in market['outcomes']]
                        }
                        event_data['markets'].append(market_data)
                events.append(event_data)
            return events
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            logging.error(f"Error parsing JSON data: {e}")
            return None

    def get_mlb_events(self):
        url = self.host + self.mlb_path
        data = self.fetch_data(url)
        if data:
            events = self.parse_json(data)
            return events
        return []

def load_dataset(path):
    try:
        data = pd.read_csv(path)
        print(f"Loaded data from {path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {path}: {e}")
        return pd.DataFrame()

def clean_and_scale_data(df, columns):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

def prepare_datasets(batting_path, adv_batting_path, pitching_path, adv_pitching_path, common_columns_batting, common_columns_pitching):
    batting_data = load_dataset(batting_path)
    adv_batting_data = load_dataset(adv_batting_path)
    pitching_data = load_dataset(pitching_path)
    adv_pitching_data = load_dataset(adv_pitching_path)

    # Convert common columns to float64 for consistency
    for col in common_columns_batting:
        try:
            batting_data[col] = batting_data[col].astype('float64')
        except ValueError as e:
            logging.warning(f"Skipping column {col} in batting data due to conversion error: {e}")
            batting_data.drop(columns=[col], inplace=True)

        try:
            adv_batting_data[col] = adv_batting_data[col].astype('float64')
        except ValueError as e:
            logging.warning(f"Skipping column {col} in advanced batting data due to conversion error: {e}")
            adv_batting_data.drop(columns=[col], inplace=True)

    for col in common_columns_pitching:
        try:
            pitching_data[col] = pitching_data[col].astype('float64')
        except ValueError as e:
            logging.warning(f"Skipping column {col} in pitching data due to conversion error: {e}")
            pitching_data.drop(columns=[col], inplace=True)

        try:
            adv_pitching_data[col] = adv_pitching_data[col].astype('float64')
        except ValueError as e:
            logging.warning(f"Skipping column {col} in advanced pitching data due to conversion error: {e}")
            adv_pitching_data.drop(columns=[col], inplace=True)

    # Merge datasets on common columns
    try:
        merged_batting_data = batting_data.merge(adv_batting_data, how='inner', on=list(common_columns_batting))
        merged_pitching_data = pitching_data.merge(adv_pitching_data, how='inner', on=list(common_columns_pitching))
    except Exception as e:
        logging.error(f"Error during merging: {e}")
        raise

    if merged_batting_data.empty or merged_pitching_data.empty:
        raise ValueError("Merged data is empty. Check your data merging and formulas.")
    else:
        print("Datasets merged successfully.")

    return merged_batting_data, merged_pitching_data

def train_models(X_train, y_train):
    # Define and train a RandomForest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred_rf)}")

    # Define and train a GradientBoosting model
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    print(f"Gradient Boosting MSE: {mean_squared_error(y_test, y_pred_gb)}")

    # Define and train a Neural Network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(formulas), activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    y_pred_nn = model.predict(X_test)
    print(f"Neural Network MSE: {mean_squared_error(y_test, y_pred_nn)}")

    return rf, gb, model

def make_predictions(new_data, models):
    rf, gb, model = models
    new_data_scaled, _ = clean_and_scale_data(new_data, new_data.columns)
    X_new = new_data_scaled.drop(columns=['target'])  # replace 'target' with actual target column name

    rf_preds = rf.predict(X_new)
    gb_preds = gb.predict(X_new)
    nn_preds = model.predict(X_new)

    return rf_preds, gb_preds, nn_preds

def main():
    # Define paths to datasets and common columns
    batting_path = '/home/jesse/g00d3y3/data/batting_common.csv'
    adv_batting_path = '/home/jesse/g00d3y3/data/adv_batting_common.csv'
    pitching_path = '/home/jesse/g00d3y3/data/pitching_common.csv'
    adv_pitching_path = '/home/jesse/g00d3y3/data/adv_pitching_common.csv'
    common_columns_batting = {'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB', 'CS'}
    common_columns_pitching = {'W', 'L', 'ERA', 'G', 'GS', 'CG', 'SHO', 'SV', 'IP', 'H', 'R', 'ER', 'HR', 'BB', 'SO'}

    # Prepare datasets
    batting_data, pitching_data = prepare_datasets(
        batting_path, adv_batting_path, pitching_path, adv_pitching_path,
        common_columns_batting, common_columns_pitching
    )

    # Define formulas (dummy example, replace with actual)
    formulas = {
        'BA': 'H / AB',
        'OBP': '(H + BB) / PA',
        'SLG': '(H + 2B + 2*3B + 3*HR) / AB',
        'OPS': 'OBP + SLG'
    }

    # Feature engineering
    for formula_name, formula in formulas.items():
        try:
            batting_data[formula_name] = eval(formula)
        except Exception as e:
            logging.warning(f"Skipping formula {formula_name} due to error: {e}")

    # Split data into training and test sets
    X = batting_data.drop(columns=['target'])  # replace 'target' with actual target column name
    y = batting_data['target']  # replace 'target' with actual target column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    models = train_models(X_train, y_train)

    # Save models
    joblib.dump(models[0], 'random_forest_model.pkl')
    joblib.dump(models[1], 'gradient_boosting_model.pkl')
    models[2].save('neural_network_model.h5')

    print("Models trained and saved successfully.")

    # Make predictions on new data (example)
    new_data = load_dataset('/home/jesse/g00d3y3/data/new_data.csv')
    rf_preds, gb_preds, nn_preds = make_predictions(new_data, models)
    print("Predictions made on new data.")

    # Print or save the predictions
    predictions_df = pd.DataFrame({
        'RandomForest_Predictions': rf_preds,
        'GradientBoosting_Predictions': gb_preds,
        'NeuralNetwork_Predictions': nn_preds
    })
    predictions_df.to_csv('/home/jesse/g00d3y3/data/predictions.csv', index=False)
    print("Predictions saved to '/home/jesse/g00d3y3/data/predictions.csv'.")

if __name__ == "__main__":
    main()
