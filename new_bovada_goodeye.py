import pandas as pd
import numpy as np
from datetime import datetime
import requests
import joblib
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
from pybaseball import statcast
import logging
import time
from pybaseball import cache

cache.enable()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetching Statcast Data from the start of the 2023 season to current
def fetch_statcast_data(start_date='2023-03-30', end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    data = statcast(start_dt=start_date, end_dt=end_date)
    return data

# Define a function to fetch the necessary columns from the Statcast data
def get_required_statcast_data(data):
    required_columns = [
        'player_name', 'game_date', 'home_runs', 'hits', 'strikeouts', 
        'release_speed', 'pfx_z', 'effective_speed', 'release_spin_rate', 'inning', 'balls', 'strikes',
        'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'bat_score', 'fld_score'
    ]
    return data[required_columns]

# BovadaScraper class adapted from bovada.py
class BovadaScraper:
    def __init__(self):
        self.host = "https://www.bovada.lv"
        self.mlb_path = "/services/sports/event/v2/events/A/description/baseball/mlb"
        self.max_retries = 5
        self.retry_delay = 10  # seconds
        self.headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",  # Do Not Track Request Header
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
            events = []
            for event in data[0]['events']:  # Access the first element and then 'events'
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
                            'outcomes': [{'description': outcome['description'], 'price': outcome['price'], 'handicap': outcome['price'].get('handicap')} for outcome in market['outcomes']]
                        }
                        event_data['markets'].append(market_data)
                events.append(event_data)
            logging.info(f"Parsed {len(events)} events from JSON data")
            return events
        except Exception as e:
            logging.error(f"Error parsing JSON data: {e}")
            return None

    def verify_data(self, events):
        if not events:
            return False
        required_keys = {'id', 'description', 'start_time', 'competitors', 'markets'}
        for event in events:
            if not all(key in event for key in required_keys):
                logging.error(f"Missing keys in event: {event}")
                return False
        return True

    def retry_strategy(self, url):
        retries = 0
        while retries < self.max_retries:
            data = self.fetch_data(url)
            if data:
                events = self.parse_json(data)
                if events and self.verify_data(events):
                    return events
            retries += 1
            logging.info(f"Retrying... ({retries}/{self.max_retries})")
            time.sleep(self.retry_delay)
        return None

    def scrape_bovada(self):
        mlb_url = f"{self.host}{self.mlb_path}"
        mlb_bets = self.retry_strategy(mlb_url)
        if mlb_bets:
            return mlb_bets
        else:
            logging.error("Failed to fetch and verify all required bets after multiple retries.")
            return None

# Function to handle missing values
def handle_missing_values(df):
    try:
        df = df.dropna(thresh=len(df) * 0.5, axis=1)
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            df[column].fillna(df[column].mean(), inplace=True)
        for column in df.select_dtypes(include=['object']).columns:
            df[column].fillna(df[column].mode()[0], inplace=True)
    except Exception as e:
        print(f"Error handling missing values: {e}")
    return df

# Function to handle outliers using z-scores
def remove_outliers(df):
    try:
        z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        df = df[filtered_entries]
    except Exception as e:
        print(f"Error handling outliers: {e}")
    return df

# Standardize and normalize the data with error handling
def standardize_normalize_data(df):
    try:
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include(['object'])).columns

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        df_cleaned = preprocessor.fit_transform(df)
        df_cleaned = pd.DataFrame(df_cleaned, columns=numerical_cols.tolist() + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols).tolist())
    except Exception as e:
        print(f"Error standardizing and normalizing data: {e}")
    return df_cleaned

# Enhanced feature engineering incorporating additional insights
def enhanced_feature_engineering_v2(statcast_data_cleaned_df, pitcher_vs_batter_stats, ballpark_data, weather_data):
    try:
        player_stats = statcast_data_cleaned_df.groupby(['player_name', 'game_date']).agg({
            'home_runs': 'sum',
            'hits': 'sum',
            'strikeouts': 'sum',
            'release_speed': 'mean',
            'pfx_z': 'mean',
            'effective_speed': 'mean',
            'release_spin_rate': 'mean',
            'inning': 'first',  # Adding game context
            'balls': 'mean',    # Adding count information
            'strikes': 'mean',
            'game_date': 'first',  # Keeping game_date for merge
            'ballpark': 'first'    # Keeping ballpark for merge
        }).reset_index()

        # Add recent and season statistics
        player_stats['recent_home_runs'] = player_stats.groupby('player_name')['home_runs'].transform(lambda x: x.rolling(window=10).mean())
        player_stats['season_home_runs'] = player_stats.groupby('player_name')['home_runs'].transform(lambda x: x.cumsum())

        # Merge with pitcher vs batter stats
        player_stats = pd.merge(player_stats, pitcher_vs_batter_stats[['player_name', 'opposing_pitcher', 'home_runs']], on=['player_name', 'opposing_pitcher'], how='left')
        player_stats['ballpark_home_run_factor'] = player_stats['ballpark'].map(ballpark_data['home_run_factor'])

        # Add weather data
        player_stats = pd.merge(player_stats, weather_data[['game_date', 'ballpark', 'wind_speed', 'wind_direction', 'temperature']], on=['game_date', 'ballpark'], how='left')

        # Similar feature engineering for Total Strikeouts and Total Hits
        player_stats['recent_strikeouts'] = player_stats.groupby('player_name')['strikeouts'].transform(lambda x: x.rolling(window=10).mean())
        player_stats['season_strikeouts'] = player_stats.groupby('player_name')['strikeouts'].transform(lambda x: x.cumsum())
        player_stats = pd.merge(player_stats, pitcher_vs_batter_stats[['player_name', 'opposing_pitcher', 'strikeouts']], on=['player_name', 'opposing_pitcher'], how='left')
        player_stats['pitcher_strikeout_rate'] = player_stats['opposing_pitcher'].map(pitcher_vs_batter_stats.set_index('opposing_pitcher')['strikeouts_per_game'])
        player_stats['ballpark_strikeout_factor'] = player_stats['ballpark'].map(ballpark_data.set_index('ballpark')['strikeout_factor'])

        player_stats['recent_hits'] = player_stats.groupby('player_name')['hits'].transform(lambda x: x.rolling(window=10).mean())
        player_stats['season_hits'] = player_stats.groupby('player_name')['hits'].transform(lambda x: x.cumsum())
        player_stats = pd.merge(player_stats, pitcher_vs_batter_stats[['player_name', 'opposing_pitcher', 'hits']], on=['player_name', 'opposing_pitcher'], how='left')
        player_stats['pitcher_hit_rate'] = player_stats['opposing_pitcher'].map(pitcher_vs_batter_stats.set_index('opposing_pitcher')['hits_per_game'])
        player_stats['ballpark_hit_factor'] = player_stats['ballpark'].map(ballpark_data.set_index('ballpark')['hit_factor'])
    except Exception as e:
        print(f"Error in enhanced feature engineering: {e}")
    return player_stats

# Fetch Statcast data and apply necessary preprocessing
statcast_data = fetch_statcast_data()
statcast_data = get_required_statcast_data(statcast_data)
statcast_data = handle_missing_values(statcast_data)
statcast_data = remove_outliers(statcast_data)
statcast_data_cleaned_df = standardize_normalize_data(statcast_data)

# Placeholder: Assume pitcher_vs_batter_stats, ballpark_data, and weather_data are fetched similarly
pitcher_vs_batter_stats = pd.DataFrame()  # Replace with actual data fetching
ballpark_data = pd.DataFrame()            # Replace with actual data fetching
weather_data = pd.DataFrame()             # Replace with actual data fetching

# Perform enhanced feature engineering
player_stats = enhanced_feature_engineering_v2(statcast_data_cleaned_df, pitcher_vs_batter_stats, ballpark_data, weather_data)

# Fetching and processing Bovada data using BovadaScraper
bovada_scraper = BovadaScraper()
bovada_json = bovada_scraper.scrape_bovada()
player_props_markets = bovada_scraper.parse_json(bovada_json) if bovada_json else []
player_props_markets_df = pd.DataFrame(player_props_markets)

# Define targets
def define_targets(player_props_markets_df):
    try:
        player_props_markets_df['home_run_target'] = player_props_markets_df.apply(lambda x: 1 if 'Home Run' in [outcome['description'] for outcome in x['outcomes']] else 0, axis=1)
        player_props_markets_df['strikeout_target'] = player_props_markets_df.apply(lambda x: 1 if 'Strikeout' in [outcome['description'] for outcome in x['outcomes']] else 0, axis=1)
        player_props_markets_df['hits_target'] = player_props_markets_df.apply(lambda x: 1 if 'Hits' in [outcome['description'] for outcome in x['outcomes']] else 0, axis=1)
    except Exception as e:
        print(f"Error defining targets: {e}")
    return player_props_markets_df

player_props_markets_df = define_targets(player_props_markets_df)

# Merge player props targets with player stats with error handling
try:
    merged_data = pd.merge(player_props_markets_df, player_stats, on='player_name', how='left')
except Exception as e:
    print(f"Error merging player props with stats: {e}")

# Train and evaluate models with error handling
def train_and_evaluate_with_insights(merged_data, player_props_markets_df):
    try:
        def train_and_evaluate(X, y, model_name):
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                print(f'{model_name} Model Accuracy: {accuracy_score(y_test, y_pred)}')
                print(classification_report(y_test, y_pred))
                return model
            except Exception as e:
                print(f"Error training {model_name} model: {e}")

        # Example models
        X_hr = merged_data.drop(columns=['home_run_target'])
        y_hr = merged_data['home_run_target']
        model_hr = train_and_evaluate(X_hr, y_hr, 'Home Run')

        X_so = merged_data.drop(columns=['strikeout_target'])
        y_so = merged_data['strikeout_target']
        model_so = train_and_evaluate(X_so, y_so, 'Strikeout')

        X_hits = merged_data.drop(columns=['hits_target'])
        y_hits = merged_data['hits_target']
        model_hits = train_and_evaluate(X_hits, y_hits, 'Hits')

        return model_hr, model_so, model_hits

    except Exception as e:
        print(f"Error in model training and evaluation process: {e}")

# Train and evaluate the models
model_hr, model_so, model_hits = train_and_evaluate_with_insights(merged_data, player_props_markets_df)

# Save the models with error handling
try:
    joblib.dump(model_hr, 'model_hr.pkl')
    joblib.dump(model_so, 'model_so.pkl')
    joblib.dump(model_hits, 'model_hits.pkl')
    print("Models saved successfully.")
except Exception as e:
    print(f"Error saving models: {e}")

# Flask application to serve predictions
app = Flask(__name__)

# Load trained models with error handling
try:
    model_hr = joblib.load('model_hr.pkl')
    model_so = joblib.load('model_so.pkl')
    model_hits = joblib.load('model_hits.pkl')
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Model file not found. {e}")
except Exception as e:
    print(f"Error loading models: {e}")

# Function to generate reasoning based on feature importances
def generate_reasoning(model, feature_names):
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        top_features = importance_df.head(3)
        reasons = [f"{row['feature']} (importance: {row['importance']:.2f})" for _, row in top_features.iterrows()]
        return "Key factors: " + ", ".join(reasons)
    except Exception as e:
        return f"Error generating reasoning: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])

        # Predictions for each player prop market
        prediction_hr = model_hr.predict(df)
        prediction_so = model_so.predict(df)
        prediction_hits = model_hits.predict(df)

        # Generate reasoning for each prediction
        reasoning_hr = generate_reasoning(model_hr, df.columns)
        reasoning_so = generate_reasoning(model_so, df.columns)
        reasoning_hits = generate_reasoning(model_hits, df.columns)

        response = {
            'home_run_prediction': int(prediction_hr[0]),
            'home_run_reasoning': reasoning_hr,
            'strikeout_prediction': int(prediction_so[0]),
            'strikeout_reasoning': reasoning_so,
            'hits_prediction': int(prediction_hits[0]),
            'hits_reasoning': reasoning_hits
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)