import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

cache.enable()

# Install necessary packages
os.system("pip3 install --upgrade pandas[complete] scikit-learn[complete] joblib[complete] tensorflow[complete] tensorRT[complete] xgboost[complete] bs4[complete] selenium[complete] numpy[complete] pybaseball[complete] tqdm[complete] python-mlb-statsapi[complete] --break-system-packages")

# Increase the recursion limit
sys.setrecursionlimit(3000)

# MLB Data fetching and saving functions
def get_real_time_statcast_data():
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    data = statcast(start_dt=yesterday, end_dt=today)
    return data

def save_data():
    data = get_real_time_statcast_data()
    if data.empty:
        logging.error("No data fetched from statcast.")
    else:
        data.to_csv('/kaggle/input/statcast-data-csv/data/baseball_savant.csv', index=False)
        data.to_excel('/kaggle/input/statcast-data-csv/data/baseball_savant.xlsx', index=False)
        print(f"Data saved at {datetime.now()}")

def job():
    save_data()

def start_schedule():
    schedule.every(15).seconds.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Bovada Scraper class
class BovadaScraper:
    def __init__(self):
        self.host = "https://www.bovada.lv"
        self.mlb_path = "/services/sports/event/v2/events/A/description/baseball/mlb"
        self.max_retries = 5
        self.retry_delay = 10
        self.headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Referer": "https://www.bovada.lv/"
        }

    def fetch_data(self, url):
        for attempt in range(self.max_retries):
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
                time.sleep(self.retry_delay)
        return None

    def parse_json(self, data):
        try:
            logging.debug("JSON data structure:\n" + json.dumps(data, indent=2))
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
                            'outcomes': [{'description': outcome['description'], 'price': outcome['price'], 'handicap': outcome['price'].get('handicap')} for outcome in market['outcomes']]
                        }
                        event_data['markets'].append(market_data)
                events.append(event_data)
            logging.info(f"Parsed {len(events)} events from JSON data")
            return events
        except Exception as e:
            logging.error(f"Error parsing JSON data: {e}")
            return None

# Dataset loading function
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return pd.DataFrame()

def main():
    # Load datasets
    pitching_data = load_dataset('/kaggle/input/statcast-data-csv/data/pitching_data.csv')
    batting_data = load_dataset('/kaggle/input/statcast-data-csv/data/batting_data.csv')
    fielding_data = load_dataset('/kaggle/input/statcast-data-csv/data/fielding_data.csv')
    umpire_data = load_dataset('/kaggle/input/statcast-data-csv/data/umpire_scorecard.csv')
    
    # Check if datasets are loaded properly
    if pitching_data.empty or batting_data.empty or fielding_data.empty or umpire_data.empty:
        raise ValueError("One or more datasets are empty. Please check your data sources.")

    # Combine datasets
    combined_data = pd.concat([pitching_data, batting_data, fielding_data, umpire_data], axis=0, ignore_index=True)
    
    if combined_data.empty:
        raise ValueError("Merged data is empty. Check your data merging and formulas.")
    
    # Proceed with the rest of the data processing and model training...
    print("Data merging successful. Proceeding with the next steps...")

if __name__ == "__main__":
    main()
