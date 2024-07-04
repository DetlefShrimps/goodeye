import pandas as pd
from joblib import load
from pybaseball import statcast
from datetime import datetime, timedelta
import logging
from xgboost import XGBRegressor
import re

# Set up logging
logging.basicConfig(filename='/home/jesse/goodeye/logs/prediction_script.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_bovada_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Successfully loaded Bovada data from {file_path}')
        print("Bovada Data Columns:", df.columns.tolist())
        return df
    except Exception as e:
        logging.error(f"Error loading Bovada data {file_path}: {e}")
        return pd.DataFrame()

def load_latest_statcast_data():
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        data = statcast(start_dt=yesterday, end_dt=today)
        logging.info('Successfully loaded latest Statcast data')
        print("Statcast Data Columns:", data.columns.tolist())
        return data
    except Exception as e:
        logging.error(f"Error loading Statcast data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    bovada_data = load_bovada_data('/home/jesse/goodeye/api/bovada_data.csv')
    statcast_data = load_latest_statcast_data()
    # Print the first few rows to understand the structure
    print("Bovada Data Sample:\n", bovada_data.head())
    print("Statcast Data Sample:\n", statcast_data.head())
