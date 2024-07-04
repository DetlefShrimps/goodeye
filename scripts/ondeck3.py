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
            logging.debug(f"Problematic JSON data: {json.dumps(data, indent=2)}")
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

    def save_data(self, events):
        df = pd.DataFrame(events)
        df.to_csv('data/mlb_bets.csv', index=False)
        logging.info("Data saved successfully to data/mlb_bets.csv")

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
            self.save_data(mlb_bets)
        else:
            logging.error("Failed to fetch and verify all required bets after multiple retries.")

def load_dataset(path):
    try:
        return pd.read_csv(path).sample(frac=0.1)
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        return None

def get_common_columns(df1, df2):
    if df1 is not None and df2 is not None:
        return set(df1.columns).intersection(set(df2.columns))
    return set()

def extract_non_nan_columns(data):
    non_nan_columns = [col for col in data.columns if data[col].notna().all()]
    return non_nan_columns

def extend_formula(formula, additional_features):
    def wrapper(df):
        valid_features = [feature for feature in additional_features if feature in df.columns]
        return formula(df) + sum(df[feature] for feature in valid_features)
    return wrapper

def apply_formulas(df, formulas):
    for target, formula in formulas.items():
        try:
            df[target] = formula(df)
        except KeyError as e:
            missing_feature = str(e).strip("'")
            similar_feature = find_similar_column(df, missing_feature)
            if similar_feature:
                print(f"Feature '{missing_feature}' not found. Using similar feature '{similar_feature}' instead.")
                df = df.rename(columns={similar_feature: missing_feature})
                df[target] = formula(df)
            else:
                print(f"Feature '{missing_feature}' not found and no similar feature available. Skipping formula for '{target}'.")
        except Exception as e:
            print(f"Unexpected error while applying formula for '{target}': {e}")
    return df

def find_similar_column(df, target):
    if target in df.columns:
        return target
    for col in df.columns:
        if target.lower() in col.lower():
            return col
    return None

<<<<<<< HEAD
=======
def clean_and_scale_data(df, columns):

    # Convert set to list
    columns_list = list(columns)
    
    # Separate numeric and non-numeric columns
    numeric_cols = df[columns_list].select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with all missing values
    numeric_cols = [col for col in numeric_cols if df[col].notna().sum() > 0]
    
    # Replace infinities with NaNs
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Impute missing values and scale features for numeric columns
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    # Fit and transform the numeric columns
    numeric_data = imputer.fit_transform(df[numeric_cols])
    numeric_data = scaler.fit_transform(numeric_data)
    
    # Check that the shape of the transformed data matches the original DataFrame
    if numeric_data.shape[1] != len(numeric_cols):
        raise ValueError(f"Shape of passed values is {numeric_data.shape}, indices imply {df[numeric_cols].shape}")
    
    # Create a DataFrame for the transformed numeric data
    numeric_df = pd.DataFrame(numeric_data, columns=numeric_cols, index=df.index)
    
    # Explicitly cast to a compatible dtype
    numeric_df = numeric_df.astype(float)
    
    # Update the original DataFrame with the transformed numeric data
    df.update(numeric_df)

    # Impute missing values and scale features
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    df[columns] = imputer.fit_transform(df[columns])
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler

def log_transform(x):
    return np.log1p(x) if x > 0 else 0

def total_hits_formula_improved(df):
    return (
        log_transform(df['H']) +
        (log_transform(df['pfx_z']) * log_transform(df['pfx_x']) if 'pfx_z' in df.columns and 'pfx_x' in df.columns else 0) +
        (log_transform(df['launch_angle']) * log_transform(df['launch_speed']) if 'launch_angle' in df.columns and 'launch_speed' in df.columns else 0) +
        log_transform(df['maxEV']) if 'maxEV' in df.columns else 0 +
        log_transform(df['HardHit%']) if 'HardHit%' in df.columns else 0 +
        log_transform(df['OBP']) if 'OBP' in df.columns else 0 +
        log_transform(df['SLG']) if 'SLG' in df.columns else 0 +
        log_transform(df['BABIP']) if 'BABIP' in df.columns else 0 +
        log_transform(df['wOBA']) if 'wOBA' in df.columns else 0
    )

def record_2_plus_rbis_formula_improved(df):
    return (
        log_transform(df['RBI']) +
        (log_transform(df['OBP']) * log_transform(df['SLG']) if 'OBP' in df.columns and 'SLG' in df.columns else 0) +
        log_transform(df['H']) if 'H' in df.columns else 0 +
        log_transform(df['BB']) if 'BB' in df.columns else 0 +
        log_transform(df['wOBA']) if 'wOBA' in df.columns else 0 +
        log_transform(df['wRAA']) if 'wRAA' in df.columns else 0
    )

>>>>>>> 3993e43 ( Please enter the commit message for your changes. Lines starting)
def main():
    # Load datasets
    batting_data = load_dataset('/home/jesse/g00d3y3/data/batting_common.csv')
    adv_batting_data = load_dataset('/home/jesse/g00d3y3/data/adv_batting_common.csv')
    pitching_data = load_dataset('/home/jesse/g00d3y3/data/pitching_common.csv')
    adv_pitching_data = load_dataset('/home/jesse/g00d3y3/data/adv_pitching_common.csv')

    # Check if datasets are loaded successfully
    datasets = [batting_data, adv_batting_data, pitching_data, adv_pitching_data]
    dataset_names = ['batting_data', 'adv_batting_data', 'pitching_data', 'adv_pitching_data']
    for name, dataset in zip(dataset_names, datasets):
        if dataset is not None:
            print(f"{name} loaded successfully with columns: {dataset.columns}")
        else:
            print(f"{name} could not be loaded.")

    if any(dataset is None for dataset in datasets):
        raise ValueError("One or more datasets could not be loaded. Exiting.")


# Get common columns for merging
    common_columns_batting = get_common_columns(batting_data, adv_batting_data)
    common_columns_pitching = get_common_columns(pitching_data, adv_pitching_data)
    print("Common columns for batting data merge:", common_columns_batting)
    print("Common columns for pitching data merge:", common_columns_pitching)

    if not common_columns_batting:
        raise ValueError("No common columns found for batting data merge. Exiting.")
    if not common_columns_pitching:
        raise ValueError("No common columns found for pitching data merge. Exiting.")

    # Convert common columns to float64 for consistency
    for col in common_columns_batting:
        batting_data[col] = batting_data[col].astype('float64')
        adv_batting_data[col] = adv_batting_data[col].astype('float64')
    for col in common_columns_pitching:
        pitching_data[col] = pitching_data[col].astype('float64')
        adv_pitching_data[col] = adv_pitching_data[col].astype('float64')

    # Merge datasets on common columns
    try:
        merged_batting_data = batting_data.merge(adv_batting_data, how='inner', on=list(common_columns_batting))
        merged_pitching_data = pitching_data.merge(adv_pitching_data, how='inner', on=list(common_columns_pitching))
        merged_data = merged_batting_data.merge(merged_pitching_data, how='inner', left_index=True, right_index=True)
        print("Datasets merged successfully.")
    except KeyError as e:
        print(f"KeyError during merging: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during merging: {e}")
        raise

    mlb_data = get_real_time_statcast_data()
    non_nan_columns = extract_non_nan_columns(mlb_data)
    print("Non-NaN columns from MLB data:", non_nan_columns)

    additional_features = [
        "CH%", "Contact% (sc)", "KN% (sc)", "Z-Swing% (pi)", "wCH (sc)", "LD", "RE24", "wKN/C", "CH-Z (pi)",
        "CHv", "FS% (pi)", "CH-X (sc)", "CH-X (pi)", "K%", "KN-X (sc)", "FT-X (sc)", "SF%", "SI-X (pi)", "SC-X (sc)",
        "FS-Z (pi)", "FA% (sc)", "IFH", "wSF", "SL-X (pi)", "EV", "KN%", "FA-Z (sc)", "FO-X (sc)", "-WPA", "Swing% (pi)",
        "AVG", "FO% (sc)", "wFC (sc)", "Z-Swing%", "HardHit%",
        "wKN", "FA-X (sc)", "Contact%", "HR/FB", "KNv", "Swing% (sc)",
        "wKN (pi)", "maxEV", "phLI", "wSB/C (pi)", "BB/K", "ISO", "xSLG", "Lg", "wSB", "xBA", "OBP", "OPS", "L-WAR", "Bat",
        "SF", "wRC", "SB-X (pi)", "ISO+", "UBR", "LD+%", "Pos", "SLG+", "BsR", "GDP", "Off", "xwOBA", "wOBA", "SB-Z (pi)",
        "wSB (pi)", "Spd", "wGDP", "SH", "FB% (Pitch)", "SB% (pi)", "wRAA", "Rep", "SLG", "wRC+", "OBP+", "vSB (pi)"
    ]

    for col in non_nan_columns:
        if col not in additional_features:
            additional_features.append(col)

    # Define formulas
    total_hits_formula = lambda df: (
        df['H'] +
        (df['pfx_z'] * df['pfx_x'] if 'pfx_z' in df.columns and 'pfx_x' in df.columns else 0) +
        (df['launch_angle'] * df['launch_speed'] if 'launch_angle' in df.columns and 'launch_speed' in df.columns else 0) +
        df['maxEV'] if 'maxEV' in df.columns else 0 +
        df['HardHit%'] if 'HardHit%' in df.columns else 0 +
        df['OBP'] if 'OBP' in df.columns else 0 +
        df['SLG'] if 'SLG' in df.columns else 0 +
        df['BABIP'] if 'BABIP' in df.columns else 0 +
        df['wOBA'] if 'wOBA' in df.columns else 0
    )
    first_home_run_formula = lambda df: (
        df['HR'] +
        (df['pfx_z'] * df['pfx_x'] if 'pfx_z' in df.columns and 'pfx_x' in df.columns else 0) +
        (df['launch_angle'] * df['launch_speed'] if 'launch_angle' in df.columns and 'launch_speed' in df.columns else 0) +
        df['maxEV'] if 'maxEV' in df.columns else 0 +
        df['HardHit%'] if 'HardHit%' in df.columns else 0 +
        df['ISO'] if 'ISO' in df.columns else 0 +
        df['SLG'] if 'SLG' in df.columns else 0 +
        df['FB%'] if 'FB%' in df.columns else 0
    )
    record_2_plus_rbis_formula = lambda df: (
        df['RBI'] +
        (df['OBP'] * df['SLG'] if 'OBP' in df.columns and 'SLG' in df.columns else 0) +
        df['H'] if 'H' in df.columns else 0 +
        df['BB'] if 'BB' in df.columns else 0 +
        df['wOBA'] if 'wOBA' in df.columns else 0 +
        df['wRAA'] if 'wRAA' in df.columns else 0
    )
    record_3_plus_rbis_formula = lambda df: (
        df['RBI'] +
        (df['OBP'] * df['SLG'] if 'OBP' in df.columns and 'SLG' in df.columns else 0) +
        df['H'] if 'H' in df.columns else 0 +
        df['BB'] if 'BB' in df.columns else 0 +
        df['wOBA'] if 'wOBA' in df.columns else 0 +
        df['wRAA'] if 'wRAA' in df.columns else 0
    )
    total_strikeouts_formula = lambda df: (
        df['SO'] +
        (df['K%'] * df['pfx_z'] * df['pfx_x'] if 'K%' in df.columns and 'pfx_z' in df.columns and 'pfx_x' in df.columns else 0) +
        df['IP'] if 'IP' in df.columns else 0 +
        df['Swing%'] if 'Swing%' in df.columns else 0 +
        df['Contact%'] if 'Contact%' in df.columns else 0 +
        df['Zone%'] if 'Zone%' in df.columns else 0
    )
    record_single_formula = lambda df: (
        df['1B'] +
        (df['pfx_z'] * df['pfx_x'] if 'pfx_z' in df.columns and 'pfx_x' in df.columns else 0) +
        df['H'] if 'H' in df.columns else 0 +
        df['BABIP'] if 'BABIP' in df.columns else 0 +
        df['OBP'] if 'OBP' in df.columns else 0
    )

    total_hits = extend_formula(total_hits_formula, additional_features)
    first_home_run = extend_formula(first_home_run_formula, additional_features)
    record_2_plus_rbis = extend_formula(record_2_plus_rbis_formula, additional_features)
    record_3_plus_rbis = extend_formula(record_3_plus_rbis_formula, additional_features)
    total_strikeouts = extend_formula(total_strikeouts_formula, additional_features)
    record_single = extend_formula(record_single_formula, additional_features)

    formulas = {
        'total_hits': total_hits,
        'first_home_run': first_home_run,
        'record_2_plus_rbis': record_2_plus_rbis,
        'record_3_plus_rbis': record_3_plus_rbis,
        'total_strikeouts': total_strikeouts,
        'record_single': record_single
    }

    merged_data = apply_formulas(merged_data, formulas)
    print(f"Merged data has {len(merged_data)} rows.")

    if len(merged_data) == 0:
        raise ValueError("Merged data is empty. Check your data merging and formulas.")

    X = merged_data.drop(columns=list(formulas.keys()))
    y = merged_data[list(formulas.keys())]

    if X.empty or y.empty:
        raise ValueError("Feature matrix X or target matrix y is empty after splitting.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"RandomForestRegressor MSE: {mean_squared_error(y_test, y_pred_rf)}")

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    print(f"GradientBoostingRegressor MSE: {mean_squared_error(y_test, y_pred_gb)}")

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(formulas), activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    y_pred_nn = model.predict(X_test)
    nn_mse = mean_squared_error(y_test, y_pred_nn)
    print(f"Neural Network MSE: {nn_mse}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/random_forest_regressor.pkl')
    joblib.dump(gb, 'models/gradient_boosting_regressor.pkl')
    model.save('models/neural_network_model.h5')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Models and scaler saved successfully.")

def load_models():
    rf = joblib.load('models/random_forest_regressor.pkl')
    gb = joblib.load('models/gradient_boosting_regressor.pkl')
    model = tf.keras.models.load_model('models/neural_network_model.h5')
    scaler = joblib.load('models/scaler.pkl')
    return rf, gb, model, scaler

def make_predictions(new_data):
    rf, gb, model, scaler = load_models()
    new_data_scaled = scaler.transform(new_data)
    rf_pred = rf.predict(new_data_scaled)
    gb_pred = gb.predict(new_data_scaled)
    nn_pred = model.predict(new_data_scaled)
    return rf_pred, gb_pred, nn_pred

if __name__ == "__main__":
    daemon_thread = threading.Thread(target=start_schedule)
    daemon_thread.daemon = True
    daemon_thread.start()
    scraper = BovadaScraper()
    scraper.scrape_bovada()
    main()
    # Example usage of prediction function
    # Assuming new_data is a DataFrame with the same structure as the training data
<<<<<<< HEAD
=======
    # new_data = pd.DataFrame(...)  # Replace with actual data
    # rf_pred, gb_pred, nn_pred = make_predictions(new_data)
    # print(f"Random Forest Predictions: {rf_pred}")
    # print(f"Gradient Boosting Predictions: {gb_pred}")
    # print(f"Neural Network Predictions: {nn_pred}")
>>>>>>> 3993e43 ( Please enter the commit message for your changes. Lines starting)
