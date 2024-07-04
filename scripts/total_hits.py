import pandas as pd
from joblib import load
from pybaseball import statcast
from datetime import datetime, timedelta
import logging
from xgboost import XGBRegressor
import re
import ast

# Set up logging
logging.basicConfig(filename='/home/jesse/goodeye/logs/prediction_script.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_bovada_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Successfully loaded Bovada data from {file_path}')
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
        return data
    except Exception as e:
        logging.error(f"Error loading Statcast data: {e}")
        return pd.DataFrame()

def load_model(file_path):
    try:
        model = load(file_path)
        logging.info(f'Successfully loaded model from {file_path}')
        return model
    except Exception as e:
        logging.error(f"Error loading model {file_path}: {e}")
        return None

def extract_player_and_team(bet_name):
    try:
        match = re.search(r" - (.*?) \((.*?)\)", bet_name)
        if match:
            player_name, team_abbr = match.groups()
            return player_name, team_abbr
        return None, None
    except Exception as e:
        logging.error(f"Error extracting player and team from bet name '{bet_name}': {e}")
        return None, None

def prepare_bovada_data(bovada_data):
    try:
        player_names = []
        team_abbrs = []
        for idx, row in bovada_data.iterrows():
            try:
                markets = ast.literal_eval(row['markets'])
                logging.info(f"Processing markets: {markets}")
                for market in markets:
                    if 'Total Hits' in market['market_key']:
                        for outcome in market['outcomes']:
                            player_name, team_abbr = extract_player_and_team(outcome['description'])
                            logging.info(f"Extracted player_name: {player_name}, team_abbr: {team_abbr} from bet name: {outcome['description']}")
                            if player_name and team_abbr:
                                player_names.append(player_name)
                                team_abbrs.append(team_abbr)
            except Exception as e:
                logging.error(f"Error parsing markets: {e}")
        
        bovada_prepared = pd.DataFrame({'player_name': player_names, 'team_abbr': team_abbrs})
        # Ensure player_name and team_abbr are strings
        bovada_prepared['player_name'] = bovada_prepared['player_name'].astype(str).str.strip()
        bovada_prepared['team_abbr'] = bovada_prepared['team_abbr'].astype(str).str.strip()
        logging.info('Successfully prepared Bovada data')
        logging.info(f"Sample prepared Bovada data:\n{bovada_prepared.head()}")
        return bovada_prepared
    except Exception as e:
        logging.error(f"Error preparing Bovada data: {e}")
        return pd.DataFrame()

def merge_data(bovada_data, statcast_data):
    try:
        # Ensure player_name and team_abbr are strings
        statcast_data['player_name'] = statcast_data['player_name'].astype(str).str.strip()
        statcast_data['home_team'] = statcast_data['home_team'].astype(str).str.strip()
        statcast_data['away_team'] = statcast_data['away_team'].astype(str).str.strip()
        
        statcast_data['team_abbr'] = statcast_data.apply(
            lambda row: row['home_team'] if row['home_team'] in bovada_data['team_abbr'].values else row['away_team'], axis=1
        )
        
        statcast_data['team_abbr'] = statcast_data['team_abbr'].astype(str).str.strip()

        # Debug logging for sample data
        logging.info(f"Sample Bovada data:\n{bovada_data.head()}")
        logging.info(f"Sample Statcast data:\n{statcast_data[['player_name', 'team_abbr']].head()}")

        combined_data = pd.merge(bovada_data, statcast_data, on=['player_name', 'team_abbr'], how='inner')
        logging.info('Successfully merged Bovada and Statcast data')
        logging.info(f"Sample merged data:\n{combined_data.head()}")
        return combined_data
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        return pd.DataFrame()

def prepare_features(combined_data, model_features):
    try:
        X = combined_data.copy()
        missing_cols = set(model_features) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[model_features]
        X = X.fillna(0)
        logging.info('Successfully prepared features for prediction')
        return X
    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        return pd.DataFrame()

def make_predictions(model, X):
    try:
        predictions = model.predict(X)
        logging.info(f'Successfully made predictions with model {model}')
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions with model {model}: {e}")
        return []

if __name__ == "__main__":
    try:
        bovada_data = load_bovada_data('/home/jesse/goodeye/api/bovada_data.csv')
        bovada_prepared = prepare_bovada_data(bovada_data)

        statcast_data = load_latest_statcast_data()

        combined_data = merge_data(bovada_prepared, statcast_data)
        
        if combined_data.empty:
            logging.critical("Merged data is empty. Check if 'player_name' and 'team_abbr' columns are present in both datasets.")
            raise ValueError("Merged data is empty")
        
        # Load pre-trained model for predicting hits
        model_hits = load_model('/home/jesse/goodeye/models/xgb_model_player_H.pkl')
        
        if model_hits is None:
            logging.critical("Model could not be loaded. Ensure the file path is correct.")
            raise FileNotFoundError("Model file not found")
        
        # Example of using the model to get the expected features
        model_features = model_hits.get_booster().feature_names if isinstance(model_hits, XGBRegressor) else model_hits.coef_
        X = prepare_features(combined_data, model_features)
        
        # Make predictions for hits
        predictions_hits = make_predictions(model_hits, X)
        
        # Print or save the hit predictions
        combined_data['predicted_hits'] = predictions_hits
        combined_data.to_csv('/home/jesse/goodeye/data/predicted_hits.csv', index=False)
        logging.info('Script completed successfully')
        print(combined_data[['player_name', 'team_abbr', 'predicted_hits']])
        
    except Exception as e:
        logging.critical(f"Critical error in the script: {e}")
