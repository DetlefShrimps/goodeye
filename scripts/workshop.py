import pandas as pd
from joblib import load
from pybaseball import statcast
from datetime import datetime, timedelta
import logging
from xgboost import XGBRegressor

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

def merge_data(bovada_data, statcast_data):
    try:
        combined_data = pd.merge(bovada_data, statcast_data, on=['player', 'team'], how='inner')
        logging.info('Successfully merged Bovada and Statcast data')
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

def make_bet_selections(predictions_home_score, predictions_away_score, predictions_at_bat, predictions_player_props, combined_data):
    try:
        bets = []
        for i in range(len(predictions_home_score)):
            if predictions_home_score[i] > predictions_away_score[i]:
                bets.append({'bet_type': 'win', 'team': combined_data['home_team'][i]})
            # Add logic for player props, at-bat results, etc.
        logging.info('Successfully made bet selections')
        return bets
    except Exception as e:
        logging.error(f"Error making bet selections: {e}")
        return []

if __name__ == "__main__":
    try:
        bovada_data = load_bovada_data('/home/jesse/goodeye/api/bovada_data.csv')
        statcast_data = load_latest_statcast_data()
        combined_data = merge_data(bovada_data, statcast_data)

        # Load pre-trained models and their expected features
        models = {
            'linear_model_home': load_model('/home/jesse/goodeye/models/linear_model_home.pkl'),
            'linear_model_away': load_model('/home/jesse/goodeye/models/linear_model_away.pkl'),
            'xgb_model_home': load_model('/home/jesse/goodeye/models/xgb_model_home.pkl'),
            'xgb_model_away': load_model('/home/jesse/goodeye/models/xgb_model_away.pkl'),
            'xgb_model_at_bat': load_model('/home/jesse/goodeye/models/xgb_model_at_bat.pkl'),
        }
        xgb_models_player = [
            load_model(f'/home/jesse/goodeye/models/xgb_model_player_{target}.pkl')
            for target in ['H', '2B', 'HR', 'RBI', 'SB', 'R', 'ER', 'SO', 'IP', 'BB']
        ]
        models.update({f'xgb_model_player_{target}': model for target, model in zip(['H', '2B', 'HR', 'RBI', 'SB', 'R', 'ER', 'SO', 'IP', 'BB'], xgb_models_player)})

        # Example of using one model to get the expected features
        example_model = models['xgb_model_home']
        model_features = example_model.get_booster().feature_names if isinstance(example_model, XGBRegressor) else example_model.coef_

        # Ensure all columns are present at once using pd.concat
        missing_cols = set(model_features) - set(combined_data.columns)
        missing_df = pd.DataFrame(0, index=combined_data.index, columns=missing_cols)
        combined_data = pd.concat([combined_data, missing_df], axis=1)
        X = combined_data[model_features]

        # Fill any missing values
        X = X.fillna(0)

        # Make predictions
        predictions = {
            'home_score': make_predictions(models['linear_model_home'], X),
            'away_score': make_predictions(models['linear_model_away'], X),
            'at_bat': make_predictions(models['xgb_model_at_bat'], X),
            'player_props': [make_predictions(models[f'xgb_model_player_{target}'], X) for target in ['H', '2B', 'HR', 'RBI', 'SB', 'R', 'ER', 'SO', 'IP', 'BB']]
        }

        # Make bet selections
        bets = make_bet_selections(predictions['home_score'], predictions['away_score'], predictions['at_bat'], predictions['player_props'], combined_data)

        # Print or save the bet selections
        print(bets)
        logging.info('Script completed successfully')
    except Exception as e:
        logging.critical(f"Critical error in the script: {e}")
