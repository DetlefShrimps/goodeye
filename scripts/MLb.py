import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
from tqdm import tqdm

# Load datasets
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return pd.DataFrame()

# Load datasets
pitching_data = load_dataset('/home/jesse/g00d3y3/data/pitching_data.csv')
batting_data = load_dataset('/home/jesse/g00d3y3/data/batting_data.csv')
fielding_data = load_dataset('/home/jesse/g00d3y3/data/fielding_data.csv')
umpire_data = load_dataset('/home/jesse/g00d3y3/data/umpire_scorecard.csv')

# Combine datasets
combined_data = pd.concat([pitching_data, batting_data, fielding_data, umpire_data], axis=0, ignore_index=True)

# Define target columns
target_outcome_score_home = 'post_home_score'
target_outcome_score_away = 'post_away_score'
target_at_bat_result = ['H', '2B', 'HR', 'RBI', 'SB', 'R', 'ER', 'SO', 'IP', 'BB']
target_player_game_prop = ['H', '2B', 'HR', 'RBI', 'SB', 'R', 'ER', 'SO', 'IP', 'BB']

# Check for target columns
required_targets = [target_outcome_score_home, target_outcome_score_away] + target_at_bat_result + target_player_game_prop
missing_targets = [col for col in required_targets if col not in combined_data.columns]
if missing_targets:
    print(f"Missing target columns: {missing_targets}")
else:
    # Select relevant columns
    columns_to_use = combined_data.columns.intersection([
        'player', 'player_name', 'game_type', 'game_pk', 'game_date', 'game_year',
        'batter', 'pitcher', 'KN-Z (pi)', 'Swing% (sc)', 'wCU (pi)', 'RA9-WAR',
        'rPM', 'vSC (sc)', 'XX% (pi)', 'GDP', 'MD', 'CS-X (pi)', 'CU-Z (sc)',
        'Med%+', 'RAR', 'H/9', 'Stf+ KC', 'RBI', 'SL%', 'xERA', 'Loc+ SL', '1B',
        'DPS', 'KN-X (sc)', 'total_run_impact', 'XX-X (pi)', 'DPF', 'exLI', 'Strikes',
        'botOvr FC', 'phLI', 'xFIP-', 'Stf+ CU', 'CU% (sc)', 'SI% (sc)', 'FBv',
        'vFS (pi)', 'FRM', 'SB-Z (pi)', '90-100%', '60-90%', 'wSB (pi)', 'wXX (pi)',
        'ShO', 'Hard%+', 'FS% (sc)', 'LD+%', 'wSL', 'FO% (sc)', 'SwStr%', 'wCB',
        'Loc+ FA', 'LA', 'Stf+ CH', 'Off', 'FA-X (pi)', 'Loc+ FS', 'wRC+', 'AB', 'TE',
        'wCT/C', 'DP', 'LOB%', 'wFC (sc)', 'vFS (sc)', 'Fld', 'botCmd KC', 'rSZ',
        'vFT (sc)', 'botCmd CH', 'botStf FC', 'botCmd FS', 'FC-X (pi)', 'Plays', 'wSF/C',
        'FIP', 'wFS/C (sc)', 'wKN', 'CS-Z (pi)', 'FC-Z (pi)', 'SF%', 'Barrel%', 'CU% (pi)',
        'wEP/C (sc)', 'WHIP+', 'botStf CH', 'GB', 'wSB/C (pi)', 'FA-Z (pi)', 'KC-Z (sc)',
        'CHv', 'OPS', 'GS', 'Cent%+', '40-60%', 'wRC', 'wFA/C (sc)', 'DRS', 'FB%+',
        'HR/FB%+', 'botOvr KC', 'PA', 'EP-Z (sc)', 'Stf+ FO', 'Spd', 'PB', 'FSR',
        'CT%', 'WAR', 'BS', 'Relieving', 'O-Contact% (pi)', 'Pulls', 'wSI (pi)', 'SLG',
        'wSL/C (sc)', 'WP', 'wSL/C', 'vFC (sc)', 'IDfg', 'rARM', 'Soft%+',
        'expected_accuracy', 'CH%', 'Swing%', 'wKN/C (sc)', 'CU-Z (pi)', 'Z-Contact%',
        'botOvr CU', 'Range', 'wSI (sc)', 'wSC (sc)', 'KN-Z (sc)', 'O-Contact% (sc)',
        'K/9', 'HardHit', 'Stf+ SL', 'Loc+ CU', 'RZR', 'botCmd FA', 'Pit+ FO',
        'O-Swing% (pi)', 'CH-X (pi)', 'wFC/C (pi)', 'O-Swing% (sc)', 'KN% (sc)',
        'xFIP', 'wFA/C (pi)', 'vCS (pi)', 'EV', 'LD%', 'Cent%', 'KN%', 'FT-Z (sc)',
        'Hard%', 'wSI/C (sc)', 'IFH==%', 'OAA', 'SH', 'vKN (sc)', 'botCmd CU', 'BsR',
        'UZR', 'KN% (pi)', 'K/BB', '0', 'kwERA', 'BB/9', 'CH% (pi)', 'FO-X (sc)',
        'LOB-Wins', 'SL% (sc)', 'correct_calls_above_expected', 'tERA', 'BIP-Wins', 'FP',
        'Start-IP', 'SF', 'Inn', 'LD%+', 'wSF', 'incorrect_calls', 'pitches_called',
        'FO-Z (sc)', 'CTv', 'wFB/C', 'IFFB', 'botOvr', 'expected_incorrect_calls',
        'botOvr FA', 'Z-Swing%', 'CU-X (pi)', 'FS-Z (pi)', 'wCB/C', 'CU-X (sc)', 'wKN/C',
        'BB', 'vFA (pi)', 'GB%+', 'wFO (sc)', 'SL-Z (sc)', 'maxEV', 'correct_calls',
        'botCmd SI', 'Loc+ CH', 'FA-X (sc)', 'FS-Z (sc)', 'PO', 'FA% (sc)', 'Pos',
        'home_team_runs', 'wXX/C (pi)', 'wFB', 'F-Strike%', 'away_team_runs',
        'consistency', 'AVG', 'botStf CU', 'RS/9', 'vSI (pi)', 'FC% (sc)', 'wEP (sc)',
        'ER', 'BABIP+', 'TBF', 'Pit+ FC', 'CS', 'Contact%', 'FC-Z (sc)', 'TTO%',
        'botOvr FS', '3B', 'K%', 'SIERA', 'KC% (sc)', 'FA-Z (sc)', 'botStf KC', 'OBP',
        'Starting', 'AVG+', 'botOvr SI', 'Pit+ CH', 'Stf+ FC', 'accuracy_above_expected',
        'wCH (pi)', 'wCH/C (sc)', 'OBP+', 'id', 'REW', 'botOvr SL', 'FS-X (sc)', 'wGDP',
        'wCH', 'Balls', 'wCH/C (pi)', 'wCU/C (sc)', 'vEP (sc)', 'Pitching+', 'CH-Z (pi)',
        'xwOBA', 'BIZ', 'botCmd FC', 'Swing% (pi)', 'Soft%', 'botCmd SL', 'wKC (sc)',
        'SV', 'G', 'FB', 'Pit+ CU', 'W', 'Oppo%+', 'wCS (pi)', 'FA% (pi)', 'R',
        'accuracy', 'LD', 'CH-Z (sc)', 'SB-X (pi)', 'O-Swing%', 'K/9+', 'wSB',
        'vKC (sc)', 'Z-Contact% (sc)', 'FE', 'WPA/LI', 'rSB', 'GB%', 'favor_home',
        'wRAA', 'HLD', 'HR', 'botERA', 'E-F', 'SL-X (pi)', 'ISO+', 'EP-X (sc)',
        'botStf SI', 'Pit+ FA', 'RngR', 'wCH (sc)', 'vSB (pi)', 'ISO', 'WPA', 'FB% 2',
        'wOBA', 'K/BB+', 'ARM', 'BK', 'CG', 'XX-Z (pi)', 'wSL (sc)', 'CSW%', 'FB%',
        'Z-Swing% (sc)', 'Barrels', 'Age', 'FB% (Pitch)', 'vFO (sc)',
        'K-BB%', 'HBP', 'Clutch', 'FC% (pi)', 'Pace', 'SI-X (sc)', 'Pit+ SL', 'SLv',
        'wFT (sc)', 'LOB%+', 'Pull%+', 'H/9+', 'Season', 'SI-X (pi)', 'SL-X (sc)',
        'HR/9', 'XX%', 'O-Contact%', 'wKC/C (sc)', 'E', 'ERA', 'OOZ', 'BUH%', 'H',
        'xBA', 'wCU (sc)', 'vSL (sc)', 'BB%', 'SI-Z (sc)', 'vSL (pi)', 'Contact% (pi)',
        'CH-X (sc)', 'pLI', 'BB%+', 'BB/K', 'Loc+ FC', 'Zone% (pi)', 'vCH (pi)',
        'HardHit%', 'SC% (sc)', 'wFS (sc)', 'IFFB%', 'rGFP', 'IFH', 'Pit+ SI',
        'Loc+ FO', 'vCU (sc)', 'A', 'KN-X (pi)', 'gmLI', 'Bat', 'wFT/C (sc)', 'botCmd',
        'wSL (pi)', 'SC-Z (sc)', 'Contact% (sc)', 'vXX (pi)', 'Lg', 'CB%', 'botStf FA',
        'Relief-IP', 'CH% (sc)', '2B', 'SC-X (sc)', '1-10%', 'Stf+ FS', 'Rep',
        'UZR/150', 'BU', 'wSL/C (pi)', 'vFA (sc)', 'wSI/C (pi)', 'Pitches', 'Pit+ FS',
        'rGDP', 'wCU/C (pi)', 'FS-X (pi)', 'WHIP', 'KC-X (sc)', 'HR/FB', 'DPR',
        'expected_correct_calls', 'wKN (sc)', 'GB/FB', 'wCT', 'SI% (pi)', 'L-WAR',
        'CBv', 'wFO/C (sc)', 'Events', 'PH', 'SD', 'FS% (pi)', 'Z-Swing% (pi)', 'SO',
        'vKN (pi)', 'botStf', 'wCH/C', 'KNv', 'Scp', 'K%+', 'FC-X (sc)', 'SI-Z (pi)',
        'Stf+ SI', 'DPT', 'ERA-', 'Z-Contact% (pi)', 'wSC/C (sc)', 'botxRV100',
        'botStf SL', 'wFA (pi)', 'SLG+', 'SL-Z (pi)', 'Med%', 'wFC (pi)', 'wKN/C (pi)',
        'wFC/C (sc)', 'BUH', 'Pull%', 'UBR', 'RS', 'EP% (sc)', 'Loc+ SI', 'FIP-',
        'Stf+ FA', 'RE24', 'SB% (pi)', 'wFS/C (pi)', 'SB', 'wFS (pi)', 'vCH (sc)',
        'CS% (pi)', 'IP', 'Stuff+', 'wCS/C (pi)', 'FT% (sc)', 'SFv', 'FT-X (sc)', 'IBB',
        'BABIP', 'xSLG', 'wKN (pi)', 'Oppo%', 'SL% (pi)', 'rCERA', 'Loc+ KC', 'Def',
        'FDP-Wins', 'wFA (sc)', 'vFC (pi)', '10-40%', 'Zone% (sc)', 'Zone%', 'CStr%',
        'ErrR', 'vSI (sc)', 'Location+', 'vCU (pi)', 'Pace (pi)', target_outcome_score_home,
        target_outcome_score_away] + target_at_bat_result + target_player_game_prop)
    
    data = combined_data[columns_to_use].copy()
    
    # Fill missing values
    data = data.fillna(0)
    
    # Prepare features and targets
    X = data.drop(columns=[target_outcome_score_home, target_outcome_score_away] + target_at_bat_result + target_player_game_prop)
    y_outcome_score_home = data[target_outcome_score_home] if target_outcome_score_home in data.columns else None
    y_outcome_score_away = data[target_outcome_score_away] if target_outcome_score_away in data.columns else None
    y_at_bat_result = data[target_at_bat_result] if all(col in data.columns for col in target_at_bat_result) else None
    y_player_game_prop = data[target_player_game_prop] if all(col in data.columns for col in target_player_game_prop) else None
    
    # Function to split data
    def split_data(X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state) if y is not None else (None, None, None, None)
    
    # Split data into training and testing sets
    X_train, X_test, y_train_home, y_test_home = split_data(X, y_outcome_score_home)
    _, _, y_train_away, y_test_away = split_data(X, y_outcome_score_away)
    _, _, y_train_at_bat, y_test_at_bat = split_data(X, y_at_bat_result)
    _, _, y_train_player, y_test_player = split_data(X, y_player_game_prop)
    
    # Initialize models
    xgb_model_home = XGBRegressor(objective='reg:squarederror')
    xgb_model_away = XGBRegressor(objective='reg:squarederror')
    xgb_model_at_bat = XGBRegressor(objective='reg:squarederror')
    xgb_models_player = [XGBRegressor(objective='reg:squarederror') for _ in target_player_game_prop]
    
    # Train models
    def train_model(model, X_train, y_train, model_name):
        try:
            if X_train is not None and y_train is not None:
                with tqdm(total=len(X_train), desc=f'Training {model_name}') as pbar:
                    model.fit(X_train, y_train)
                    pbar.update(len(X_train))
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    train_model(xgb_model_home, X_train, y_train_home, 'XGBoost Model Home')
    train_model(xgb_model_away, X_train, y_train_away, 'XGBoost Model Away')
    train_model(xgb_model_at_bat, X_train, y_train_at_bat, 'XGBoost Model At Bat')
    for i, target in enumerate(target_player_game_prop):
        if y_train_player is not None and target in y_train_player.columns:
            train_model(xgb_models_player[i], X_train, y_train_player[target], f'XGBoost Model Player {target}')
    
    # Evaluate models
    def evaluate_model(model, X_test, y_test, model_name):
        try:
            if X_test is not None and y_test is not None:
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                print(f"{model_name} - Mean Squared Error: {mse}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    evaluate_model(xgb_model_home, X_test, y_test_home, 'XGBoost Model Home')
    evaluate_model(xgb_model_away, X_test, y_test_away, 'XGBoost Model Away')
    evaluate_model(xgb_model_at_bat, X_test, y_test_at_bat, 'XGBoost Model At Bat')
    for i, target in enumerate(target_player_game_prop):
        if y_test_player is not None and target in y_test_player.columns:
            evaluate_model(xgb_models_player[i], X_test, y_test_player[target], f'XGBoost Model Player {target}')
    
    # Save models
    def save_model(model, filename):
        try:
            joblib.dump(model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving model {filename}: {e}")
    
    save_model(xgb_model_home, 'xgb_model_home.pkl')
    save_model(xgb_model_away, 'xgb_model_away.pkl')
    save_model(xgb_model_at_bat, 'xgb_model_at_bat.pkl')
    for i, target in enumerate(target_player_game_prop):
        save_model(xgb_models_player[i], f'xgb_model_player_{target}.pkl')
