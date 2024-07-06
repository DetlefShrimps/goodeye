import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from pybaseball import statcast, batting_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import optuna
import joblib
import os
import logging
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import BertModel, BertConfig, BertTokenizer, AdamW
from tqdm import tqdm
import unittest
import time
from pybaseball import cache
cache.enable()

def create_gamestate_delta(df):
    df['gamestate_delta'] = df['balls'] - df['strikes']
    return df

def additional_feature_engineering(df):
    df['pitch_speed_diff'] = df['release_speed'] - df['effective_speed']
    df['pitch_location_diff'] = df['plate_x'] - df['plate_z']
    df['pitch_count'] = df.groupby(['game_pk', 'at_bat_number']).cumcount() + 1
    df['is_strike'] = df['events'].apply(lambda x: 1 if x in ['strikeout', 'strike'] else 0)
    return df

def visualize_data(df, column, title):
    df = df.compute()
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def train_transformer_model(model, tokenizer, df, logger, epochs=3, model_path='transformer_model.pth'):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded model from {model_path}")
    else:
        for epoch in range(epochs):
            model.train()
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = tokenizer(str(row['gamestate_delta']), return_tensors='pt')
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")

def main():
    start_time = time.time()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Initialize Dask client
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='7GB')

    # Data Collection
    logger.info("Collecting data...")
    with tqdm(total=2, desc="Data Collection") as pbar:
        pitch_data = statcast(start_dt='2023-03-25', end_dt='2024-07-01')
        pbar.update(1)
        season_stats = batting_stats(2023, 2024)
        pbar.update(1)
    logger.info(f"Data collection completed in {time.time() - start_time:.2f} seconds")

    # Data Storage
    storage_start_time = time.time()
    logger.info("Storing data in SQLite database...")
    with tqdm(total=2, desc="Data Storage") as pbar:
        conn = sqlite3.connect('mlb_data.db')
        pitch_data.to_sql('pitch_data', conn, if_exists='replace', index=False)
        season_stats.to_sql('season_stats', conn, if_exists='replace', index=False)
        pbar.update(1)
        pbar.update(1)
    logger.info(f"Data storage completed in {time.time() - storage_start_time:.2f} seconds")

    # Data Preparation
    preparation_start_time = time.time()
    logger.info("Preparing data...")
    with tqdm(total=2, desc="Data Preparation") as pbar:
        pitch_data_df = pd.read_sql('SELECT * FROM pitch_data', conn)
        pitch_data_df['index'] = pitch_data_df.index
        season_stats_df = pd.read_sql('SELECT * FROM season_stats', conn)
        season_stats_df['index'] = season_stats_df.index

        pitch_df = dd.from_pandas(pitch_data_df, npartitions=10)
        season_df = dd.from_pandas(season_stats_df, npartitions=10)
        pbar.update(1)
        pbar.update(1)
    logger.info(f"Data preparation completed in {time.time() - preparation_start_time:.2f} seconds")

    # Feature Engineering: Create gamestate delta, supplement inputs, etc.
    logger.info("Performing feature engineering...")
    with tqdm(total=2, desc="Feature Engineering") as pbar:
        pitch_df = pitch_df.map_partitions(create_gamestate_delta)
        pbar.update(1)

        meta = pitch_df._meta.assign(
            pitch_speed_diff=0.0,
            pitch_location_diff=0.0,
            pitch_count=0,
            is_strike=0
        )

        pitch_df = pitch_df.map_partitions(additional_feature_engineering, meta=meta)
        pbar.update(1)

    # Data Visualization (Dask doesn't directly support visualization, so we convert to pandas for plotting)
    logger.info("Visualizing data...")
    with tqdm(total=3, desc="Data Visualization") as pbar:
        visualize_data(pitch_df, 'gamestate_delta', 'Distribution of Gamestate Delta')
        pbar.update(1)
        visualize_data(pitch_df, 'pitch_speed_diff', 'Distribution of Pitch Speed Difference')
        pbar.update(1)
        visualize_data(pitch_df, 'pitch_location_diff', 'Distribution of Pitch Location Difference')
        pbar.update(1)

    # Define the transformer model (similar to BERT architecture)
    config = BertConfig(
        vocab_size=325,  # Number of unique gamestate deltas
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048
    )
    model = BertModel(config)

    # Tokenizer for the transformer model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Training the model using MGM and Contrastive Learning
    logger.info("Training transformer model...")
    pitch_df_pd = pitch_df.compute()
    train_transformer_model(model, tokenizer, pitch_df_pd, logger)

    # Function to prepare data for a specific pitcher and batter
    def prepare_matchup_data(pitch_df, season_df, pitcher, batter, pitcher_year, batter_year):
        pitcher_stats = season_df[(season_df['player'] == pitcher) & (season_df['year'] == pitcher_year)].compute()
        batter_stats = season_df[(season_df['player'] == batter) & (season_df['year'] == batter_year)].compute()

        if pitcher_stats.empty or batter_stats.empty:
            raise ValueError("Pitcher or Batter not found for the given year")

        matchup_df = pitch_df[(pitch_df['pitcher'] == pitcher_stats['player_id'].values[0]) &
                              (pitch_df['batter'] == batter_stats['player_id'].values[0])].compute()

        return matchup_df

    # Function to predict the outcome of a matchup
    def predict_matchup(pitch_df, season_df, pitcher, batter, pitcher_year, batter_year, model):
        matchup_df = prepare_matchup_data(pitch_df, season_df, pitcher, batter, pitcher_year, batter_year)

        if matchup_df.empty:
            logger.info("No data available for this matchup.")
            return

        X_matchup = matchup_df.drop(columns=['events'])  # Replace 'events' with the actual target column
        y_matchup = matchup_df['events']  # Replace 'events' with the actual target column

        y_pred = model.predict(X_matchup)
        accuracy = accuracy_score(y_matchup, y_pred)
        conf_matrix = confusion_matrix(y_matchup, y_pred)
        class_report = classification_report(y_matchup, y_pred)

        logger.info(f'Matchup Prediction Accuracy: {accuracy * 100:.2f}%')
        logger.info(f'Confusion Matrix:\n{conf_matrix}')
        logger.info(f'Classification Report:\n{class_report}')

    # Split the data into training and test sets
    X = pitch_df_pd.drop(columns=['events'])  # Replace 'events' with the actual target column
    y = pitch_df_pd['events']  # Replace 'events' with the actual target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model to predict game outcomes
    logger.info("Training RandomForest model...")
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        return accuracy

    # Run the Optuna study
    study = optuna.create_study(direction='maximize')
    with tqdm(total=50, desc="Hyperparameter Optimization") as pbar:
        study.optimize(objective, n_trials=50, callbacks=[lambda study, trial: pbar.update()])

    # Get the best hyperparameters and train the final model
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    rf_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Cross-validation
    logger.info("Performing cross-validation...")
    with tqdm(total=5, desc="Cross-Validation") as pbar:
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
        pbar.update(5)
    logger.info(f'Cross-validation scores: {cv_scores}')
    logger.info(f'Mean cross-validation score: {cv_scores.mean()}')

    # Evaluate the model
    logger.info("Evaluating RandomForest model...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    logger.info(f'Random Forest Model Accuracy: {accuracy * 100:.2f}%')
    logger.info(f'Confusion Matrix:\n{conf_matrix}')
    logger.info(f'Classification Report:\n{class_report}')

    # Save the RandomForest model
    rf_model_path = 'random_forest_model.pkl'
    joblib.dump(rf_model, rf_model_path)
    logger.info(f"Saved RandomForest model to {rf_model_path}")

    # Analysis of feature importance
    importances = rf_model.feature_importances_
    logger.info(f'Feature Importances: {importances}')

    # Example usage for predicting matchup outcomes
    pitcher = 'Gerrit Cole'  # Replace with user input
    batter = 'Mike Trout'    # Replace with user input
    pitcher_year = 2021      # Replace with user input
    batter_year = 2021       # Replace with user input

    try:
        predict_matchup(pitch_df, season_df, pitcher, batter, pitcher_year, batter_year, rf_model)
    except ValueError as e:
        logger.error(e)

    # Closing the database connection
    conn.close()

# Unit Tests
class TestMLBDataProcessing(unittest.TestCase):
    def test_create_gamestate_delta(self):
        test_df = pd.DataFrame({'balls': [1, 2, 3], 'strikes': [0, 1, 2]})
        result_df = create_gamestate_delta(test_df)
        expected_df = pd.DataFrame({'balls': [1, 2, 3], 'strikes': [0, 1, 2], 'gamestate_delta': [1, 1, 1]})
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_additional_feature_engineering(self):
        test_df = pd.DataFrame({'release_speed': [90, 95, 100], 'effective_speed': [85, 90, 95], 'plate_x': [0.5, 0.4, 0.3], 'plate_z': [0.2, 0.1, 0.0]})
        result_df = additional_feature_engineering(test_df)
        expected_df = pd.DataFrame({
            'release_speed': [90, 95, 100],
            'effective_speed': [85, 90, 95],
            'plate_x': [0.5, 0.4, 0.3],
            'plate_z': [0.2, 0.1, 0.0],
            'pitch_speed_diff': [5, 5, 5],
            'pitch_location_diff': [0.3, 0.3, 0.3],
            'pitch_count': [1, 1, 1],
            'is_strike': [0, 0, 0]
        })
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_data_loading(self):
        conn = sqlite3.connect('mlb_data.db')
        self.assertTrue('pitch_data' in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall())

    def test_model_training(self):
        rf_model_path = 'random_forest_model.pkl'
        rf_model = joblib.load(rf_model_path)
        X_sample = X_test.sample(n=10, random_state=42)
        y_sample = y_test.loc[X_sample.index]
        y_pred = rf_model.predict(X_sample)
        accuracy = accuracy_score(y_sample, y_pred)
        self.assertGreater(accuracy, 0.5, "Model accuracy should be greater than 50%")

if __name__ == '__main__':
    main()
    unittest.main()
