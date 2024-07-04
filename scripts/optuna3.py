# Install missing packages
!pip install dask dask-ml scikit-learn optuna tqdm joblib

# Import necessary libraries
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import optuna
from tqdm.dask import TqdmCallback
import pandas as pd
import numpy as np
import joblib
import os

def load_dataset(path):
    try:
        return dd.read_csv(path)
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        return None

# Load datasets from Kaggle input directory
batting_data = load_dataset('/kaggle/input/statcast-data-csv/data/batting_data.csv')
adv_batting_data = load_dataset('/kaggle/input/statcast-data-csv/data/adv_batting_data.csv')
pitching_data = load_dataset('/kaggle/input/statcast-data-csv/data/pitching_data.csv')
adv_pitching_data = load_dataset('/kaggle/input/statcast-data-csv/data/adv_pitching_data.csv')

# Check if datasets are loaded successfully
datasets = [batting_data, adv_batting_data, pitching_data, adv_pitching_data]
dataset_names = ['batting_data', 'adv_batting_data', 'pitching_data', 'adv_pitching_data']

for name, dataset in zip(dataset_names, datasets):
    if dataset is not None:
        print(f"{name} loaded successfully with columns: {dataset.columns}")
    else:
        print(f"{name} could not be loaded.")

# Ensure all necessary datasets are loaded
if any(dataset is None for dataset in datasets):
    raise ValueError("One or more datasets could not be loaded. Exiting.")

# Inspect columns and find common columns for merging
def get_common_columns(df1, df2):
    if df1 is not None and df2 is not None:
        return set(df1.columns).intersection(set(df2.columns))
    return set()

common_columns_batting = get_common_columns(batting_data, adv_batting_data)
common_columns_pitching = get_common_columns(pitching_data, adv_pitching_data)

print("Common columns for batting data merge:", common_columns_batting)
print("Common columns for pitching data merge:", common_columns_pitching)

# Ensure common columns exist before merging
if not common_columns_batting:
    raise ValueError("No common columns found for batting data merge. Exiting.")

if not common_columns_pitching:
    raise ValueError("No common columns found for pitching data merge. Exiting.")

# Merge datasets
try:
    merged_data = batting_data.merge(adv_batting_data, how='inner', on=list(common_columns_batting))
    merged_data = merged_data.merge(pitching_data, how='inner', on=list(common_columns_pitching))
    merged_data = merged_data.merge(adv_pitching_data, how='inner', on=list(common_columns_pitching))
    print("Datasets merged successfully.")
except KeyError as e:
    print(f"KeyError during merging: {e}")
    raise
except Exception as e:
    print(f"Unexpected error during merging: {e}")
    raise

# Select all non-NaN columns
all_non_nan_cols = list(set(batting_data.columns.tolist() + adv_batting_data.columns.tolist() + pitching_data.columns.tolist() + adv_pitching_data.columns.tolist()))

# Load features from text file
text_file_path = '/kaggle/input/additional/features2.txt'
try:
    with open(text_file_path, 'r') as file:
        features_list = [line.strip() for line in file.readlines()]
    print("Features list loaded successfully.")
except Exception as e:
    print(f"Error loading features list from {text_file_path}: {e}")
    features_list = []

# Filter out already used features
existing_features = set(all_non_nan_cols)
unique_additional_features = [feature for feature in features_list if feature not in existing_features]

# Combine all non-NaN columns and unique additional features
all_non_nan_cols_extended = list(set(all_non_nan_cols + unique_additional_features))

# Filter merged_data to only include non-NaN columns and additional features
merged_data = merged_data[all_non_nan_cols_extended].dropna()

# Define target columns
target_columns = {
    'total_hits': 'H',
    'first_home_run': 'HR',
    'record_2_plus_rbis': 'RBI',
    'record_3_plus_rbis': 'RBI',
    'total_strikeouts': 'SO',
    'record_single': 'H',
}

# Function to create features and targets
def create_features_and_target(data, target):
    try:
        X = data.drop(columns=[target])
        y = data[target]
        return X, y
    except KeyError as e:
        print(f"KeyError creating features and target for {target}: {e}")
        return None, None

# Function to train and evaluate model for each target
def train_and_evaluate_model(target):
    X, y = create_features_and_target(merged_data, target)
    if X is None or y is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature Selection using Feature Importance
    initial_model = RandomForestRegressor(n_estimators=100, random_state=42)
    initial_model.fit(X_train, y_train)
    importances = initial_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = [X.columns[i] for i in indices[:10]]  # Select top 10 features
    X_train = X_train[:, indices[:10]]
    X_test = X_test[:, indices[:10]]

    # List selected features
    print(f"Selected features for {target}: {selected_features}")

    # Define objective function for Optuna with extended hyperparameter tuning
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        return -scores.mean()

    # Optuna study
    study = optuna.create_study(direction='minimize')
    with TqdmCallback() as tqdm_callback:
        study.optimize(objective, n_trials=50, callbacks=[tqdm_callback])

    # Best parameters
    best_params = study.best_params
    print(f"Best parameters for {target}: {best_params}")

    # Train final model
    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {target}: {mse}")

    # Save the model
    model_filename = f'final_model_{target}.pkl'
    joblib.dump(final_model, model_filename)

# Error handling
try:
    for target in target_columns.values():
        print(f"Processing target: {target}")
        model_filename = f'final_model_{target}.pkl'
        if os.path.exists(model_filename):
            print(f"Loading existing model for {target}")
            final_model = joblib.load(model_filename)
        else:
            train_and_evaluate_model(target)
except Exception as e:
    print(f"An error occurred: {e}")
