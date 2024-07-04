import pandas as pd

# Paths to the CSV files
batting_path = 'data/batting_common.csv'
adv_batting_path = 'data/adv_batting_common.csv'
pitching_path = 'data/pitching_common.csv'
adv_pitching_path = 'data/adv_pitching_common.csv'

# Load the datasets
batting_data = pd.read_csv(batting_path)
adv_batting_data = pd.read_csv(adv_batting_path)
pitching_data = pd.read_csv(pitching_path)
adv_pitching_data = pd.read_csv(adv_pitching_path)

# Extract data types of columns from each dataset
batting_data_dtypes = batting_data.dtypes
adv_batting_data_dtypes = adv_batting_data.dtypes
pitching_data_dtypes = pitching_data.dtypes
adv_pitching_data_dtypes = adv_pitching_data.dtypes

# Function to print the column names and their data types
def print_dtypes(data_dtypes, dataset_name):
    print(f"\n{dataset_name} Data Types:")
    for column, dtype in data_dtypes.items():
        print(f"{column}: {dtype}")

# Print data types for each dataset
print_dtypes(batting_data_dtypes, "Batting")
print_dtypes(adv_batting_data_dtypes, "Advanced Batting")
print_dtypes(pitching_data_dtypes, "Pitching")
print_dtypes(adv_pitching_data_dtypes, "Advanced Pitching")
