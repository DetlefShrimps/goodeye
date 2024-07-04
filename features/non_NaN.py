import pandas as pd

# Load the datasets
batting_data_path = '/home/jesse/g00d3y3/data/adv_batting_data.csv'
pitching_data_path = '/home/jesse/g00d3y3/data/adv_pitching_data.csv'
fielding_data_path = '/home/jesse/g00d3y3/data/fielding_data.csv'
umpire_data_path = '/home/jesse/g00d3y3/data/Cleaned_Umpire_Scorecard.csv'

batting_data = pd.read_csv(batting_data_path)
pitching_data = pd.read_csv(pitching_data_path)
fielding_data = pd.read_csv(fielding_data_path)
umpire_data = pd.read_csv(umpire_data_path)

# Check which columns have non-NaN values
batting_non_nan = batting_data.notna().sum() > 0
pitching_non_nan = pitching_data.notna().sum() > 0
fielding_non_nan = fielding_data.notna().sum() > 0
umpire_non_nan = umpire_data.notna().sum() > 0

# Combine the checks
combined_non_nan = batting_non_nan.astype(int) + pitching_non_nan.astype(int) + fielding_non_nan.astype(int) + umpire_non_nan.astype(int)

# Find columns with non-NaN values in at least three groups
columns_with_non_nan_in_three_groups = combined_non_nan[combined_non_nan >= 3].index.tolist()
count_columns_with_non_nan_in_three_groups = len(columns_with_non_nan_in_three_groups)

print(count_columns_with_non_nan_in_three_groups)

