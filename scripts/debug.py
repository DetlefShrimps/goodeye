import pandas as pd
import re

# Function to extract player name and team abbreviation
def extract_player_team(description):
    player_name_match = re.findall(r'-\s(.*?)\s\(', description)
    team_abbr_match = re.findall(r'\((.*?)\)', description)
    player_name = player_name_match[0] if player_name_match else None
    team_abbr = team_abbr_match[0] if team_abbr_match else None
    return player_name, team_abbr

# Load Bovada data
try:
    bovada_data = pd.read_csv('/home/jesse/goodeye/api/bovada_data.csv')
    print("Successfully loaded Bovada data")
except Exception as e:
    print(f"Error loading Bovada data: {e}")

# Prepare Bovada data
try:
    bovada_data[['player_name', 'team_abbr']] = bovada_data['description'].apply(lambda x: pd.Series(extract_player_team(x)))
    bovada_data = bovada_data[['player_name', 'team_abbr']]
    print("Successfully prepared Bovada data")
except Exception as e:
    print(f"Error preparing Bovada data: {e}")

# Load Statcast data
try:
    statcast_data = pd.read_csv('/home/jesse/goodeye/data/batting_data.csv')
    print("Successfully loaded latest Statcast data")
except Exception as e:
    print(f"Error loading Statcast data: {e}")

# Ensure correct types for merging
try:
    bovada_data['player_name'] = bovada_data['player_name'].astype(str)
    bovada_data['team_abbr'] = bovada_data['team_abbr'].astype(str)
    statcast_data['player_name'] = statcast_data['player_name'].astype(str)
    statcast_data['team_abbr'] = statcast_data['team_abbr'].astype(str)
    print("Successfully converted column types for merging")
except Exception as e:
    print(f"Error converting column types: {e}")

# Merge data
try:
    merged_data = pd.merge(bovada_data, statcast_data, on=['player_name', 'team_abbr'])
    print("Successfully merged Bovada and Statcast data")
    print(f"Sample merged data:\n{merged_data.head()}")
except Exception as e:
    print(f"Error merging data: {e}")

# Save merged data
try:
    merged_data.to_csv('/home/jesse/goodeye/data/merged_data.csv', index=False)
    print("Successfully saved merged data")
except Exception as e:
    print(f"Error saving merged data: {e}")
