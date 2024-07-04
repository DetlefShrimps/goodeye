import pandas as pd
from pybaseball import (
    batting_stats, pitching_stats, standings, team_batting, team_pitching,
    top_prospects, batting_stats_bref, batting_stats_range, bwar_bat, 
    bwar_pitch, chadwick_register, lahman, playerid_lookup, retrosheet, 
    schedule_and_record, split_stats, statcast, statcast_batter, 
    statcast_fielding, statcast_pitcher, statcast_pitcher_spin, 
    statcast_running, statcast_single_game, team_fielding, team_fielding_bref,
    team_game_logs, teamid_lookup, cache
)
from tqdm import tqdm

# Enable caching
cache.enable()

# Function to save DataFrame to both CSV and Excel
def save_data(df, filename):
    try:
        csv_path = f"{filename}.csv"
        xlsx_path = f"{filename}.xlsx"
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        # Save to Excel
        df.to_excel(xlsx_path, index=False)
        
        print(f"Data successfully saved to {csv_path} and {xlsx_path}")
    except Exception as e:
        print(f"Failed to save data for {filename}: {e}")

# Parameters
start_season = 1871  # The earliest available year for most baseball stats
end_season = 2023
statcast_start_date = '2015-01-01'  # Statcast data available from 2015
statcast_end_date = '2023-12-31'

# List of data retrieval functions and their corresponding file names
data_functions = [
    (batting_stats, [start_season, end_season, 'all', 1, 1], 'batting_stats'),
    (pitching_stats, [start_season, end_season, 'all', 1, 1], 'pitching_stats'),
    (standings, [end_season], 'standings'),
    (team_batting, [start_season, end_season, 'all', 1], 'team_batting'),
    (team_pitching, [start_season, end_season, 'all', 1], 'team_pitching'),
    (top_prospects, [], 'top_prospects'),
    (batting_stats_bref, [2008], 'batting_stats_bref'),
    (batting_stats_range, ["2008-01-01", "2023-12-31"], 'batting_stats_range'),
    (bwar_bat, [], 'bwar_bat'),
    (bwar_pitch, [], 'bwar_pitch'),
    (chadwick_register, [], 'chadwick_register'),
    (lahman.people, [], 'people'),
    (playerid_lookup, ["trout", "mike"], 'playerid_lookup'),
    (retrosheet.season_game_logs, [1871], 'season_game_logs'),
    (schedule_and_record, [1871, "PHI"], 'schedule_and_record'),
    (split_stats, ['troutmi01'], 'split_stats'),
    (statcast, [statcast_start_date, statcast_end_date], 'statcast_data'),
    (statcast_batter, [statcast_start_date, statcast_end_date, 514888], 'statcast_batter'),
    (statcast_fielding, [statcast_start_date, statcast_end_date], 'statcast_fielding'),
    (statcast_pitcher, [statcast_start_date, statcast_end_date, 514888], 'statcast_pitcher'),
    (statcast_pitcher_spin, [statcast_start_date, statcast_end_date], 'statcast_pitcher_spin'),
    (statcast_running, [statcast_start_date, statcast_end_date], 'statcast_running'),
    (statcast_single_game, [statcast_start_date, statcast_end_date, '514888'], 'statcast_single_game'),
    (team_fielding, [start_season, end_season, 'all', 1], 'team_fielding'),
    (team_fielding_bref, [start_season, end_season, 'all', 1], 'team_fielding_bref'),
    (team_game_logs, [start_season, end_season, 'all', 1], 'team_game_logs'),
    (teamid_lookup, ["Yankees"], 'teamid_lookup'),
]

# Loop through each function and save its data
for func, args, filename in tqdm(data_functions):
    try:
        data = func(*args)
        
        # Handle cases where the returned data is not a DataFrame
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        save_data(data, filename)
    except Exception as e:
        print(f"Failed to retrieve data for {filename}: {e}")
