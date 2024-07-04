# Display the first few rows of the cleaned datasets to review them
cleaned_data = {
    'fielding_data': fielding_data.head(),
    'team_pitching_data': team_pitching_data.head(),
    'team_batting_data': team_batting_data.head(),
    'adv_batting_data': adv_batting_data.head(),
    'pitching_data': pitching_data.head(),
    'adv_pitching_data': adv_pitching_data.head(),
    'batting_data': batting_data.head(),
    'umpire_scorecard': umpire_scorecard.head(),
    'team_fielding_data': team_fielding_data.head()
}

import ace_tools as tools
tools.display_dataframe_to_user(name="Cleaned Fielding Data", dataframe=fielding_data.head())
tools.display_dataframe_to_user(name="Cleaned Team Pitching Data", dataframe=team_pitching_data.head())
tools.display_dataframe_to_user(name="Cleaned Team Batting Data", dataframe=team_batting_data.head())
tools.display_dataframe_to_user(name="Cleaned Advanced Batting Data", dataframe=adv_batting_data.head())
tools.display_dataframe_to_user(name="Cleaned Pitching Data", dataframe=pitching_data.head())
tools.display_dataframe_to_user(name="Cleaned Advanced Pitching Data", dataframe=adv_pitching_data.head())
tools.display_dataframe_to_user(name="Cleaned Batting Data", dataframe=batting_data.head())
tools.display_dataframe_to_user(name="Cleaned Umpire Scorecard Data", dataframe=umpire_scorecard.head())
tools.display_dataframe_to_user(name="Cleaned Team Fielding Data", dataframe=team_fielding_data.head())

cleaned_data
