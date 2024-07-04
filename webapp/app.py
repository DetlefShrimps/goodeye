from flask import Flask, render_template, request, jsonify
import sqlite3
import joblib
import pandas as pd

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('mlb_data.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    pitchers = conn.execute('SELECT DISTINCT player_name FROM season_stats').fetchall()
    batters = conn.execute('SELECT DISTINCT player_name FROM season_stats').fetchall()
    conn.close()
    return render_template('index.html', pitchers=pitchers, batters=batters)

@app.route('/get_years', methods=['POST'])
def get_years():
    player_name = request.form['player_name']
    conn = get_db_connection()
    years = conn.execute('SELECT DISTINCT game_year FROM season_stats WHERE player_name = ?', (player_name,)).fetchall()
    conn.close()
    return jsonify([year['game_year'] for year in years])

@app.route('/predict', methods=['POST'])
def predict():
    pitcher = request.form['pitcher']
    batter = request.form['batter']
    pitcher_year = request.form['pitcher_year']
    batter_year = request.form['batter_year']

    model = joblib.load('random_forest_model.pkl')
    conn = get_db_connection()
    pitch_df = pd.read_sql('SELECT * FROM pitch_data', conn)
    season_df = pd.read_sql('SELECT * FROM season_stats', conn)
    conn.close()

    # Implementing prediction logic here using the loaded model and data
    def prepare_matchup_data(pitch_df, season_df, pitcher, batter, pitcher_year, batter_year):
        pitcher_stats = season_df[(season_df['player_name'] == pitcher) & (season_df['game_year'] == pitcher_year)]
        batter_stats = season_df[(season_df['player_name'] == batter) & (season_df['game_year'] == batter_year)]

        if pitcher_stats.empty or batter_stats.empty:
            raise ValueError("Pitcher or Batter not found for the given year")

        matchup_df = pitch_df[(pitch_df['pitcher'] == pitcher_stats['player_id'].values[0]) &
                              (pitch_df['batter'] == batter_stats['player_id'].values[0])]

        return matchup_df

    try:
        matchup_df = prepare_matchup_data(pitch_df, season_df, pitcher, batter, pitcher_year, batter_year)
        if matchup_df.empty:
            return jsonify({'error': 'No data available for this matchup'})

        X_matchup = matchup_df.drop(columns=['events'])  # Replace 'events' with the actual target column
        y_pred = model.predict(X_matchup)
        result = y_pred[0]  # This will be adjusted based on the model's output

        return jsonify({'result': result})
    except ValueError as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
