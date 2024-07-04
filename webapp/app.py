from flask import Flask, request, jsonify, render_template
import pandas as pd
import sqlite3
import joblib
from tqdm import tqdm

app = Flask(__name__)

# Load model
model = joblib.load('/home/g00d3y3/g00d3y3/models/xgb_model_at_bat.pkl')

# Connect to SQLite database
conn = sqlite3.connect('g00d3y3.db', check_same_thread=False)

# Load player data
player_data = pd.read_sql_query("SELECT * FROM player_name", conn)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_player', methods=['GET'])
def search_player():
    query = request.args.get('q')
    results = player_data[player_data['name'].str.contains(query, case=False, na=False)].to_dict('records')
    return jsonify(results)

@app.route('/get_players_by_team_year', methods=['GET'])
def get_players_by_team_year():
    team = request.args.get('team')
    year = request.args.get('year')
    players = pd.read_sql_query(f"SELECT * FROM players WHERE team = '{team}' AND year = {year}", conn).to_dict('records')
    return jsonify(players)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    batter = data['batter']
    pitcher = data['pitcher']
    
    # Create input features for the model
    features = create_features(batter, pitcher)
    
    # Predict the event
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

def create_features(batter, pitcher):
    # Placeholder function to create features for the model
    # Extract features from batter and pitcher data
    return [batter['feature1'], pitcher['feature1'], ...]

if __name__ == '__main__':
    app.run(debug=True)
