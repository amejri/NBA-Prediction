from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import json
from sklearn.externals import joblib
import pickle
import numpy as np

MODELS_BASE_PATH="models"

# Get headers for payload
headers = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA','3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']

with open('models/SVC.pkl', 'rb') as fid:
    model = pickle.load(fid)

scaler = joblib.load(MODELS_BASE_PATH + "/scaler.save")


app = Flask(__name__)
CORS(app)

def preprocess(data):
    data["min_per_game"] = data["MIN"]/data["GP"]
    data["rebound_per_game"] = data["REB"]/data["GP"]
    data["assist_per_game"] = data["AST"]/data["GP"]
    data["steals_per_game"] = data["STL"]/data["GP"]
    data["blocks_per_game"] = data["BLK"]/data["GP"]
    data["turnovers_per_game"] = data["TOV"]/data["GP"]
    
    data.drop(columns = ["PTS", "OREB", "FTM"], inplace = True)
    
    data.fillna(0, inplace = True)

    columns_to_delete = ["MIN","REB","AST","STL","BLK","TOV"]
    data.drop(columns = columns_to_delete, inplace = True)
    columns_to_process = ['FGM', 'FGA', '3P Made', '3PA', 'FTA', 'FT%', 'DREB', 'min_per_game', 'rebound_per_game', 'assist_per_game', 'steals_per_game', 'blocks_per_game', 'turnovers_per_game']
    for col in columns_to_process :
        data['log_'+col] = np.log1p(data[col])
    return data

def compute(values):
    input_variables = pd.DataFrame([values],
                                columns=headers,
                                dtype=float,
                                index=['input'])

    input_variables = preprocess(input_variables)
    input_variables = scaler.transform(input_variables)

    # Get the model's prediction
    prediction = model.predict_proba(input_variables)
    pred = prediction

    return float(pred)

@app.route("/", methods=['GET'])
def hello():
    return "Welcome to NBA talents predictor tool"

@app.route("/api/nba-ml/player", methods=['POST'])
def predict():
    payload = request.json['data']
    values = [float(i) for i in payload]
    return jsonify({'prediction':compute(values)})

@app.route("/nba-ml/player", methods=['POST','GET'])
def predict_interface():
    if request.method == 'POST':
        values = []
        result = request.form
        for v in result.values():
            values.append(v)
        res = compute(values)

        return render_template('index.html',result='The player will succeed with probability {}%'.format( 100 * round(res , 2) ) )
    return render_template('index.html')



# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
