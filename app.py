from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import joblib
from keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)
# Load the pretrained models
decision_tree_model = joblib.load('Models/decision_tree_model.pkl')
ann_model = load_model('Models/ann_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive the file and other data from the request
    file = request.files['file']
    start = int(request.form['start'])
    stop = int(request.form['stop'])
    
    # Read the excel file
    df = pd.read_csv(file)

    if 'output' in df.columns:
            df.drop(columns=['output'], inplace=True)

    # Select the specified rows
    data_to_predict = df.iloc[start:stop]

    # Convert the data to numpy array if necessary
    data_to_predict_np = data_to_predict.to_numpy()

    # Make predictions using the pretrained models
    decision_tree_predictions = decision_tree_model.predict(data_to_predict)
    ann_predictions = ann_model.predict(data_to_predict_np)

    # Convert ANN predictions to a list
    ann_predictions = ann_predictions.tolist()

    # Prepare the response
    response = {
        'decision_tree_predictions': decision_tree_predictions.tolist(),
        'ann_predictions': ann_predictions,
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
