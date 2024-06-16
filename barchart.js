//FRONTEND
import React, { useState } from "react";
import axios from "axios";

const UploadComponent = () => {
  const [file, setFile] = useState(null);
  const [start, setStart] = useState("");
  const [stop, setStop] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleStartChange = (event) => {
    setStart(event.target.value);
  };

  const handleStopChange = (event) => {
    setStop(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      setError("Please select a file.");
      return;
    }

    if (!start || !stop) {
      setError("Please enter start and stop indices.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("start", start);
    formData.append("stop", stop);

    try {
      const response = await axios.post(
        "http://localhost:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setPredictions(response.data.shap_values_list);
      setError("");
    } catch (error) {
      setError("Error predicting: " + error.message);
    }
  };

  return (
    <div>
      <h2>Upload CSV File and Get Predictions</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Select CSV File:</label>
          <input type="file" onChange={handleFileChange} accept=".csv" />
        </div>
        <div>
          <label>Start Index:</label>
          <input type="number" value={start} onChange={handleStartChange} />
        </div>
        <div>
          <label>Stop Index:</label>
          <input type="number" value={stop} onChange={handleStopChange} />
        </div>
        <button type="submit">Predict</button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {predictions && (
        <div>
          <h3>Predictions and SHAP Values:</h3>
          {predictions.map((item, index) => (
            <div key={index}>
              <h4>Instance {index + 1}</h4>
              <p>
                Decision Tree Predictions:{" "}
                {JSON.stringify(item.decision_tree_predictions)}
              </p>
              <p>ANN Predictions: {JSON.stringify(item.ann_predictions)}</p>
              <p>
                Top 3 Influential Features:{" "}
                {JSON.stringify(item.top_3_features)}
              </p>
              <img
                src={`data:image/png;base64,${item.shap_bar_img}`}
                alt={`SHAP Bar Graph for Instance ${index + 1}`}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default UploadComponent;


//BACKEND
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from keras import models
import shap
from flask_cors import CORS
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load your pretrained models and feature names
decision_tree_model = joblib.load('Models/decision_tree_model.pkl')
ann_model = models.load_model('Models/ann_model.h5')
feature_names = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']

# Assuming X_train is your training data (numpy array)
xtrain = np.load('xtrain.npy')
X_train = xtrain

def get_mean_abs_shap_values(model, X_sample, masker):
    explainer = shap.Explainer(model, masker=masker)
    shap_values = explainer(X_sample)
    shap_values = shap_values.values if len(shap_values.shape) == 2 else shap_values.values[:, :, 0]
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
    return mean_abs_shap_values

def create_shap_bar_graph(shap_values, feature_names):
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values
    })
    shap_df = shap_df.sort_values(by='shap_value', ascending=False)
    plt.figure(figsize=(10, 5))
    plt.barh(shap_df['feature'], shap_df['shap_value'], color=plt.cm.viridis(np.linspace(0, 1, len(shap_df))))
    plt.xlabel('SHAP Value')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graph_base64 = base64.b64encode(image_png).decode('utf-8')
    return graph_base64

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the file and other data from the request
        file = request.files['file']
        start = int(request.form['start'])
        stop = int(request.form['stop'])

        # Read the CSV file
        df = pd.read_csv(file)

        # Drop the output column if it exists (assuming 'output' is the column name)
        if 'output' in df.columns:
            df.drop(columns=['output'], inplace=True)

        # Initialize an empty list to store SHAP values
        shap_values_list = []

        # Define masker
        masker = shap.maskers.Independent(X_train)

        # Iterate over rows from start to stop
        for i in range(start, stop):
            # Select the i-th row to predict
            data_to_predict = df.iloc[i:i+1]

            # Convert the data to numpy array
            data_to_predict_np = data_to_predict.to_numpy()

            # Make predictions using the pretrained models
            decision_tree_predictions = decision_tree_model.predict(data_to_predict_np)
            ann_predictions = ann_model.predict(data_to_predict_np)

            # Compute mean absolute SHAP values for each model
            ann_shap_values = get_mean_abs_shap_values(ann_model, data_to_predict_np, masker)
            dt_shap_values = get_mean_abs_shap_values(decision_tree_model, data_to_predict_np, masker)

            # Combine the mean absolute SHAP values
            combined_shap_values = (ann_shap_values + dt_shap_values) / 2

            # Create SHAP bar graph
            shap_bar_graph = create_shap_bar_graph(combined_shap_values, feature_names)

            # Get top 3 features
            top_3_features = np.argsort(combined_shap_values)[-3:][::-1]
            top_3_feature_names = [feature_names[idx] for idx in top_3_features]

            # Append SHAP values and predictions to the list
            shap_values_list.append({
                'decision_tree_predictions': decision_tree_predictions.tolist(),
                'ann_predictions': ann_predictions.tolist(),
                'shap_values': combined_shap_values.tolist(),
                'feature_names': feature_names,
                'shap_bar_img': shap_bar_graph,
                'top_3_features': top_3_feature_names
            })

        # Prepare the response with all SHAP values and predictions
        response = {
            'shap_values_list': shap_values_list
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
