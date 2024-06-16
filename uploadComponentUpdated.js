import React, { useState } from "react";
import axios from "axios";

const UploadComponent = () => {
  const [file, setFile] = useState(null);
  const [start, setStart] = useState("");
  const [stop, setStop] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [annPredictions, setAnnPredictions] = useState([]);
  const [dtPredictions, setDtPredictions] = useState([]);
  const [topFeatures, setTopFeatures] = useState([]);
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

      const data = response.data.shap_values_list;

      setAnnPredictions(data.map((item) => item.ann_predictions));
      setDtPredictions(data.map((item) => item.decision_tree_predictions));
      setTopFeatures(data.map((item) => item.top_3_features));

      setPredictions(data);
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
          <h3>ANN Predictions:</h3>
          <ul>
            {annPredictions.map((prediction, index) => (
              <li key={index}>{JSON.stringify(prediction)}</li>
            ))}
          </ul>

          <h3>Decision Tree Predictions:</h3>
          <ul>
            {dtPredictions.map((prediction, index) => (
              <li key={index}>{JSON.stringify(prediction)}</li>
            ))}
          </ul>
          
          <h3>Top 3 Influential Features:</h3>
          <ul>
            {topFeatures.map((features, index) => (
              <li key={index}>{features.join(", ")}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default UploadComponent;
