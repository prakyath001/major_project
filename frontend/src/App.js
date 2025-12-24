import React, { useState } from "react";
import "./App.css";

function App() {
  const fields = [
    "ER status measured by IHC",
    "3-Gene classifier subtype",
    "Pam50 + Claudin-low subtype",
    "PR Status",
    "Nottingham prognostic index",
    "Tumor Size",
    "HER2 Status",
  ];

  // States
  const [form, setForm] = useState(Object.fromEntries(fields.map(f => [f, ""])));
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [suggestion, setSuggestion] = useState(null);
  const [doctorExplain, setDoctorExplain] = useState(null); // ✅ added missing state
  const [loading, setLoading] = useState(false);
  const [featureValues, setFeatureValues] = useState([]); // to store feature values from /predict

  // Input change handler
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  // Predict API call
  const handlePredict = async () => {
    const values = fields.map(f => Number(form[f]));
    if (values.some(isNaN)) {
      alert("Please fill all fields with valid numbers.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();

      if (data.error) {
        alert("Predict API error: " + data.error);
        setLoading(false);
        return;
      }

      setPrediction(data.prediction);
      setProbability(data.probability ? data.probability[1] : null);
      setExplanation(null);
      setSuggestion(null);
      setDoctorExplain(null);
      setFeatureValues(data.feature_values || values); // store for /explain
    } catch (err) {
      console.error(err);
      alert("Predict API failed");
    }
    setLoading(false);
  };

  // Explain API call
  const handleExplain = async () => {
    if (!featureValues || !prediction) {
      alert("Please run prediction first");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          feature_values: featureValues,
          prediction: prediction,
        }),
      });

      const data = await res.json();
      console.log("EXPLAIN RESPONSE →", data);

      if (!data || !Array.isArray(data.shap_sorted)) {
        alert("Explain API error");
        setLoading(false);
        return;
      }

      setExplanation(
        Object.fromEntries(
          data.shap_sorted.map(item => [item.name, item.shap])
        )
      );

      setSuggestion(data.patient_explain || "");
      setDoctorExplain(data.doctor_explain || "");

    } catch (err) {
      console.error(err);
      alert("Explain API failed");
    }
    setLoading(false);
  };

  return (
    <div className="page">
      <div className="card">
        <h2 className="title">Breast Cancer Risk Prediction</h2>

        {/* Input fields */}
        {fields.map((field) => (
          <div className="input-group" key={field}>
            <label>{field}</label>
            <input
              type="number"
              step="any"
              name={field}
              value={form[field]}
              onChange={handleChange}
            />
          </div>
        ))}

        {/* Buttons */}
        <div className="btn-row">
          <button className="btn predict-btn" onClick={handlePredict} disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button
            className="btn explain-btn"
            disabled={!prediction || loading}
            onClick={handleExplain}
          >
            {loading ? "Explaining..." : "Explain"}
          </button>
        </div>

        {/* Prediction result */}
        {prediction !== null && (
          <div className="result-box">
            <h3>
              Prediction:{" "}
              <span className={prediction ? "high" : "low"}>
                {prediction ? "High Risk" : "Low Risk"}
              </span>
            </h3>
            {probability !== null && <p>Probability: {(probability * 100).toFixed(2)}%</p>}
          </div>
        )}

        {/* Explanation result */}
        {explanation && (
          <div className="explain-box">
            <h4>Feature Contributions (SHAP)</h4>
            <ul>
              {Object.entries(explanation).map(([k, v]) => (
                <li key={k}>
                  <strong>{k}:</strong> {v.toFixed(4)}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* AI Suggestion */}
        {suggestion && (
          <div className="suggest-box">
            <h4>Patient Suggestion</h4>
            <p>{suggestion}</p>
          </div>
        )}

        {/* Doctor Explanation */}
        {doctorExplain && (
          <div className="doctor-box">
            <h4>Doctor Explanation</h4>
            <p>{doctorExplain}</p>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
