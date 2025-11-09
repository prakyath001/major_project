from sanic import Sanic
from sanic.response import json
from sanic_cors import CORS
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ----------------------------
# Define top features
# ----------------------------
top_features = [
    "ER status measured by IHC",
    "3-Gene classifier subtype",
    "Pam50 + Claudin-low subtype",
    "PR Status",
    "Nottingham prognostic index",
    "Tumor Size",
    "HER2 Status"
]

# ----------------------------
# Initialize app
# ----------------------------
app = Sanic("BreastCancerAPI")
CORS(app)

# ----------------------------
# Load model and scaler
# ----------------------------
try:
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("❌ Error loading model or scaler:", e)
    model, scaler = None, None

# ----------------------------
# Predict route
# ----------------------------
@app.post("/predict")
async def predict(request):
    try:
        data = request.json
        if model is None or scaler is None:
            return json({"error": "Model or scaler not loaded properly"}, status=500)
        features = [float(data[f]) for f in top_features]
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return json({"prediction": bool(prediction)})
    except KeyError as e:
        return json({"error": f"Missing field: {e}"}, status=400)
    except Exception as e:
        return json({"error": str(e)}, status=500)

# ----------------------------
# Explain route
# ----------------------------
@app.post("/explain")
async def explain(request):
    try:
        data = request.json
        prediction = data.get("prediction", None)
        if model is None or scaler is None:
            return json({"error": "Model or scaler not loaded properly"}, status=500)
        features = []
        for f in top_features:
            val = data[f]
            if isinstance(val, list):
                val = val[0]
            if isinstance(val, str):
                val = val.replace(",", ".")
            features.append(float(val))
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        X_df = pd.DataFrame(features_scaled, columns=top_features)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.array(shap_values).reshape(1, -1)
        explanation = {f: float(shap_values[0, idx]) for idx, f in enumerate(top_features)}
        threshold = 0.05
        if prediction == 1:
            positive_risk = {f: v for f, v in explanation.items() if v > 0}
            suggestion = "High risk due to: " + ", ".join([f"{f} ({v:.2f})" for f, v in positive_risk.items()])
        else:
            potential_risk = {f: v for f, v in explanation.items() if v > threshold}
            if potential_risk:
                suggestion = "Low risk currently, but keep an eye on: " + ", ".join(
                    [f"{f} ({v:.2f})" for f, v in potential_risk.items()]
                )
            else:
                suggestion = "Low risk currently, no immediate risk factors."
        return json({
            "explanation": explanation,
            "suggestion": suggestion
        })
    except Exception as e:
        return json({"error": str(e)}, status=500)

# ----------------------------
# Evaluation Endpoint
# ----------------------------
@app.get("/evaluate")
async def evaluation(request):
    result = evaluate_model(return_results=True)
    return json(result)

# ----------------------------
# Evaluation Function (prints metrics)
# ----------------------------
def evaluate_model():
    try:
        df = pd.read_csv("mp_dataset.csv")
        target_column = "Overall Survival Status"
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found!")
            return
        cols_to_keep = top_features + [target_column]
        df = df[cols_to_keep].dropna()
        if df.empty:
            print("No valid rows left for evaluation!")
            return
        mapping = {
            "positive": 1, "pos": 1, "positve": 1, "posotive": 1,
            "negative": 0, "neg": 0, "na": np.nan, "nan": np.nan, "": np.nan
        }
        for col in top_features:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.lower().replace(mapping)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        # Expanded target mapping
        y_test = df[target_column].astype(str).str.strip().replace({
            "1": 1, "0": 0,
            "Living": 0, "Alive": 0, "Dead": 1, "Deceased": 1,
            "Died of Disease": 1, "Died of Other Causes": 1,
            "Yes": 1, "No": 0
        })
        df = df[y_test.isin([0, 1])]
        y_test = y_test[y_test.isin([0, 1])].astype(int)
        if y_test.empty:
            print("No valid target labels found!")
            return
        X_test = df[top_features]
        X_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y_test, y_pred)
        print("\n🧮 Confusion Matrix:\n", cm)
    except Exception as e:
        print("⚠️ Evaluation error:", e)


# ----------------------------
# Run Server and Print Evaluation
# ----------------------------
if __name__ == "__main__":
    print("✅ BreastCancerAPI running on http://localhost:8000")
    evaluate_model()   # Directly prints metrics to console
    app.run(host="0.0.0.0", port=8000, debug=False)
