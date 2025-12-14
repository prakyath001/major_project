import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import asyncio
from sanic import Sanic
from sanic.response import json as sjson
from sanic_cors import CORS

# ----------------------------
# Config (optional for LLM)
# ----------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_CMD = os.getenv("OLLAMA_CMD_PATH", r"C:\Users\HP\AppData\Roaming\Microsoft\Windows\Start Menu\Programs")

# ----------------------------
# Top features
# ----------------------------
top_features = [
    "ER status measured by IHC",
    "3-Gene classifier subtype",
    "Pam50 + Claudin-low subtype",
    "PR Status",
    "Nottingham prognostic index",
    "Tumor Size",
    "HER2 Status",
]

# ----------------------------
# Sanic app
# ----------------------------
app = Sanic("BreastCancerAPI")
CORS(app)

# ----------------------------
# Load model & scaler
# ----------------------------
try:
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded")
except Exception as e:
    print("❌ Error loading model/scaler:", e)
    model, scaler = None, None
def call_ollama(prompt: str) -> str:
    import requests,subprocess
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            },
            timeout=200
        )
        data = resp.json()
        print("____________________________________________")
        print(data.get("response", ""))
        print("____________________________________________")
        return data.get("response", "")
    #     result = subprocess.run(
    #         [
    #             r"G:\New folder (3)\ollama.exe",   # 👈 adjust if path changes
    #             "run",
    #             "tinyllama",
    #             "hi"
    #         ],
    #         #input=prompt,
    #         text=True,
    #         capture_output=True,
    #         timeout=120
    #     )

    #     if result.returncode != 0:
    #         print("OLLAMA STDERR:", result.stderr)
    #         return ""

    #     return result.stdout.strip()

    # except subprocess.TimeoutExpired:
    #     print("OLLAMA ERROR: timeout")
    #     return ""
    except Exception as e:
        print("OLLAMA ERROR:", e)
        return ""


# ----------------------------
# Predict route
# ----------------------------
@app.post("/predict")
async def predict(request):
    try:
        data = request.json
        if model is None or scaler is None:
            return sjson({"error": "Model/scaler not loaded"}, status=500)

        # Convert features to float
        try:
            feature_values = [float(data[f]) for f in top_features]
        except KeyError as e:
            return sjson({"error": f"Missing field: {e}"}, status=400)
        except Exception as e:
            return sjson({"error": f"Invalid feature value: {e}"}, status=400)

        # DataFrame for scaler
        X_df = pd.DataFrame([feature_values], columns=top_features)
        X_scaled = scaler.transform(X_df)

        # Prediction
        pred_class = int(model.predict(X_scaled)[0])
        prob_arr = model.predict_proba(X_scaled)[0].tolist() if hasattr(model, "predict_proba") else None

        return sjson({
            "prediction": pred_class,
            "probability": prob_arr,
            "feature_values": feature_values
        })

    except Exception as e:
        return sjson({"error": str(e)}, status=500)

# ----------------------------
# Explain route
# ----------------------------
@app.post("/explain")
async def explain(request):
    try:
        data = request.json
        if model is None or scaler is None:
            return sjson({"error": "Model/scaler not loaded"}, status=500)
        print("i am in line number 90" )
        feature_values = data.get("feature_values")
        prediction = data.get("prediction", None)
        print("i am in line number 93")
        if feature_values is None or len(feature_values) != len(top_features):
            return sjson({"error": f"feature_values must be a list of {len(top_features)} numbers"}, status=400)
        print("i am in line number 96")
        # DataFrame for SHAP
        X_df = pd.DataFrame([feature_values], columns=top_features)
        X_scaled = scaler.transform(X_df)
        print("i am in line number 100")
        # SHAP computation
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, (list, tuple)):
                shap_vals_pos = np.array(shap_values[1])[0] if len(shap_values) > 1 else np.array(shap_values[0])[0]
            else:
                shap_vals_pos = np.array(shap_values)[0]
        except Exception:
            shap_vals_pos = np.zeros(len(top_features))
        print("i m in line number 112")
        # Build top features
        feat_pairs = [
            {"name": name, "value": feature_values[idx], "shap": float(np.array(shap_vals_pos[idx]).flatten()[0])}
            for idx, name in enumerate(top_features)
        ]
       # Keep only risk-increasing features
        positive_feats = [f for f in feat_pairs if f["shap"] > 0]

# Sort by contribution strength
        top_n = sorted(positive_feats, key=lambda x: x["shap"], reverse=True)[:5]

        print("i am in line number 119")
        print(feat_pairs)
        # Placeholder LLM explanations
        current_exp = ", ".join([
            f"{item['name']} = {item['value']} (SHAP: {item['shap']})"
            for item in feat_pairs
        ])

        print("line num 135")
        prompt = f"""
                High breast cancer risk.
                SHAP features: {current_exp}
                Doctor and patient explanation.
                JSON ONLY: {{"doctor":"","patient":""}}
                """



        print("line num 146")
        llm_response=call_ollama(prompt)
        print("llm ",llm_response)
        if llm_response:
            try:
                output=json.loads(llm_response)
            except Exception:
                output=None
        if not output:
            doctor_text = (
                    "The prediction indicates high breast cancer risk driven mainly by ER status, "
                    "Nottingham prognostic index, and hormone receptor markers. Further diagnostic "
                    "tests such as biopsy, imaging, and molecular profiling are recommended."
                )

            patient_text = (
                    "Your result shows a higher risk based on hormone and tumor-related factors. "
                    "This does not confirm cancer, but it is important to consult a doctor, "
                    "follow screening advice, and maintain a healthy lifestyle."
                )

        else:
            doctor_text=output.get("doctor","")
            patient_text=output.get("patient","")
        print("FINAL RESPONSE →", {
            "shap_sorted": top_n,
            "doctor_explain": doctor_text,
            "patient_explain": patient_text
        })

        return sjson({
            "shap_sorted": top_n if top_n else [],
            "doctor_explain": doctor_text or "",
            "patient_explain": patient_text or ""
        })

    except Exception as e:
        return sjson({"error": str(e)}, status=500)

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    print("✅ BreastCancerAPI starting on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
