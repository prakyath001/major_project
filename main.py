import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# # Load dataset
# df = pd.read_csv('mp_dataset.csv')
# df.info()

# df.head()
# # ----------------------------
# # Drop Unnecessary Columns
# # ----------------------------

# df.drop(['Patient ID', 'Entry_date', 'End_of_time', 'Date_died'], axis=1, inplace=True)


# # Drop columns with >40% missing values
# df = df[df.columns[df.isnull().mean() < 0.4]]

# # Fill missing values
# for col in df.columns:
#     if df[col].dtype == 'object':
#         df[col].fillna(df[col].mode()[0], inplace=True)
#     else:
#         df[col].fillna(df[col].median(), inplace=True)

# # ----------------------------
# #Encode Categorical Variables
# # ----------------------------

# categorical_cols = df.select_dtypes(include=['object']).columns
# le = LabelEncoder()
# for col in categorical_cols:
#     df[col] = le.fit_transform(df[col])

# # ----------------------------
# #  Define Features and Target (use only top 5 features)
# # ----------------------------

# target = 'ER Status'

# # Select only the top 5 important features:
top_features = [
    'ER status measured by IHC',
    '3-Gene classifier subtype',
    'Pam50 + Claudin-low subtype',
    'PR Status',
    'Nottingham prognostic index',
    'Tumor Size',
    'HER2 Status'
]

# X = df[top_features]
# y = df[target]

# # Scale features
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ----------------------------
# #  Train/Test Split
# # ----------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )
# print (y)
# # ----------------------------
# #  Train Model with Controlled Depth
# # ----------------------------

# clf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=10,             # Limit depth to reduce overfitting
#     random_state=42,
#     class_weight='balanced'   # Helps with class imbalance
# )
# clf.fit(X_train, y_train)

# # ----------------------------
# #  Evaluate Model
# # ----------------------------

# # Train performance
# y_train_pred = clf.predict(X_train)
# print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
# print("Train Classification Report:\n", classification_report(y_train, y_train_pred))

# # Test performance
# y_test_pred = clf.predict(X_test)
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
# print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

# # Confusion Matrix
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# # ----------------------------
# #  Cross-Validation Score
# # ----------------------------

# cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
# print("Cross-Validation Scores:", cv_scores)
# print("Mean CV Accuracy:", cv_scores.mean())

# # ----------------------------
# #  Feature Importances
# # ----------------------------

# feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
# feature_importances.nlargest(20).plot(kind='barh', figsize=(10, 6))
# plt.title("Top 7 Feature Importances")
# plt.tight_layout()
# plt.show()



# import joblib

# # Save the model to a file
# joblib.dump(clf, 'breast_cancer_model.pkl')

# # Optionally save your scaler if you plan to use it in production
# joblib.dump(scaler, 'scaler.pkl')

# #frontend and shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

app = FastAPI()

# # Load your existing model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# # Top features used in your model
# top_features = [
#     'ER status measured by IHC',
#     '3-Gene classifier subtype',
#     'Pam50 + Claudin-low subtype',
#     'PR Status',
#     'Nottingham prognostic index',
#     'Tumor Size',
#     'HER2 Status'
# ]

# Pydantic model to receive JSON from frontend
class Features(BaseModel):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    g: float

@app.post("/predict")
def predict(data: Features):
    try:
        # Convert JSON to DataFrame
        X = pd.DataFrame([[data.a, data.b, data.c, data.d, data.e, data.f, data.g]],
                         columns=top_features)

        # Scale features
        X_scaled = scaler.transform(X)

        # Prediction
        pred_class = int(model.predict(X_scaled)[0])
        pred_proba = model.predict_proba(X_scaled)[0].tolist()

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        shap_vals_pos = shap_values[1][0].tolist()  # positive class

        # Return JSON
        return {
            "prediction": pred_class,
            "probability": pred_proba,
            "shap_values": shap_vals_pos,
            "features": top_features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)