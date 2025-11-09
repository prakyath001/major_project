import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the same top 7 features used during training
top_features = [
    'ER status measured by IHC',
    '3-Gene classifier subtype',
    'Pam50 + Claudin-low subtype',
    'PR Status',
    'Nottingham prognostic index',
    'Tumor Size',
    'HER2 Status'
]

if __name__ == "__main__":
    # Load and preprocess dataset
    df = pd.read_csv('mp_dataset.csv')
    df.drop(['Patient ID', 'Entry_date', 'End_of_time', 'Date_died'], axis=1, inplace=True)
    df = df[df.columns[df.isnull().mean() < 0.4]]

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Select features
    X = df[top_features]

    # Scale features
    X_scaled = scaler.transform(X)
    X_df = pd.DataFrame(X_scaled, columns=top_features)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # For binary classification, take positive class (usually class 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Print shape for debugging
    print("SHAP values shape:", shap_values.shape)
    print("Feature matrix shape:", X_df.shape)

    # Global summary plot
    shap.summary_plot(shap_values, X_df, feature_names=top_features)

    # SHAP values for the first sample (JSON-friendly)
    explanation = {feature: float(shap_values[0, idx]) 
                   for idx, feature in enumerate(top_features)}

    print("SHAP values for first sample:")
    print(explanation)
