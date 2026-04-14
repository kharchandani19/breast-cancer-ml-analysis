import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load breast cancer dataset for feature names and means
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Default values (just put 0.0 or some constant)
mean_vals = [0.0] * len(feature_names)

st.title("Breast Cancer Prediction App")
st.write(
    "Enter the values for each feature in the sidebar and click Predict to see the likelihood of malignancy. (0 = malignant, 1 = benign)"
)

# Sidebar inputs
st.sidebar.header("Input Features")

input_data = []
for i, feature in enumerate(feature_names):
    val = st.sidebar.number_input(
        feature, 
        value=float(np.round(mean_vals[i], 3)),
        format="%.3f"
    )
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][0]  # Probability for class 0 (malignant)
    if prediction == 0:
        st.error(
            f"Prediction: Malignant\nProbability (malignant): {proba:.3f}"
        )
    else:
        st.success(
            f"Prediction: Benign\nProbability (malignant): {proba:.3f}"
        )