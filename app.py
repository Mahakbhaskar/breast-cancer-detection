"""
Breast Cancer Detection - Streamlit Application

Loads a trained neural network and its matching feature scaler, collects
30 diagnostic measurements from the user, and predicts whether a tumor is
Benign or Malignant.
"""

import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

# ---------------------------------------------------------------------------
# Load trained model and scaler
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
    "breast_cancer_model.h5",
    compile=False
)

    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------------------------------------------
# 30 diagnostic features, in the order expected by the model
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# Realistic min/max/mean for each feature, based on the training dataset.
# These define sensible slider ranges instead of arbitrary bounds.
FEATURE_STATS = {
    "radius_mean": (6.981, 28.11, 14.13),
    "texture_mean": (9.71, 39.28, 19.29),
    "perimeter_mean": (43.79, 188.5, 91.97),
    "area_mean": (143.5, 2501.0, 654.89),
    "smoothness_mean": (0.05263, 0.1634, 0.09636),
    "compactness_mean": (0.01938, 0.3454, 0.10434),
    "concavity_mean": (0.0, 0.4268, 0.08880),
    "concave points_mean": (0.0, 0.2012, 0.04892),
    "symmetry_mean": (0.106, 0.304, 0.18116),
    "fractal_dimension_mean": (0.04996, 0.09744, 0.06280),
    "radius_se": (0.1115, 2.873, 0.40517),
    "texture_se": (0.3602, 4.885, 1.21685),
    "perimeter_se": (0.757, 21.98, 2.86606),
    "area_se": (6.802, 542.2, 40.33708),
    "smoothness_se": (0.001713, 0.03113, 0.00704),
    "compactness_se": (0.002252, 0.1354, 0.02548),
    "concavity_se": (0.0, 0.396, 0.03189),
    "concave points_se": (0.0, 0.05279, 0.01180),
    "symmetry_se": (0.007882, 0.07895, 0.02054),
    "fractal_dimension_se": (0.0008948, 0.02984, 0.00379),
    "radius_worst": (7.93, 36.04, 16.26919),
    "texture_worst": (12.02, 49.54, 25.67722),
    "perimeter_worst": (50.41, 251.2, 107.26121),
    "area_worst": (185.2, 4254.0, 880.58313),
    "smoothness_worst": (0.07117, 0.2226, 0.13237),
    "compactness_worst": (0.02729, 1.058, 0.25427),
    "concavity_worst": (0.0, 1.252, 0.27219),
    "concave points_worst": (0.0, 0.291, 0.11461),
    "symmetry_worst": (0.1565, 0.6638, 0.29008),
    "fractal_dimension_worst": (0.05504, 0.2075, 0.08395),
}

# Group features for a cleaner sidebar layout
SECTIONS = {
    "Mean values": [f for f in FEATURE_NAMES if f.endswith("_mean")],
    "Standard Error (SE) values": [f for f in FEATURE_NAMES if f.endswith("_se")],
    "Worst values": [f for f in FEATURE_NAMES if f.endswith("_worst")],
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Detector", layout="wide")
st.title("🔬 Breast Cancer Prediction")
st.write(
    "This app uses a neural network trained on the Breast Cancer Wisconsin "
    "(Diagnostic) dataset to classify a tumor as **Benign** or **Malignant** "
    "based on 30 cell-nuclei measurements taken from a digitized image of a "
    "breast mass."
)
st.warning(
    "⚠️ **Disclaimer:** This project is for academic and educational "
    "purposes only. It is not a validated medical diagnostic tool and must "
    "not be used for real clinical decisions."
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Diagnostic Features (30)")
input_values = {}

for section_title, features in SECTIONS.items():
    st.sidebar.subheader(section_title)
    for feature in features:
        fmin, fmax, fmean = FEATURE_STATS[feature]
        display_name = feature.replace("_", " ").title()
        step = (fmax - fmin) / 200
        input_values[feature] = st.sidebar.slider(
            display_name,
            min_value=float(fmin),
            max_value=float(fmax),
            value=float(fmean),
            step=float(step) if step > 0 else 0.001,
            format="%.5f",
        )

# Assemble input in the exact feature order the scaler and model expect
input_df = pd.DataFrame([[input_values[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("🧪 Run Diagnosis"):
    scaled_input = scaler.transform(input_df)
    prob_malignant = float(model.predict(scaled_input, verbose=0)[0][0])
    result = "Malignant 🔴" if prob_malignant > 0.5 else "Benign 🟢"

    st.markdown(f"## Result: **{result}**")
    st.markdown(f"**Malignant Probability**: `{prob_malignant:.4f}`")
    st.progress(min(max(prob_malignant, 0.0), 1.0))

    with st.expander("📌 Model & Pipeline Details"):
        st.markdown(
            """
            - **Model type**: Sequential neural network (Dense 32 → 16 → 1)
            - **Framework**: TensorFlow / Keras
            - **Preprocessing**: input scaled with the `StandardScaler`
              fitted on the training set (`scaler.pkl`)
            - **Output**: sigmoid probability of malignancy (0.0 – 1.0)
            - **Decision threshold**: probability > 0.5 → Malignant
            - **Test-set performance**: see the project README for accuracy,
              precision, recall and F1-score.
            """
        )

st.markdown("---")
st.caption(
    "Academic project · Breast Cancer Wisconsin (Diagnostic) dataset · "
    "TensorFlow + Streamlit"
)
