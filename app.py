import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_output"
MODEL_PATH = MODEL_DIR / "disease_prediction_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Disease Symptom Prediction",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

st.title("Symptom-Based Disease Prediction")
st.caption("Research prototype — machine-learning classification using selected symptoms.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("The trained model or metadata file is missing. Ensure the model_output folder is beside app.py.")
    st.stop()

model, metadata = load_assets()
features = metadata["feature_columns"]

st.write(
    "Select the symptoms reported by the person. The system will return "
    "the disease class predicted by the trained Random Forest model."
)

selected_symptoms = st.multiselect(
    "Search for and select symptoms",
    options=features,
    help="You can type in this box to search the 132 available symptom features."
)

if st.button("Predict disease", type="primary", use_container_width=True):
    if not selected_symptoms:
        st.warning("Please select at least one symptom before making a prediction.")
    else:
        input_data = pd.DataFrame(
            [[1 if feature in selected_symptoms else 0 for feature in features]],
            columns=features
        )
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction result")
        st.success(f"Predicted disease class: **{prediction}**")

        with st.expander("Review selected symptoms"):
            for symptom in selected_symptoms:
                st.write(f"- {symptom.replace('_', ' ')}")

        st.info(
            "This is an academic research prototype based on a structured symptom dataset. "
            "It is not a medical diagnosis, does not replace a qualified healthcare professional "
            "or laboratory testing, and does not independently detect or forecast disease outbreaks. "
            "Seek professional medical care for symptoms or health concerns."
        )

st.divider()
st.caption(
    f"Model: {metadata.get('final_application_model', 'Random Forest')} | "
    f"Symptom inputs: {len(features)} | "
    f"Disease classes: {len(metadata.get('class_labels', []))}"
)
