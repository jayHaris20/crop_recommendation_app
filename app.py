import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Setup
st.set_page_config(
    page_title="Crop Recommendation System", page_icon="🌾", layout="centered"
)

st.title("🌾 Crop Recommendation System")
st.write(
    "Enter the soil and environmental conditions to receive an instant ML crop recommendation."
)

MODEL_PATH = "models/crop_model.pkl"


# Load Trained Model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

if model is None:
    st.error(
        f"Model file not found at '{MODEL_PATH}'. Run 'python train_model.py' first!"
    )
else:
    # 2-Column Inputs (Soil vs Climate Metrics)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Soil Metrics")
        n = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=140.0,
            value=90.0,
            step=1.0,
        )
        p = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=145.0,
            value=42.0,
            step=1.0,
        )
        k = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=205.0,
            value=43.0,
            step=1.0,
        )
        ph = st.number_input(
            "Soil pH Level",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
        )

    with col2:
        st.subheader("🌤️️ Climate Metrics")
        temp = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=20.87,
            step=0.1,
        )
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            step=0.1,
        )
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=300.0,
            value=202.93,
            step=1.0,
        )

    st.markdown("---")
    if st.button("🔍 Get Recommendation", type="primary", use_container_width=True):
        input_data = pd.DataFrame(
            [
                {
                    "N": n,
                    "P": p,
                    "K": k,
                    "temperature": temp,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                }
            ]
        )

        prediction = model.predict(input_data)[0]
        st.success(
            f"**Recommended Crop for Cultivation:** {prediction.upper()}"
        )