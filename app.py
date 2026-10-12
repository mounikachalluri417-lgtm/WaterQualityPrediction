
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="centered"
)

# App title
st.title("💧 Water Quality Prediction")
st.info("Model test accuracy: 66.62%")
st.write("Machine Learning Model: Random Forest Classifier")
st.write(
    "Enter water quality measurements below to predict "
    "whether a water sample is classified as potable."
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("water_quality_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# Input measurements
st.subheader("Water Quality Measurements")

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
    solids = st.number_input("Solids", min_value=0.0, value=20000.0)
    chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)

with col2:
    conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

# Prediction
if st.button("🔍 Predict Water Quality", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The model classifies this sample as POTABLE.")
    else:
        st.warning("The model classifies this sample as NOT POTABLE.")

    st.caption(
        "This is an ML prediction, not a laboratory test or a guarantee "
        "that water is safe to drink."
    )

# Footer
st.divider()
st.caption("Water Quality Prediction | Python • Scikit-learn • Streamlit")
