import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the trained model, scaler, and polynomial feature transformer
model = joblib.load("lco_xgboost_model.pkl")  # Save your model as 'lco_model.pkl'
scaler = joblib.load("lco_xgboost_scaler.pkl")  # Save your scaler as 'lco_scaler.pkl'
poly = joblib.load("lco_xgboost_poly.pkl")  # Save your polynomial transformer as 'lco_poly.pkl'

# Set the page configuration for a better appearance
st.set_page_config(page_title="LCO 'D 95' Prediction", page_icon="🌟", layout="centered")

# Add a header with color and description
st.markdown(
    """
    <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #1625f5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-header">LCO D 95 Lab Value Prediction Model</div>', unsafe_allow_html=True)
st.write("Provide the values of **Tray temperature** and **Draw down temperature** of the 'Main Fractionator' to predict the lab value.")

# Add sliders for input values
tray = st.slider(
    "FCCU Fractionator Tray temperature,°C", min_value=200.0, max_value=300.0, step=1.0, value=270.0
)
draw = st.slider(
    "FCCU Fractionator Draw down temperature,°C", min_value=150.0, max_value=250.0, step=1.0, value=215.0
)
press = st.slider(
    "FCCU Fractionator Press, kg/cm²", min_value=1.0, max_value=2.0, step=0.1, value=1.3
)
run = st.slider(
    "FCCU Fractionator Run down flow, m³/hr", min_value=20.0, max_value=50.0, step=1.0, value=35.0
) 

# Predict button with styling
if st.button("Predict 🔮"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({'draw': [draw], 'tray': [tray], 'press': [press], 'run': [run]})
    
    # Apply the polynomial feature transformation (from your trained model)
    input_poly = poly.transform(input_data)

    # Scale the input data
    input_scaled = scaler.transform(input_poly)
    
    # Make predictions
    prediction = model.predict(input_scaled)
    
    # Display the prediction with color
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 24px; color: #9c100b;">
            <strong>Predicted LCO D 95 °C: {prediction[0]:.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
        
else:
    st.info("Use the sliders to select values, then click 'Predict 🔮' to see the result.")
