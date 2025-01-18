import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("lco_model.pkl")
scaler = joblib.load("lco_scaler.pkl")

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
st.markdown('<div class="main-header">LCO D 95 Lab value prediction model</div>', unsafe_allow_html=True)
st.write("Provide the values of **Tray temperature** and **Draw down temperature** of the 'Main Fractionator' to predict the lab value.")

# Add sliders for input values
tray = st.slider(
    "FCCU Fractionator Tray temperature (°C)", min_value=200.0, max_value=300.0, step=1.0, value=270.0
)
run = st.slider(
    "FCCU Fractionator Draw down temperature (°C)", min_value=150.0, max_value=250.0, step=1.0, value=215.0
)

# Predict button with styling
if st.button("Predict 🔮"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({'tray': [tray], 'run': [run]})
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
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
