import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Define the time step (must match the value used during training)
time_step = 10

# Load the trained model and scaler
model = load_model('nifty_version_1.0.h5')  # Ensure this file exists in the same directory
scaler = joblib.load('scaler_version_1.0.pkl')  # Ensure this file exists in the same directory

# Function to predict if the market will open high or low based on previous day's data
def predict_opening(input_data):
    input_data_scaled = scaler.transform(input_data)  # Scale input using the fitted scaler
    input_data_reshaped = np.reshape(input_data_scaled, (1, time_step, input_data_scaled.shape[1]))  # Reshape for LSTM
    prediction = model.predict(input_data_reshaped)
    return "Open High" if prediction[0][0] > 0.5 else "Open Low"

# Streamlit UI setup
st.title("Nifty 50 Index 'Market Opening Prediction' model")

# Input features using sliders with matching types for min, max, and step
open_price = st.slider("Open Price", min_value=20000.0, max_value=30000.0, value=23000.0, step=1.0)
open_price_input = st.number_input("Enter Open Price", min_value=20000.0, max_value=30000.0, value=23000.0)

high_price = st.slider("High Price", min_value=20000.0, max_value=30000.0, value=23500.0, step=1.0)
high_price_input = st.number_input("Enter High Price", min_value=20000.0, max_value=30000.0, value=23500.0)

low_price = st.slider("Low Price", min_value=20000.0, max_value=30000.0, value=22500.0, step=1.0)
low_price_input = st.number_input("Enter Low Price", min_value=20000.0, max_value=30000.0, value=22500.0)

close_price = st.slider("Close Price", min_value=20000.0, max_value=30000.0, value=23500.0, step=1.0)
close_price_input = st.number_input("Enter Close Price", min_value=20000.0, max_value=30000.0, value=23500.0)

# Use the slider values or input box values for prediction
open_price = open_price_input if open_price_input else open_price
high_price = high_price_input if high_price_input else high_price
low_price = low_price_input if low_price_input else low_price
close_price = close_price_input if close_price_input else close_price

# Calculate additional features for prediction
pct_change = ((close_price - open_price) / open_price) * 100  # Percentage change
ma_5 = np.mean([close_price] * 5)  # Placeholder for moving average (replace with actual logic)
ma_10 = np.mean([close_price] * 10)  # Placeholder for moving average (replace with actual logic)

# Prepare input data for prediction
input_data = np.array([[open_price, high_price, low_price, close_price, pct_change, ma_5, ma_10]])

# Pad input data to match time_step if necessary
if len(input_data) < time_step:
    padding = [input_data[0]] * (time_step - len(input_data))  # Duplicate first entry for padding
    input_data = np.vstack((padding, input_data))

# Button to make prediction
if st.button("Predict"):
    result = predict_opening(input_data)
    st.success(f"The market is predicted to: **{result}**")

# Developer information at the bottom of the app
st.markdown("---")
st.markdown("### Developed by Basheer")
st.markdown(f"Current date: {pd.Timestamp.now().strftime('%A, %B %d, %Y %I:%M %p IST')}")

# Run the app using: streamlit run app.py in terminal/command prompt.
