
import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('car_prediction_model.pkl')

# Streamlit app
st.title("Car Price Prediction App by Basheer")

st.write("""
### Enter the details of the car to predict its price
""")

# Input fields for the car's features
year = st.number_input("Year of Purchase", min_value=2000, max_value=2025, step=1, value=2017)
buy_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, value=10.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500, value=10000)
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", options=["Dealer", "Individual"])
transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
owner = st.number_input("Number of Previous Owners", min_value=0, max_value=3, step=1, value=0)

# Encode categorical values
fuel_type_encoded = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type]
seller_type_encoded = {"Dealer": 0, "Individual": 1}[seller_type]
transmission_encoded = {"Manual": 0, "Automatic": 1}[transmission]

# Predict button
if st.button("Predict Price"):
    # Prepare the input data as a numpy array
    input_data = np.array([[year, buy_price, kms_driven, owner, fuel_type_encoded, seller_type_encoded, transmission_encoded]])
    # Predict the price
    prediction = model.predict(input_data)
    st.write(f"The predicted price of the car is ₹{prediction[0]:,.2f} lakhs.")
