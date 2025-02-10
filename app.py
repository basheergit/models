import streamlit as st
import joblib
import numpy as np


# Load the saved model and scaler
best_model = joblib.load("lco_knn.pkl")
scaler = joblib.load("lco_knn_scaler.pkl")

# --- Configuration ---
PRIMARY_COLOR = "#2e9bff"  # A vibrant blue
SECONDARY_COLOR = "#f0f8ff"  # A light, airy background color
ACCENT_COLOR = "#ffa500"  # A warm orange for highlights
FONT_COLOR = "#333333"  # Dark grey for text
BACKGROUND_COLOR = "#c7f0d6"  # Very light grey background

# Page Layout:  MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# --- Custom CSS ---
st.markdown(
    f"""
    <style>
        /* General Body Styling */
        body {{
            background-color: {BACKGROUND_COLOR}; /* Apply background color here */
            color: {FONT_COLOR};
            font-family: 'Arial', sans-serif;
        }}

        .stApp {{
            background-color: {BACKGROUND_COLOR}; /* Apply background color here */
        }}

        /* Title Styling */
        .stApp h1 {{
            color: {PRIMARY_COLOR};
            text-align: center;
            padding-bottom: 20px;
            border-bottom: 2px solid {ACCENT_COLOR};
        }}

        /* Subheader Styling */
        .stApp h3 {{
            color: {ACCENT_COLOR};
            text-align: center;
        }}

        /* Slider Styling */
        .stSlider>div>div>div>div {{
            background-color: {PRIMARY_COLOR};
        }}

        /* Button Styling */
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {ACCENT_COLOR};
        }}

        /* Success Box Styling */
        .stAlert {{
            background-color: #c9bce8;
            color: #161716;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
        }}

        /* Footer Styling */
        .footer {{
            text-align: center;
            margin-top: 50px;
            font-size: 16px;
            color: red;
        }}

        /* Image Styling */
        .stImage > img {{
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        /* Sidebar Styling */
        .stSidebar {{
            background-color: {SECONDARY_COLOR};
            padding: 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Create two columns: Main UI and Image
col1, col2 = st.columns([2, 1])

with col1:
    # Main Title
    st.title("🔬 FCCU 'LCO D 95' Lab Value Prediction App")
    st.markdown("### Adjust the sliders to input Main Column Tray temp, Draw off temp, and Run down flow to predict Lab Value.")

    # Define slider ranges (adjust based on your dataset)
    tray = st.slider("Tray temp,°C :", min_value=250, max_value=300, step=1, value=285)
    draw = st.slider("Draw off temp,°C:", min_value=200, max_value=250, step=1, value=220)
    flow = st.slider("LCO Run down flow, m3/hr:", min_value=20, max_value=60, step=1, value=40)

    # Prediction button
    if st.button("🔍 Predict Lab Value"):
        # Prepare input data
        feature_input = np.array([[tray, draw, flow]])

        # Scale input if necessary
        if best_model.__class__.__name__ in ['SVR', 'KNeighborsRegressor', 'Lasso', 'Ridge']:
            feature_input = scaler.transform(feature_input)

        # Predict lab value
        predicted_lab = best_model.predict(feature_input)[0]

        # Display result
        st.success(f"✅ Predicted Lab Value,°C: {predicted_lab:.4f}")

with col2:
    # Right-side image (Replace 'image.png' with the actual image file)
    st.image("distillation.jpg", caption="FCCU 'LCO D 95' Lab Value Prediction", use_container_width=True)

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by <b>SKB</b> | © 2025 All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True,
)
