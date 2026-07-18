import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Ford Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# -------------------------------
# Load Model & Files
# -------------------------------
model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("🚗 Ford Car Price Prediction App")
st.write("Enter the car details below to predict its price.")

# -------------------------------
# User Inputs
# -------------------------------

model_name = st.selectbox(
    "Select Model",
    [
        " Fiesta"," Focus"," Kuga"," EcoSport",
        " Mondeo"," KA"," Puma"," C-MAX",
        " B-MAX"," S-MAX"," Galaxy"," Edge",
        " Grand C-MAX"," Tourneo Custom"," Mustang"
    ]
)

year = st.number_input(
    "Year",
    min_value=1996,
    max_value=2025,
    value=2018,
    step=1
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=30000,
    step=1000
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric"]
)

tax = st.number_input(
    "Tax",
    min_value=0,
    value=150,
    step=1
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    value=55.4,
    step=0.1
)

engineSize = st.number_input(
    "Engine Size",
    min_value=1.0,
    max_value=5.0,
    value=1.5,
    step=0.1
)

# -------------------------------
# Predict Button
# -------------------------------

if st.button("Predict Price"):

    # Create DataFrame
    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    # One-Hot Encoding
    input_data = pd.get_dummies(input_data)

    # Align with training columns
    input_data = input_data.reindex(columns=encoded_columns, fill_value=0)

    # Scale numeric columns
    numeric_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Prediction
    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")