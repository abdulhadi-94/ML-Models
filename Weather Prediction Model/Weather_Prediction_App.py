from pathlib import Path
BASE_DIR = Path(__file__).parent

import streamlit as st
import pandas as pd
import joblib
import base64

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="Weather Prediction",
    page_icon=BASE_DIR / "weather_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Background ----------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64(BASE_DIR / "weather_bg.jpg")
logo = get_base64(BASE_DIR / "weather_logo.png")

st.markdown(f"""
<style>

/* Background */

.stApp{{
background-image:url("data:image/png;base64,{bg}");
background-size:cover;
background-repeat:no-repeat;
background-position:center;
background-attachment:fixed;
animation:bgMove 20s ease-in-out infinite alternate;
}}

@keyframes bgMove{{
0%{{background-position:center top;}}
100%{{background-position:center bottom;}}
}}

/* Glass Card */

.block-container{{
background:rgba(0,0,0,.35);
padding:2rem;
border-radius:25px;
backdrop-filter:blur(8px);
}}

h1,h2,h3,h4,h5,h6,p,label,span{{
color:white!important;
}}

.stButton>button{{
width:100%;
height:55px;
font-size:20px;
font-weight:bold;
background:#1e88e5;
color:white;
border:none;
border-radius:12px;
}}

.stButton>button:hover{{
background:#1565c0;
}}

div[data-testid="stMetric"]{{
background:white;
padding:15px;
border-radius:15px;
}}

@media (max-width:768px){{

.block-container{{
padding:15px;
}}

h1{{
font-size:30px!important;
text-align:center;
}}

.stButton>button{{
height:50px;
font-size:18px;
}}

}}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------

model = joblib.load(BASE_DIR / "LR_Weather_Predictor.pkl")
scaler = joblib.load(BASE_DIR / "scaler_Weather_Predictor.pkl")
columns = joblib.load(BASE_DIR / "columns_Weather_Predictor.pkl")

# ---------------- Sidebar ----------------

st.sidebar.image(BASE_DIR / "weather_logo.png", width=120)

st.sidebar.title("🌦 Weather Prediction")

st.sidebar.info("""
Predict tomorrow's weather using Machine Learning.

Developer:

Abdul Hadi Shaikh

GitHub:
https://github.com/abdulhadi-94/ML-Models
""")

# ---------------- Title ----------------

st.title("🌦 Weather Prediction System")

st.write(
    "Enter today's weather conditions below and click **Predict Weather**."
)

st.divider()
# ---------------- Inputs ----------------

col1, col2 = st.columns(2)

with col1:

    MinTemp = st.number_input(
        "Minimum Temperature (°C)",
        -20.0,
        50.0,
        15.0
    )

    MaxTemp = st.number_input(
        "Maximum Temperature (°C)",
        -20.0,
        60.0,
        28.0
    )

    Rainfall = st.number_input(
        "Rainfall (mm)",
        0.0,
        500.0,
        0.0
    )

    Evaporation = st.number_input(
        "Evaporation",
        0.0,
        50.0,
        5.0
    )

    Sunshine = st.number_input(
        "Sunshine (Hours)",
        0.0,
        15.0,
        8.0
    )

    WindGustSpeed = st.number_input(
        "Wind Gust Speed (km/h)",
        0,
        150,
        35
    )

    WindSpeed9am = st.number_input(
        "Wind Speed at 9 AM",
        0,
        120,
        15
    )

    WindSpeed3pm = st.number_input(
        "Wind Speed at 3 PM",
        0,
        120,
        20
    )

    RainToday = st.selectbox(
        "Rain Today",
        ["No", "Yes"]
    )

with col2:

    Humidity9am = st.number_input(
        "Humidity at 9 AM (%)",
        0,
        100,
        65
    )

    Humidity3pm = st.number_input(
        "Humidity at 3 PM (%)",
        0,
        100,
        55
    )

    Pressure9am = st.number_input(
        "Pressure at 9 AM (hPa)",
        900.0,
        1100.0,
        1015.0
    )

    Pressure3pm = st.number_input(
        "Pressure at 3 PM (hPa)",
        900.0,
        1100.0,
        1012.0
    )

    Cloud9am = st.slider(
        "Cloud Cover at 9 AM",
        0,
        8,
        3
    )

    Cloud3pm = st.slider(
        "Cloud Cover at 3 PM",
        0,
        8,
        4
    )

    Temp9am = st.number_input(
        "Temperature at 9 AM (°C)",
        -20.0,
        50.0,
        20.0
    )

    Temp3pm = st.number_input(
        "Temperature at 3 PM (°C)",
        -20.0,
        60.0,
        27.0
    )

st.write("")
# ---------------- Prediction ----------------

if st.button("🌦 Predict Weather"):

    sample = pd.DataFrame({

        "MinTemp": [MinTemp],
        "MaxTemp": [MaxTemp],
        "Rainfall": [Rainfall],
        "Evaporation": [Evaporation],
        "Sunshine": [Sunshine],
        "WindGustSpeed": [WindGustSpeed],
        "WindSpeed9am": [WindSpeed9am],
        "WindSpeed3pm": [WindSpeed3pm],
        "Humidity9am": [Humidity9am],
        "Humidity3pm": [Humidity3pm],
        "Pressure9am": [Pressure9am],
        "Pressure3pm": [Pressure3pm],
        "Cloud9am": [Cloud9am],
        "Cloud3pm": [Cloud3pm],
        "Temp9am": [Temp9am],
        "Temp3pm": [Temp3pm],
        "RainToday": [RainToday]

    })

    # One-Hot Encode categorical columns
    sample = pd.get_dummies(sample)

    # Match training columns
    sample = sample.reindex(
        columns=columns,
        fill_value=0
    )

    # Numerical columns to scale
    numeric_cols = [
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Evaporation",
        "Sunshine",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Temp9am",
        "Temp3pm"
    ]

    sample[numeric_cols] = scaler.transform(sample[numeric_cols])

    # Prediction
    prediction = model.predict(sample)

    # Probability (if supported by model)
    try:
        probability = model.predict_proba(sample)[0]

        if prediction[0] == 1:
            st.error("🌧️ Rain Expected Tomorrow")
        else:
            st.success("☀️ No Rain Expected Tomorrow")
          
    except Exception:

        if prediction[0] == 1:
            st.error("🌧️ Rain Expected Tomorrow")
        else:
            st.success("☀️ No Rain Expected Tomorrow")

st.divider()

st.caption(
    "© 2026 Weather Prediction | Built using Streamlit & Scikit-Learn"
)