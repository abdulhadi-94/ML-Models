import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Ford Car Price Prediction",
    page_icon="Ford_logo_only.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("Ford Model/background.jpg")


page_bg = f"""
<style>

.stApp {{

    background-image: url("data:image/png;base64,{img}");
    background-size: 100% auto;
    background-repeat: no-repeat;
    background-position: center top;
    background-color: black;
    fitler: blur(3px);
    background-attachment: fixed;
}}

.block-container{{
    background: rgba(0,0,0,0.30);
    padding:2rem;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.15);
   
}}
h1,h2,h3,h4,h5,h6,p,label,span {{
    color:white !important;
}}

.stSelectbox label,
.stNumberInput label {{
    color:white !important;
}}

.stButton>button {{
    background:#0066ff;
    color:white;
    border-radius:10px;
    height:55px;
    font-size:20px;
    font-weight:bold;
}}

.stButton>button:hover {{
    background:#0052cc;
}}

</style>
"""

st.markdown(f"""
<style>

.stApp {{
   
    background-image: url("data:image/png;base64,{img}");
    background-size: 120%;
    background-repeat: no-repeat;
    animation: verticalMove 30s ease-in-out infinite alternate;
}}

@keyframes verticalMove {{
    0% {{
        background-position: center top;
    }}

    100% {{
        background-position: center bottom;
    }}
}}

</style>
""", unsafe_allow_html=True)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
            
.stApp {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;

    /* Move image */
    background-position: center top;
}}

.main{
    padding-top:20px;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:10px;
    background-color:#1565C0;
    color:white;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#0D47A1;
}

div[data-testid="stMetric"]{
    background:#F5F5F5;
    padding:15px;
    border-radius:12px;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Files ----------------
model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# ---------------- Sidebar ----------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/3/3e/Ford_logo_flat.svg",
    width=180,
)

st.sidebar.title("About")

st.sidebar.info(
"""
### Ford Car Price Prediction

Enter vehicle details to predict the estimated selling price using a trained Machine Learning model.

Developer:
Abdul Hadi Shaikh
"""
)

logo = get_base64("Ford_logo_only.jpg")

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:center; gap:15px;">
    <img src="data:image/png;base64,{logo}" width="90">
    <h1 style="color:#003478; margin:0;">
        Ford Car Price Prediction
    </h1>
</div>
""", unsafe_allow_html=True)

# ---------------- Header ----------------

st.write("Fill all vehicle details below and click **Predict Price**.")

st.divider()

# ---------------- Input Section ----------------

col1, col2 = st.columns(2)

with col1:

    model_name = st.selectbox(
        "Model",
        [
            "Fiesta","Focus","Kuga","EcoSport","Puma",
            "Mondeo","Ka+","S-MAX","B-MAX",
            "Galaxy","Edge","Tourneo Custom",
            "Grand C-MAX","C-MAX","Mustang",
            "Ranger","Transit Tourneo"
        ]
    )

    year = st.number_input(
        "Year",
        1990,
        2030,
        2019
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual","Automatic","Semi-Auto"]
    )

    mileage = st.number_input(
        "Mileage",
        0,
        300000,
        10000
    )

with col2:

    fuelType = st.selectbox(
        "Fuel Type",
        ["Petrol","Diesel","Hybrid","Electric","Other"]
    )

    tax = st.number_input(
        "Tax",
        0,
        500,
        145
    )

    mpg = st.number_input(
        "MPG",
        0.0,
        200.0,
        55.0
    )

    engineSize = st.number_input(
        "Engine Size",
        0.8,
        6.0,
        1.5
    )

st.write("")

# ---------------- Prediction ----------------

if st.button("🚗 Predict Price"):

    input_data = pd.DataFrame({
        "model":[model_name],
        "year":[year],
        "transmission":[transmission],
        "mileage":[mileage],
        "fuelType":[fuelType],
        "tax":[tax],
        "mpg":[mpg],
        "engineSize":[engineSize]
    })

    numeric_cols = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    input_data[numeric_cols] = scaler.transform(
        input_data[numeric_cols]
    )

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)

    st.success(
        f"### 💰 Estimated Car Price : ₹ {prediction[0]:,.2f}"
    )

st.divider()

st.caption("© 2026 Ford Car Price Prediction | Built using Streamlit & Scikit-Learn")