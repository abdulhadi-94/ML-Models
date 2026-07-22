from pathlib import Path
BASE_DIR = Path(__file__).parent

import streamlit as st
import pandas as pd
import joblib
import base64

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="heart_logo.jpg",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Background ----------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64(BASE_DIR / "heart_bg.jpg")
logo = get_base64(BASE_DIR / "heart_logo.jpg")
st.markdown(f"""
<style>

/* Background */

.stApp{{
    background-image:url("data:image/png;base64,{img}");
    background-size:cover;
    background-repeat:no-repeat;
    background-position:center;
    background-attachment:fixed;
    animation:bgMove 25s ease-in-out infinite alternate;
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
background:#d62828;
color:white;
border-radius:12px;
border:none;
}}

.stButton>button:hover{{
background:#b00020;
}}

div[data-testid="stMetric"]{{
background:white;
padding:15px;
border-radius:12px;
}}

@media (max-width:768px){{

.block-container{{
padding:15px;
}}

h1{{
font-size:28px!important;
text-align:center;
}}

.stButton>button{{
height:50px;
font-size:18px;
}}

}}

</style>
""",unsafe_allow_html=True)

# ---------------- Load Model ----------------

model = joblib.load(BASE_DIR/"LR_Heart_disease.pkl")
scaler = joblib.load(BASE_DIR/"scaler_Heart.pkl")
columns = joblib.load(BASE_DIR/"columns_Heart.pkl")

# ---------------- Sidebar ----------------

st.sidebar.title("❤️ Heart Disease Prediction")

st.sidebar.info("""
Enter patient details to predict whether the patient has heart disease.

Developer:

Abdul Hadi Shaikh
                
GitHub Link:
https://github.com/abdulhadi-94/ML-Models.git
""")

# ---------------- Title ----------------

st.title("❤️ Heart Disease Prediction System")

st.write("Fill all patient details below and click **Predict**.")

st.divider()

# ---------------- Inputs ----------------

col1,col2=st.columns(2)

with col1:

    Age=st.number_input("Age",18,100,45)

    Sex=st.selectbox(
        "Sex",
        ["M","F"]
    )

    ChestPainType=st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","ASY","TA"]
    )

    RestingBP=st.number_input(
        "Resting Blood Pressure",
        50,
        250,
        120
    )

    Cholesterol=st.number_input(
        "Cholesterol",
        0,
        700,
        200
    )

with col2:

    FastingBS=st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )

    RestingECG=st.selectbox(
        "Resting ECG",
        ["Normal","LVH","ST"]
    )

    MaxHR=st.number_input(
        "Maximum Heart Rate",
        50,
        250,
        150
    )

    ExerciseAngina=st.selectbox(
        "Exercise Angina",
        ["Y","N"]
    )

    Oldpeak=st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

    ST_Slope=st.selectbox(
        "ST Slope",
        ["Up","Flat","Down"]
    )

st.write("")

# ---------------- Prediction ----------------

if st.button("❤️ Predict Heart Disease"):

    sample=pd.DataFrame({

        "Age":[Age],
        "Sex":[Sex],
        "ChestPainType":[ChestPainType],
        "RestingBP":[RestingBP],
        "Cholesterol":[Cholesterol],
        "FastingBS":[FastingBS],
        "RestingECG":[RestingECG],
        "MaxHR":[MaxHR],
        "ExerciseAngina":[ExerciseAngina],
        "Oldpeak":[Oldpeak],
        "ST_Slope":[ST_Slope]

    })

    sample=pd.get_dummies(sample)

    sample=sample.reindex(
        columns=columns,
        fill_value=0
    )

    numeric_cols=[
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak"
    ]

    sample[numeric_cols]=scaler.transform(sample[numeric_cols])

    prediction=model.predict(sample)

    if prediction[0]==1:

        st.error("⚠️ Heart Disease Detected")

    else:

        st.success("✅ No Heart Disease Detected")

st.divider()

st.caption("© 2026 Heart Disease Prediction | Built using Streamlit & Scikit-Learn")