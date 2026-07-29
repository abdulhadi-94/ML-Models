from pathlib import Path
BASE_DIR = Path(__file__).parent

import streamlit as st
import pandas as pd
import joblib
import base64

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=BASE_DIR / "churn_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD BACKGROUND ----------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64(BASE_DIR / "churn_bg.jpg")
logo = get_base64(BASE_DIR / "churn_logo.png")

# ---------------- CSS ----------------

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
background:#00897B;
color:white;
border:none;
border-radius:12px;
}}

.stButton>button:hover{{
background:#00695C;
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
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load(BASE_DIR / "KN_Customer_Churn.pkl")
scaler = joblib.load(BASE_DIR / "scaler_Customer_Churn.pkl")
columns = joblib.load(BASE_DIR / "columns_Customer_Churn.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.image(BASE_DIR / "churn_logo.png", width=120)

st.sidebar.title("📱 Customer Churn")

st.sidebar.info("""
Predict whether a customer is likely to leave the company.

Developer

Abdul Hadi Shaikh

GitHub:
https://github.com/abdulhadi-94/ML-Models.git
""")

# ---------------- TITLE ----------------

st.title("📱 Customer Churn Prediction System")

st.write(
    "Enter customer information below and click **Predict Churn**."
)

st.divider()
# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)

# ================= LEFT COLUMN =================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

# ================= RIGHT COLUMN =================

with col2:

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

st.divider()

# ================= CONTRACT DETAILS =================

col3, col4 = st.columns(2)

with col3:

    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col4:

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.divider()

# ================= BILLING =================

bill1, bill2 = st.columns(2)

with bill1:

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


st.write("")
# ---------------- PREDICTION ----------------

if st.button("📱 Predict Churn"):

    sample = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[SeniorCitizen],
        "Partner":[Partner],
        "Dependents":[Dependents],
        "tenure":[tenure],
        "PhoneService":[PhoneService],
        "MultipleLines":[MultipleLines],
        "InternetService":[InternetService],
        "OnlineSecurity":[OnlineSecurity],
        "OnlineBackup":[OnlineBackup],
        "DeviceProtection":[DeviceProtection],
        "TechSupport":[TechSupport],
        "StreamingTV":[StreamingTV],
        "StreamingMovies":[StreamingMovies],
        "Contract":[Contract],
        "PaperlessBilling":[PaperlessBilling],
        "PaymentMethod":[PaymentMethod],
        "MonthlyCharges":[MonthlyCharges]
    })

    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=columns, fill_value=0)

    numeric_cols = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges"
    ]

    sample[numeric_cols] = scaler.transform(sample[numeric_cols])

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)[0]

    churn_probability = probability[1] * 100
    stay_probability = probability[0] * 100

    st.write("")

    if prediction[0] == 1:
        st.error(f"""
⚠️ Customer is likely to Churn

Churn Probability: {churn_probability:.2f}%
""")
    else:
        st.success(f"""
✅ Customer is likely to Stay

Confidence: {stay_probability:.2f}%
""")

st.divider()

st.caption("© 2026 Credit Card Default Prediction | Built using Streamlit & Scikit-Learn")