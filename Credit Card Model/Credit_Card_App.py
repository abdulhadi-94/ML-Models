from pathlib import Path
BASE_DIR = Path(__file__).parent

import streamlit as st
import pandas as pd
import joblib
import base64

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon=BASE_DIR / "credit_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD BACKGROUND ----------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64(BASE_DIR / "credit_bg.jpg")
logo = get_base64(BASE_DIR / "credit_logo.png")

# ---------------- CSS ----------------

st.markdown(f"""
<style>

/* Background */

.stApp{{
background-image:url("data:image/png;base64,{img}");
background-size:cover;
background-position:center;
background-repeat:no-repeat;
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
background:#1565C0;
color:white;
border:none;
border-radius:12px;
}}

.stButton>button:hover{{
background:#0D47A1;
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

model = joblib.load(BASE_DIR / "LR_Credit_Card.pkl")
scaler = joblib.load(BASE_DIR / "scaler_Credit_Card.pkl")
columns = joblib.load(BASE_DIR / "columns_Credit_Card.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.image(BASE_DIR / "credit_logo.png", width=120)

st.sidebar.title("💳 Credit Card Prediction")

st.sidebar.info("""
Predict whether a customer is likely to default on the next credit card payment.

Developer

Abdul Hadi Shaikh

GitHub:
https://github.com/abdulhadi-94/ML-Models.git
""")

# ---------------- TITLE ----------------

st.title("💳 Credit Card Default Prediction")

st.write(
    "Fill customer details below and click **Predict Default Risk**."
)

st.divider()
# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)

# ================= LEFT COLUMN =================

with col1:

    ID = st.number_input(
        "Customer ID",
        min_value=1,
        value=1,
        step=1
    )

    LIMIT_BAL = st.number_input(
        "Credit Limit",
        min_value=10000,
        max_value=1000000,
        value=50000,
        step=1000
    )

    SEX = st.selectbox(
        "Gender",
        [1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

    EDUCATION = st.selectbox(
        "Education",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Others"
        }[x]
    )

    MARRIAGE = st.selectbox(
        "Marital Status",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Married",
            2: "Single",
            3: "Others"
        }[x]
    )

    AGE = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

# ================= RIGHT COLUMN =================

with col2:

    st.subheader("Payment History")

    PAY_0 = st.selectbox("PAY_0", [-2,-1,0,1,2,3,4,5,6,7,8])
    PAY_2 = st.selectbox("PAY_2", [-2,-1,0,1,2,3,4,5,6,7,8])
    PAY_3 = st.selectbox("PAY_3", [-2,-1,0,1,2,3,4,5,6,7,8])
    PAY_4 = st.selectbox("PAY_4", [-2,-1,0,1,2,3,4,5,6,7,8])
    PAY_5 = st.selectbox("PAY_5", [-2,-1,0,1,2,3,4,5,6,7,8])
    PAY_6 = st.selectbox("PAY_6", [-2,-1,0,1,2,3,4,5,6,7,8])

st.divider()

st.subheader("Bill Amounts")

bill1, bill2 = st.columns(2)

with bill1:

    BILL_AMT1 = st.number_input("Bill Amount 1", value=0)
    BILL_AMT2 = st.number_input("Bill Amount 2", value=0)
    BILL_AMT3 = st.number_input("Bill Amount 3", value=0)

with bill2:

    BILL_AMT4 = st.number_input("Bill Amount 4", value=0)
    BILL_AMT5 = st.number_input("Bill Amount 5", value=0)
    BILL_AMT6 = st.number_input("Bill Amount 6", value=0)

st.divider()

st.subheader("Previous Payments")

pay1, pay2 = st.columns(2)

with pay1:

    PAY_AMT1 = st.number_input("Payment Amount 1", value=0)
    PAY_AMT2 = st.number_input("Payment Amount 2", value=0)
    PAY_AMT3 = st.number_input("Payment Amount 3", value=0)

with pay2:

    PAY_AMT4 = st.number_input("Payment Amount 4", value=0)
    PAY_AMT5 = st.number_input("Payment Amount 5", value=0)
    PAY_AMT6 = st.number_input("Payment Amount 6", value=0)

st.write("")
# ---------------- PREDICTION ----------------

if st.button("💳 Predict Default Risk"):

    sample = pd.DataFrame({

        "ID":[ID],
        "LIMIT_BAL":[LIMIT_BAL],
        "SEX":[SEX],
        "EDUCATION":[EDUCATION],
        "MARRIAGE":[MARRIAGE],
        "AGE":[AGE],

        "PAY_0":[PAY_0],
        "PAY_2":[PAY_2],
        "PAY_3":[PAY_3],
        "PAY_4":[PAY_4],
        "PAY_5":[PAY_5],
        "PAY_6":[PAY_6],

        "BILL_AMT1":[BILL_AMT1],
        "BILL_AMT2":[BILL_AMT2],
        "BILL_AMT3":[BILL_AMT3],
        "BILL_AMT4":[BILL_AMT4],
        "BILL_AMT5":[BILL_AMT5],
        "BILL_AMT6":[BILL_AMT6],

        "PAY_AMT1":[PAY_AMT1],
        "PAY_AMT2":[PAY_AMT2],
        "PAY_AMT3":[PAY_AMT3],
        "PAY_AMT4":[PAY_AMT4],
        "PAY_AMT5":[PAY_AMT5],
        "PAY_AMT6":[PAY_AMT6]

    })

    # Match training columns
    sample = sample.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale numeric columns
    numeric_cols = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6"
    ]

    sample[numeric_cols] = scaler.transform(sample[numeric_cols])

    prediction = model.predict(sample)

    probability = model.predict_proba(sample)[0]

    default_probability = probability[1] * 100
    safe_probability = probability[0] * 100

    st.write("")

    if prediction[0] == 1:

        st.error(
            f"""
⚠️ High Risk of Credit Card Default

Risk Score: {default_probability:.2f}%
"""
        )

    else:

        st.success(
            f"""
✅ Low Risk of Credit Card Default

Confidence: {safe_probability:.2f}%
"""
        )

st.divider()

st.caption("© 2026 Credit Card Default Prediction | Built using Streamlit & Scikit-Learn")