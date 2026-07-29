from pathlib import Path
BASE_DIR = Path(__file__).parent

import streamlit as st
import pandas as pd
import joblib
import base64

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon=BASE_DIR / "loan logo.jpg",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# ---------------- Background ----------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64(BASE_DIR / "loan bg.jpg")
logo = get_base64(BASE_DIR / "loan logo.jpg")
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

model = joblib.load(BASE_DIR/"loan_model.pkl")
scaler = joblib.load(BASE_DIR/"scaler_loan.pkl")
columns = joblib.load(BASE_DIR/"columns_loan.pkl")
# ---------------- Sidebar ----------------
st.sidebar.image(BASE_DIR/"loan logo.jpg", width=120)

st.sidebar.title("🏦 Loan Approval Prediction")

st.sidebar.info("""
Predict whether a loan application is likely to be approved using a Logistic Regression model.

Developer

Abdul Hadi Shaikh

GitHub

https://github.com/abdulhadi-94/ML-Models
""")

# ---------------- Title ----------------

st.title("🏦 Loan Approval Prediction System")

st.write(
    "Fill in the applicant details below and click **Predict Loan Approval**."
)

st.divider()
# ---------------- Inputs ----------------

col1, col2 = st.columns(2)

with col1:

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    Married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    Education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    Self_Employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

with col2:

    ApplicantIncome = st.number_input(
        "Applicant Income",
        0,
        100000,
        5000
    )

    CoapplicantIncome = st.number_input(
        "Coapplicant Income",
        0,
        100000,
        0
    )

    LoanAmount = st.number_input(
        "Loan Amount",
        0,
        1000,
        120
    )

    Loan_Amount_Term = st.number_input(
        "Loan Amount Term",
        0,
        480,
        360
    )

    Credit_History = st.selectbox(
        "Credit History",
        [1, 0]
    )

    Property_Area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

st.write("")

# ---------------- Prediction ----------------

if st.button("🏦 Predict Loan Approval"):

    sample = pd.DataFrame({
        "Gender":[Gender],
        "Married":[Married],
        "Dependents":[Dependents],
        "Education":[Education],
        "Self_Employed":[Self_Employed],
        "ApplicantIncome":[ApplicantIncome],
        "CoapplicantIncome":[CoapplicantIncome],
        "LoanAmount":[LoanAmount],
        "Loan_Amount_Term":[Loan_Amount_Term],
        "Credit_History":[Credit_History],
        "Property_Area":[Property_Area]
    })

    sample = pd.get_dummies(sample)

    sample = sample.reindex(
        columns=columns,
        fill_value=0
    )

    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]

    sample[numeric_cols] = scaler.transform(sample[numeric_cols])

    prediction = model.predict(sample)

    probability = model.predict_proba(sample)[0][1]

    if prediction[0] == 1:

        st.success("✅ Loan Approved")

        st.write(f"Approval Probability : **{probability*100:.2f}%**")

        st.progress(float(probability))

        st.balloons()

    else:

        st.error("❌ Loan Rejected")

        st.write(f"Approval Probability : **{probability*100:.2f}%**")

        st.progress(float(probability))

        st.metric(
            "Approval Probability",
            f"{probability*100:.2f}%"
        )

st.divider()

st.caption("© 2026 Student Loan Prediction | Built using Streamlit & Scikit-Learn")