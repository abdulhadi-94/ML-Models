from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Multi Model Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Helper Function
# ----------------------------------------------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64(BASE_DIR / "bg.jpg")
logo = get_base64(BASE_DIR / "logo.png")

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

st.markdown(f"""
<style>

.stApp {{
background-image:url("data:image/jpg;base64,{bg}");
background-size:cover;
background-attachment:fixed;
}}

[data-testid="stHeader"]{{
background:rgba(0,0,0,0);
}}

[data-testid="stSidebar"]{{
background:rgba(10,25,41,0.95);
}}

[data-testid="stSidebar"] *{{
color:white;
}}

.main-card{{
background:rgba(255,255,255,.90);
padding:30px;
border-radius:20px;
box-shadow:0px 0px 20px rgba(0,0,0,.30);
}}

.result-card{{
background:#0E4D92;
color:white;
padding:25px;
border-radius:15px;
text-align:center;
font-size:28px;
font-weight:bold;
margin-top:20px;
}}

.stButton>button{{
width:100%;
height:55px;
border:none;
border-radius:12px;
background:#1565C0;
color:white;
font-size:18px;
font-weight:bold;
}}

.stButton>button:hover{{
background:#0B3D91;
}}

h1,h2,h3,h4{{
color:#0E4D92;
}}

footer{{
visibility:hidden;
}}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.image(str(BASE_DIR/"logo.png"), width=180)

st.sidebar.title("🤖 Multi Model")

st.sidebar.markdown("---")

page = st.sidebar.radio(

"Navigation",

[
    "🏠 Home",
    "📊 Classification",
    "💻 Regression",
    "ℹ About"
]

)

st.sidebar.markdown("---")

st.sidebar.success(
"""
Machine Learning Algorithms

✔ Logistic Regression

✔ Decision Tree

✔ SVM

✔ KNN

✔ Naive Bayes

✔ Linear Regression

✔ Decision Tree Regressor

✔ SVR

✔ KNN Regressor
"""
)

st.sidebar.title("About")

st.sidebar.info(
"""
### Multi Model Prediction App

  This App can predict Both Datasets and by all the algorithms.

Developer:
Abdul Hadi Shaikh

GitHub Link:
https://github.com/abdulhadi-94/ML-Models.git
"""
)

# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------

if page=="🏠 Home":

    st.image(str(BASE_DIR/"logo.png"), width=250)

    st.markdown(
    """
    <div class='main-card'>

    <h1 align='center'>
    🤖 Multi Model Prediction System
    </h1>

    <hr>

    <h3>Welcome 👋</h3>

    <p style="font-size:18px;">

    This application demonstrates multiple Machine Learning
    algorithms for both

    ✔ Classification

    ✔ Regression

    Users can choose an algorithm,
    enter the required details,
    and obtain predictions instantly.

    </p>

    </div>

    """,

    unsafe_allow_html=True

    )

    col1,col2=st.columns(2)

    with col1:

        st.info("""

### Classification

Dataset

• Telco Customer Churn

Algorithms

• Logistic Regression

• Decision Tree

• Support Vector Machine

• KNN

• Naive Bayes

""")

    with col2:

        st.info("""

### Regression

Dataset

• Laptop Price

Algorithms

• Linear Regression

• Decision Tree Regressor

• Support Vector Regressor

• KNN Regressor

""")

# ----------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------

elif page=="ℹ About":

    st.image(str(BASE_DIR/"logo.png"), width=180)

    st.markdown(
    """
<div class='main-card'>

# About

This project was developed as part of an AIML Assignment.

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Joblib

### Features

✔ Multiple ML Models

✔ Interactive Dashboard

✔ Modern UI

✔ Fast Prediction

✔ Responsive Design

</div>
""",
unsafe_allow_html=True
)

# ----------------------------------------------------
# Classification and Regression
# ----------------------------------------------------
# ----------------------------------------------------
# CLASSIFICATION PAGE
# ----------------------------------------------------

elif page == "📊 Classification":

    # Load Dataset
    df = pd.read_csv(BASE_DIR / "Telco-Customer-Churn.csv")

    st.markdown("""
    <div class='main-card'>
    <h2>📊 Customer Churn Prediction</h2>
    <p>Select an algorithm and enter customer details.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # -----------------------------
    # Model Selection
    # -----------------------------

    algorithm = st.selectbox(
        "🤖 Select Classification Algorithm",
        [
            "Logistic Regression",
            "Decision Tree",
            "Support Vector Machine",
            "K-Nearest Neighbors",
            "Naive Bayes"
        ]
    )

    model_dict = {
        "Logistic Regression": "logistic.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Support Vector Machine": "svm.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl"
    }

    # -----------------------------
    # Load Model
    # -----------------------------

    model = joblib.load(BASE_DIR / model_dict[algorithm])

    scaler = joblib.load(BASE_DIR / "classification_scaler.pkl")

    encoded_columns = joblib.load(BASE_DIR / "classification_columns.pkl")

    st.write("---")

    st.subheader("📝 Customer Details")

    col1, col2 = st.columns(2)

    # -----------------------------
    # Column 1
    # -----------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            sorted(df["gender"].unique())
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0,1]
        )

        partner = st.selectbox(
            "Partner",
            sorted(df["Partner"].unique())
        )

        dependents = st.selectbox(
            "Dependents",
            sorted(df["Dependents"].unique())
        )

        tenure = st.slider(
            "Tenure",
            0,
            72,
            12
        )

        phone = st.selectbox(
            "Phone Service",
            sorted(df["PhoneService"].unique())
        )

        multiple = st.selectbox(
            "Multiple Lines",
            sorted(df["MultipleLines"].unique())
        )

        internet = st.selectbox(
            "Internet Service",
            sorted(df["InternetService"].unique())
        )

        security = st.selectbox(
            "Online Security",
            sorted(df["OnlineSecurity"].unique())
        )

    # -----------------------------
    # Column 2
    # -----------------------------

    with col2:

        backup = st.selectbox(
            "Online Backup",
            sorted(df["OnlineBackup"].unique())
        )

        protection = st.selectbox(
            "Device Protection",
            sorted(df["DeviceProtection"].unique())
        )

        support = st.selectbox(
            "Tech Support",
            sorted(df["TechSupport"].unique())
        )

        tv = st.selectbox(
            "Streaming TV",
            sorted(df["StreamingTV"].unique())
        )

        movies = st.selectbox(
            "Streaming Movies",
            sorted(df["StreamingMovies"].unique())
        )

        contract = st.selectbox(
            "Contract",
            sorted(df["Contract"].unique())
        )

        billing = st.selectbox(
            "Paperless Billing",
            sorted(df["PaperlessBilling"].unique())
        )

        payment = st.selectbox(
            "Payment Method",
            sorted(df["PaymentMethod"].unique())
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=150.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1500.0
        )

    st.write("")

    predict = st.button("🚀 Predict Customer Churn")

        # ==========================================================
    # Prediction
    # ==========================================================

    if predict:

        # Create input dataframe
        input_data = pd.DataFrame({
            "gender":[gender],
            "SeniorCitizen":[senior],
            "Partner":[partner],
            "Dependents":[dependents],
            "tenure":[tenure],
            "PhoneService":[phone],
            "MultipleLines":[multiple],
            "InternetService":[internet],
            "OnlineSecurity":[security],
            "OnlineBackup":[backup],
            "DeviceProtection":[protection],
            "TechSupport":[support],
            "StreamingTV":[tv],
            "StreamingMovies":[movies],
            "Contract":[contract],
            "PaperlessBilling":[billing],
            "PaymentMethod":[payment],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total]
        })

        # ------------------------------
        # One-Hot Encoding
        # ------------------------------

        input_data = pd.get_dummies(input_data)

        # Match training columns
        input_data = input_data.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # ------------------------------
        # Scale Numeric Features
        # ------------------------------

        numeric_cols = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges"
        ]

        input_data[numeric_cols] = scaler.transform(
            input_data[numeric_cols]
        )

        # ------------------------------
        # Prediction
        # ------------------------------

        prediction = model.predict(input_data)

        # ------------------------------
        # Probability (if supported)
        # ------------------------------

        probability = None

        try:
            probability = model.predict_proba(input_data)[0][1]
        except:
            pass

        st.write("")
        st.write("---")

        # ------------------------------
        # Beautiful Output
        # ------------------------------

        if prediction[0] == 1:

            st.error("⚠ Customer is likely to Churn")

            if probability is not None:

                st.progress(float(probability))

                st.metric(
                    "Churn Probability",
                    f"{probability*100:.2f}%"
                )

        else:

            st.success("✅ Customer is NOT likely to Churn")

            if probability is not None:

                st.progress(float(1-probability))

                st.metric(
                    "Retention Probability",
                    f"{(1-probability)*100:.2f}%"
                )

        # ------------------------------
        # Summary Card
        # ------------------------------

        st.markdown("### 📋 Customer Summary")

        summary = pd.DataFrame({

         "Feature":[
             "Gender",
             "Senior Citizen",
             "Partner",
             "Dependents",
             "Tenure",
             "Phone Service",
             "Multiple Lines",
             "Internet Service",
             "Online Security",
             "Online Backup",
             "Device Protection",
             "Tech Support",
             "Streaming TV",
             "Streaming Movies",
             "Contract",
             "Paperless Billing",
             "Payment Method",
             "Monthly Charges",
             "Total Charges"
            ],

         "Value":[
             gender,
             senior,
             partner,
             dependents,
             tenure,
             phone,
             multiple,
             internet,
             security,
             backup,
             protection,
             support,
             tv,
             movies,
             contract,
             billing,
             payment,
             f"${monthly:.2f}",
             f"${total:.2f}"
            ]

        })

        st.dataframe(
          summary,
          use_container_width=True,
          hide_index=True
        )

# ----------------------------------------------------
# REGRESSION PAGE
# ----------------------------------------------------

elif page == "💻 Regression":

    # Load Dataset
    df = pd.read_csv(BASE_DIR / "Laptop_price - dataset.csv")

    st.markdown("""
    <div class='main-card'>
    <h2>💻 Laptop Price Prediction</h2>
    <p>Select an algorithm and enter laptop specifications.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # -----------------------------
    # Model Selection
    # -----------------------------

    algorithm = st.selectbox(
        "🤖 Select Regression Algorithm",
        [
            "Linear Regression",
            "Decision Tree Regressor",
            "Support Vector Regressor",
            "K-Nearest Neighbors Regressor"
        ]
    )

    model_dict = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree Regressor": "decision_tree_regressor.pkl",
        "Support Vector Regressor": "svr.pkl",
        "K-Nearest Neighbors Regressor": "knn_regressor.pkl"
    }

    model = joblib.load(BASE_DIR / model_dict[algorithm])

    scaler = joblib.load(BASE_DIR / "regression_scaler.pkl")

    encoded_columns = joblib.load(BASE_DIR / "regression_columns.pkl")

    st.write("---")
    st.subheader("💻 Laptop Specifications")

    col1, col2 = st.columns(2)

    # -----------------------------
    # LEFT COLUMN
    # -----------------------------

    with col1:

        company = st.selectbox(
            "Company",
            sorted(df["Company"].unique())
        )

        product = st.selectbox(
            "Product",
            sorted(df[df["Company"] == company]["Product"].unique())
        )

        typename = st.selectbox(
            "Type",
            sorted(df["TypeName"].unique())
        )

        inches = st.slider(
            "Screen Size (Inches)",
            10.0,
            20.0,
            15.6
        )

        resolution = st.selectbox(
            "Screen Resolution",
            sorted(df["ScreenResolution"].unique())
        )

        cpu_company = st.selectbox(
            "CPU Company",
            sorted(df["CPU_Company"].unique())
        )

        cpu_type = st.selectbox(
            "CPU Type",
            sorted(df[df["CPU_Company"] == cpu_company]["CPU_Type"].unique())
        )

    # -----------------------------
    # RIGHT COLUMN
    # -----------------------------

    with col2:

        cpu_freq = st.slider(
            "CPU Frequency (GHz)",
            0.9,
            4.5,
            2.5
        )

        ram = st.selectbox(
            "RAM (GB)",
            sorted(df["RAM (GB)"].unique())
        )

        memory = st.selectbox(
            "Memory",
            sorted(df["Memory"].unique())
        )

        gpu_company = st.selectbox(
            "GPU Company",
            sorted(df["GPU_Company"].unique())
        )

        gpu_type = st.selectbox(
            "GPU Type",
            sorted(df[df["GPU_Company"] == gpu_company]["GPU_Type"].unique())
        )

        os = st.selectbox(
            "Operating System",
            sorted(df["OpSys"].unique())
        )

        weight = st.slider(
            "Weight (kg)",
            0.5,
            5.0,
            2.0
        )

    st.write("")

    predict_price = st.button("💰 Predict Laptop Price")

    # ==========================================================
    # Prediction
    # ==========================================================

    if predict_price:

        # Create Input DataFrame
        input_data = pd.DataFrame({
            "Company": [company],
            "Product": [product],
            "TypeName": [typename],
            "Inches": [inches],
            "ScreenResolution": [resolution],
            "CPU_Company": [cpu_company],
            "CPU_Type": [cpu_type],
            "CPU_Frequency (GHz)": [cpu_freq],
            "RAM (GB)": [ram],
            "Memory": [memory],
            "GPU_Company": [gpu_company],
            "GPU_Type": [gpu_type],
            "OpSys": [os],
            "Weight (kg)": [weight]
        })

        # ----------------------------------
        # One-Hot Encoding
        # ----------------------------------

        input_data = pd.get_dummies(input_data)

        # Match Training Columns
        input_data = input_data.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # ----------------------------------
        # Scale Numeric Features
        # ----------------------------------

        numeric_cols = [
            "Inches",
            "CPU_Frequency (GHz)",
            "RAM (GB)",
            "Weight (kg)"
        ]

        input_data[numeric_cols] = scaler.transform(
            input_data[numeric_cols]
        )

        # ----------------------------------
        # Prediction
        # ----------------------------------

        prediction = model.predict(input_data)

        st.write("")
        st.markdown("---")

        # ----------------------------------
        # Beautiful Result Card
        # ----------------------------------

        st.markdown(f"""
        <div class='result-card'>
            💶 Estimated Laptop Price
            <br><br>
            <span style="font-size:40px;">
                € {prediction[0]:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # ----------------------------------
        # Laptop Summary
        # ----------------------------------

        st.markdown("### 💻 Laptop Configuration")

        summary = pd.DataFrame({
            "Feature": [
                "Company",
                "Product",
                "Type",
                "Screen Size",
                "Resolution",
                "CPU",
                "CPU Frequency",
                "RAM",
                "Memory",
                "GPU",
                "Operating System",
                "Weight"
            ],
            "Value": [
                company,
                product,
                typename,
                f"{inches} Inches",
                resolution,
                cpu_type,
                f"{cpu_freq} GHz",
                f"{ram} GB",
                memory,
                gpu_type,
                os,
                f"{weight} kg"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )
st.divider()

st.caption("© 2026 Multi-Model Prediction App | Built using Streamlit & Scikit-Learn")