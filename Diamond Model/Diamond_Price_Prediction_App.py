from pathlib import Path
BASE_DIR = Path(__file__).parent

import streamlit as st
import pandas as pd
import joblib
import base64

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon=BASE_DIR / "Diamond_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Background ----------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64(BASE_DIR / "diamond_bg.jpg")
logo = get_base64(BASE_DIR / "Diamond_logo.png")

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
background:#0077ff;
color:white;
border-radius:12px;
border:none;
}}

.stButton>button:hover{{
background:#005fd1;
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

# ---------------- Load Model ----------------

model = joblib.load(BASE_DIR / "LR_Diamond_Price.pkl")
scaler = joblib.load(BASE_DIR / "scaler_Diamond_Price.pkl")
columns = joblib.load(BASE_DIR / "columns_Diamond_Price.pkl")

# ---------------- Sidebar ----------------

st.sidebar.image(BASE_DIR / "Diamond_logo.png", width=120)
st.sidebar.title("💎 Diamond Price Prediction")

st.sidebar.info("""
Predict diamond prices using Machine Learning.

Developer:

Abdul Hadi Shaikh

GitHub:
https://github.com/abdulhadi-94/ML-Models.git
""")

# ---------------- Title ----------------

st.title("💎 Diamond Price Prediction System")

st.write("Enter the diamond details below and click **Predict Price**.")

st.divider()

# ---------------- Inputs ----------------

col1, col2 = st.columns(2)

with col1:

    carat = st.number_input(
        "Carat",
        min_value=0.10,
        max_value=10.00,
        value=1.00,
        step=0.01
    )

    cut = st.selectbox(
        "Cut",
        ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    )

    color = st.selectbox(
        "Color",
        ["D", "E", "F", "G", "H", "I", "J"]
    )

    clarity = st.selectbox(
        "Clarity",
        ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
    )

with col2:

    depth = st.number_input(
        "Depth",
        40.0,
        80.0,
        61.5
    )

    table = st.number_input(
        "Table",
        40.0,
        80.0,
        57.0
    )

    x = st.number_input(
        "Length (x)",
        0.0,
        15.0,
        5.5
    )

    y = st.number_input(
        "Width (y)",
        0.0,
        15.0,
        5.5
    )

    z = st.number_input(
        "Height (z)",
        0.0,
        15.0,
        3.5
    )

st.write("")

# ---------------- Prediction ----------------

if st.button("💎 Predict Diamond Price"):

    sample = pd.DataFrame({

        "carat":[carat],
        "cut":[cut],
        "color":[color],
        "clarity":[clarity],
        "depth":[depth],
        "table":[table],
        "x":[x],
        "y":[y],
        "z":[z]

    })

    sample = pd.get_dummies(sample)

    sample = sample.reindex(
        columns=columns,
        fill_value=0
    )

    numeric_cols = [
        "carat",
        "depth",
        "table",
        "x",
        "y",
        "z"
    ]

    sample[numeric_cols] = scaler.transform(sample[numeric_cols])

    prediction = model.predict(sample)

    st.success(
        f"💎 Estimated Diamond Price: **${prediction[0]:,.2f}**"
    )

st.divider()

st.caption("© 2026 Diamond Price Prediction | Built using Streamlit & Scikit-Learn")