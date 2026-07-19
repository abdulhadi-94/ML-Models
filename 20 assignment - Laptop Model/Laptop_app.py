import streamlit as st
import pandas as pd
import joblib

df = pd.read_csv("laptop_price - dataset.csv")

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="laptop logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("background.jpg")

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
model = joblib.load("LR_Laptop_price.pkl")
scaler = joblib.load("scaler_laptop.pkl")
encoded_columns = joblib.load("columns_laptop.pkl")


# ---------------- Sidebar ----------------
st.sidebar.image(
    "laptop logo.jpg",
    width=180,
)

st.sidebar.title("About")

st.sidebar.info(
"""
### Laptop Price Prediction

Enter laptop details to predict the estimated price using a trained Machine Learning model.

Developer:
Abdul Hadi Shaikh

GitHub Link:
https://github.com/abdulhadi-94/ML-Models.git
"""
)

logo = get_base64("laptop logo.jpg")

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:center; gap:15px;">
    <img src="data:image/png;base64,{logo}" width="90">
    <h1 style="color:#FFFFFF; margin:0;">
          Laptop Price Prediction
    </h1>
</div>
""", unsafe_allow_html=True)

# ---------------- Header ----------------

st.write("Fill all laptop details below and click **Predict Price**.")

st.divider()

#----------------- Input Section --------------

# Two Columns
col1, col2 = st.columns(2)

# ---------------- LEFT COLUMN ----------------
with col1:
    company = st.selectbox(
        "🏢 Company",
        sorted(df["Company"].unique())
    )

    product = st.selectbox(
        "💻 Product",
        sorted(df["Product"].unique())
    )

    typename = st.selectbox(
        "📂 Laptop Type",
        sorted(df["TypeName"].unique())
    )

    inches = st.number_input(
        "📏 Screen Size (Inches)",
        min_value=10.0,
        max_value=20.0,
        value=15.6,
        step=0.1
    )

    screen = st.selectbox(
        "🖥 Screen Resolution",
        sorted(df["ScreenResolution"].unique())
    )

    ram = st.selectbox(
        "🧠 RAM (GB)",
        sorted(df["RAM (GB)"].unique())
    )

    memory = st.selectbox(
        "💾 Storage",
        sorted(df["Memory"].unique())
    )

# ---------------- RIGHT COLUMN ----------------
with col2:

    cpu_company = st.selectbox(
        "⚙ CPU Company",
        sorted(df["CPU_Company"].unique())
    )

    cpu_type = st.selectbox(
        "🚀 CPU Model",
        sorted(df["CPU_Type"].unique())
    )

    cpu_freq = st.number_input(
        "⚡ CPU Speed (GHz)",
        min_value=0.5,
        max_value=6.0,
        value=2.5,
        step=0.1
    )

    gpu_company = st.selectbox(
        "🎮 GPU Company",
        sorted(df["GPU_Company"].unique())
    )

    gpu_type = st.selectbox(
        "🖥 GPU Model",
        sorted(df["GPU_Type"].unique())
    )

    os = st.selectbox(
        "🪟 Operating System",
        sorted(df["OpSys"].unique())
    )

    weight = st.number_input(
        "⚖ Weight (kg)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1
    )

# ---------------- Prediction ----------------

if st.button("💻 Predict Price"):

    input_data = pd.DataFrame({
        "Company":[company],
        "Product":[product],
        "TypeName":[typename],
        "Inches":[inches],
        "ScreenResolution":[screen],
        "RAM (GB)":[ram],
        "Memory":[memory],
        "CPU_Company":[cpu_company],
        "CPU_Type":[cpu_freq],
        "CPU_Frequency (GHz)":[cpu_freq],
        "GPU_Company":[gpu_company],
        "GPU_Type":[gpu_type],
        "OpSys":[os],
        "Weight (kg)":[weight]
    })

    numeric_cols = [
        'Inches', 'CPU_Frequency (GHz)', 'RAM (GB)', 'Weight (kg)'
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
        f"### 💰 Estimated Laptop Price : EUR {prediction[0]:,.2f}"
    )

st.divider()

st.caption("© 2026 Ford Car Price Prediction | Built using Streamlit & Scikit-Learn")
