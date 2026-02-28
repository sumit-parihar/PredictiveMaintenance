import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MachineGuard",
    page_icon="🛠️",
    layout="wide"
)

# =========================
#        STYLING
# =========================
def load_css():
    st.markdown("""
    <style>

    /* ===== Full Dark Background ===== */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #f1f5f9;
    }

    /* ===== Dark Header ===== */
    header[data-testid="stHeader"] {
        background-color: #0f172a !important;
    }

    header[data-testid="stHeader"] * {
        color: #f1f5f9 !important;
    }

    /* ===== Main Brand Title ===== */
    .brand-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #3b82f6, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 40px;
    }

    /* ===== Section Title ===== */
    .section-title {
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 30px;
        color: #f1f5f9;
    }

    /* ===== Labels ===== */
    label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    /* ===== Input Fields ===== */
    div[data-baseweb="input"] > div {
        background-color: #f8fafc !important;
        color: #111827 !important;
        border-radius: 8px;
    }

    input {
        color: #111827 !important;
        font-weight: 500;
    }

    div[data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        color: #111827 !important;
        border-radius: 8px;
    }

    /* ===== Button Centered Relative to Whole Page ===== */
    .stButton {
        text-align: center;
    }

    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        border-radius: 12px;
        background-color: #3b82f6;
        color: white;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 0 15px rgba(59,130,246,0.6);
        color: white;
    }

    /* ===== Result Boxes ===== */
    .success-box {
        background-color: rgba(34,197,94,0.15);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: #4ade80;
        margin-top: 25px;
        border: 1px solid rgba(34,197,94,0.4);
    }

    .error-box {
        background-color: rgba(239,68,68,0.15);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: #f87171;
        margin-top: 25px;
        border: 1px solid rgba(239,68,68,0.4);
    }

    </style>
    """, unsafe_allow_html=True)

load_css()

# =========================
#        LOGIC
# =========================

# Load Model
model = joblib.load("models/predictive_model.pkl")

# Header
st.markdown('<div class="brand-title">MachineGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Industrial Machine Failure Prediction System</div>', unsafe_allow_html=True)

st.write("")

# Center Layout for Inputs Only
left, center, right = st.columns([1, 2.5, 1])

with center:

    st.markdown('<div class="section-title">Machine Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("Air Temperature [K]", 200.0, 500.0, 300.0, 1.0)
        process_temp = st.number_input("Process Temperature [K]", 200.0, 500.0, 300.0, 1.0)
        rot_speed = st.number_input("Rotational Speed [rpm]", 0, 5000, 1500)

    with col2:
        torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
        tool_wear = st.number_input("Tool Wear [min]", 0, 500, 10)
        product_quality = st.selectbox(
            "Product Quality",
            options=[0, 1, 2],
            format_func=lambda x: ["Low (0)", "Medium (1)", "High (2)"][x]
        )

st.write("")
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    predict_btn = st.button("🔍 Predict Maintenance", use_container_width=True)

# Prediction Logic
if predict_btn:

    input_data = [[air_temp, process_temp, rot_speed, torque, tool_wear, product_quality]]
    prediction = model.predict(input_data)
    result = "Required" if prediction[0] == 1 else "Not Required"

    st.write("")

    if prediction[0] == 1:
        st.markdown('<div class="error-box">⚠️ Maintenance Required</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ Maintenance Not Required</div>', unsafe_allow_html=True)

    # Logging
    log_df = pd.DataFrame(
        [[air_temp, process_temp, rot_speed, torque, tool_wear, product_quality, result]],
        columns=['AirTemp', 'ProcessTemp', 'RotSpeed', 'Torque', 'ToolWear', 'ProductQuality', 'Prediction']
    )

    file_path = "data/prediction_log.csv"

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        start_index = len(existing_df) + 1
    else:
        start_index = 1

    log_df.index = range(start_index, start_index + len(log_df))

    log_df.to_csv(
        file_path,
        mode='a',
        index=True,
        index_label='Serial No.',
        header=not os.path.exists(file_path)
    )