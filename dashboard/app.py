import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pickle
import numpy as np
import time

from core.cognis import Cognis
from utils.config import THRESHOLDS

st.set_page_config(page_title="Cognis AI", layout="wide")

# =========================
# STYLING
# =========================
st.markdown(
    """
    <style>
    .block-container {
        padding-bottom: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================

st.title("Cognis - Self Healing AI System")
st.info("Meet your friendly assistant, Cognis! Upload your model and dataset to run the diagnosis process.")
st.link_button("View on GitHub", url="https://github.com/Anjana2035/Cognis")
st.markdown("---")

# =========================
# SESSION STATE
# =========================

if "chat" not in st.session_state:
    st.session_state.chat = []

if "abort" not in st.session_state:
    st.session_state.abort = False

# =========================
# FILE UPLOAD
# =========================

uploaded_model = st.file_uploader("Upload your model (.pkl)", type=["pkl"])
uploaded_data = st.file_uploader("Upload dataset (.npz / .csv / .xlsx)", type=["npz", "csv", "xlsx"])

# FIX: Optional separate baseline data upload
uploaded_baseline = st.file_uploader(
    "Upload baseline dataset (optional, .npz / .csv / .xlsx)",
    type=["npz", "csv", "xlsx"]
)

with st.expander("Dataset Format Details"):
    st.markdown("""
    **Supported Formats:**
    - `.npz` → must contain 'X' and 'y'
    - `.csv` → last column = target
    - `.xlsx` → last column = target

    **Baseline Dataset (optional):**
    Upload a separate baseline dataset for more accurate drift detection.
    If not provided, the main dataset is used as baseline.
    """)

# =========================
# BUTTONS
# =========================

col1, col2 = st.columns(2)

with col1:
    run_button = st.button("Run Diagnosis")

# FIX: Define abort_button BEFORE chat rendering so it always exists
with col2:
    abort_button = st.button("Abort 🛑")

# FIX: Handle abort outside of chat loop
if abort_button:
    st.session_state.abort = True

# =========================
# DATA LOADER HELPER
# =========================

def load_data(file):
    if file.name.endswith(".npz"):
        data = np.load(file)
        return data["X"], data["y"]

    elif file.name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(file)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    elif file.name.endswith(".xlsx"):
        import pandas as pd
        df = pd.read_excel(file)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    else:
        raise ValueError("Unsupported file format")

# =========================
# RUN COGNIS
# =========================

if run_button and uploaded_model and uploaded_data:

    st.session_state.abort = False
    st.session_state.chat = []

    try:
        model = pickle.load(uploaded_model)

        # Load main data
        X, y = load_data(uploaded_data)

        # FIX: Load baseline separately if provided
        if uploaded_baseline:
            X_baseline, y_baseline = load_data(uploaded_baseline)
            st.info("Using separate baseline dataset for drift detection.")
        else:
            X_baseline, y_baseline = X, y
            st.warning("No baseline uploaded — using main dataset as baseline.")

        cognis = Cognis(
            model=model,
            X_baseline=X_baseline,
            y_baseline=y_baseline,
            thresholds=THRESHOLDS
        )

        result = cognis.start_diagnosis(X, y)

        for step_data in result["history"]:
            st.session_state.chat.append(("bot", step_data["explanation"]))

        st.session_state.chat.append(("bot", f"Final Status: {result['final_status']}"))
        st.session_state.chat.insert(0, ("user", "Run diagnosis on uploaded model"))

    except Exception as e:
        st.error(f"Error: {str(e)}")

# =========================
# CHAT UI
# =========================

def type_writer_chat(message):
    placeholder = st.empty()
    typed = ""

    for char in message:
        if st.session_state.abort:
            return

        typed += char
        placeholder.markdown(
            f"""
            <div style='display:flex; justify-content:flex-start; margin-bottom:10px;'>
                <div style='background-color:#1f2937; padding:10px 14px; border-radius:10px; max-width:70%; color:white;'>
                    ⚙️ {typed}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.01)

chat_box = st.container(border=True)

with chat_box:
    for role, message in st.session_state.chat:

        if role == "user":
            st.markdown(
                f"""
                <div style='display:flex; justify-content:flex-end; margin-bottom:10px;'>
                    <div style='background-color:#2563eb; color:white; padding:10px 14px; border-radius:10px; max-width:70%;'>
                        👤 {message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # FIX: abort_button already defined above, no NameError
            type_writer_chat(message)
            time.sleep(0.2)
