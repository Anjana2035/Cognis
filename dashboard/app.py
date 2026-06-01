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
st.link_button("View on GitHub", url="https://github.com/cognis-ai/cognis")
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

with st.expander("Dataset Format Details"):
    st.markdown("""
    **Supported Formats:**
    - `.npz` → must contain 'X' and 'y'
    - `.csv` → last column = target
    - `.xlsx` → last column = target
    """)

# =========================
# BUTTONS
# =========================

col1, col2 = st.columns(2)

with col1:
    run_button = st.button("Run Diagnosis")


# =========================
# RUN COGNIS
# =========================

if run_button and uploaded_model and uploaded_data:

    # 🔥 Reset abort flag for new run
    st.session_state.abort = False
    st.session_state.chat = []

    try:
        model = pickle.load(uploaded_model)

        # === DATA LOADING ===
        if uploaded_data.name.endswith(".npz"):
            data = np.load(uploaded_data)
            X, y = data["X"], data["y"]

        elif uploaded_data.name.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(uploaded_data)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        elif uploaded_data.name.endswith(".xlsx"):
            import pandas as pd
            df = pd.read_excel(uploaded_data)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        else:
            st.error("Unsupported file format")
            st.stop()

        cognis = Cognis(
            model=model,
            X_baseline=X,
            y_baseline=y,
            thresholds=THRESHOLDS
        )

        # 🔥 RUN WITH MANUAL LOOP CONTROL
        history = []

        for step in range(cognis.max_iters):

            if st.session_state.abort:
                st.session_state.chat.append(("bot", "⚠️ Diagnosis aborted by user."))
                break

            result = cognis.start_diagnosis(X, y)

            for step_data in result["history"]:
                st.session_state.chat.append(("bot", step_data["explanation"]))

            st.session_state.chat.append(("bot", f"Final Status: {result['final_status']}"))
            break  # prevent nested loops explosion

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
                        🧑 {message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with col2:
              abort_button = st.button("Abort 🛑")

        if abort_button:
            st.session_state.abort = True

        else:
            type_writer_chat(message)
            time.sleep(0.2)