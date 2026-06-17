import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pickle
import numpy as np
import time
import io

from core.cognis import Cognis
from utils.config import THRESHOLDS

st.set_page_config(page_title="Cognis AI", layout="wide")
st.markdown("<style>.block-container{padding-bottom:50px;}</style>", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("Cognis — Self-Healing AI System")
st.caption("Upload your model and dataset. Cognis will diagnose, fix, and heal your model automatically.")
st.link_button("View on GitHub", url="https://github.com/Anjana2035/Cognis")
st.markdown("---")

# =========================
# SESSION STATE
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []
if "abort" not in st.session_state:
    st.session_state.abort = False
if "result" not in st.session_state:
    st.session_state.result = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None

# =========================
# FILE UPLOAD
# =========================
uploaded_model   = st.file_uploader("Upload your model (.pkl)", type=["pkl"])
uploaded_data    = st.file_uploader("Upload dataset (.npz / .csv / .xlsx)", type=["npz", "csv", "xlsx"])
uploaded_baseline = st.file_uploader("Upload baseline dataset (optional)", type=["npz", "csv", "xlsx"])

gemini_key = st.text_input("Gemini API Key (optional)", type="password")

with st.expander("Dataset Format"):
    st.markdown("""
    - `.npz` — must contain keys `X` and `y`
    - `.csv` / `.xlsx` — last column is treated as the target label
    """)

# =========================
# BUTTONS
# =========================
col1, col2 = st.columns(2)
with col1:
    run_button = st.button("Run Diagnosis")
with col2:
    abort_button = st.button("Abort")

if abort_button:
    st.session_state.abort = True

# =========================
# DATA LOADER
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
    raise ValueError("Unsupported format")

# =========================
# RUN COGNIS
# =========================
if run_button and uploaded_model and uploaded_data:
    st.session_state.abort = False
    st.session_state.chat = []
    st.session_state.result = None

    try:
        model = pickle.load(uploaded_model)
        X, y = load_data(uploaded_data)

        st.session_state.X = X
        st.session_state.y = y

        if uploaded_baseline:
            X_baseline, y_baseline = load_data(uploaded_baseline)
            st.info("Using separate baseline dataset.")
        else:
            X_baseline, y_baseline = X, y
            st.warning("No baseline uploaded — using test data as baseline.")

        with st.spinner("Initialising Cognis..."):
            cognis = Cognis(
                model=model,
                X_baseline=X_baseline,
                y_baseline=y_baseline,
                thresholds=THRESHOLDS,
                api_key=gemini_key or None,
                max_iters=10,
                temperature=1.5
            )

        st.session_state.chat.insert(0, ("user", "Run diagnosis on uploaded model"))

        result = cognis.start_diagnosis(X, y)
        st.session_state.result = result

        for step_data in result["history"]:
            st.session_state.chat.append(("bot", step_data["explanation"]))

        status   = result["final_status"]
        improved = result.get("improved", False)

        if status == "stable" and not improved:
            msg = "Model was already healthy — no fixes needed."
        elif improved:
            before_acc = round(result["baseline_metrics"]["accuracy"], 4)
            after_acc  = round(
                result["history"][-1]["monitoring_after"]["current_metrics"]["accuracy"], 4
            )
            msg = (
                f"Healing complete. Accuracy improved from {before_acc} to {after_acc}. "
                f"You can save or deploy the healed model below."
            )
        else:
            msg = "Cognis attempted all available strategies but could not improve the model. Manual review recommended."

        st.session_state.chat.append(("bot", msg))

    except Exception as e:
        st.error(f"Error: {str(e)}")

# =========================
# SAVE / DEPLOY BUTTONS
# =========================
result = st.session_state.result

if result and result.get("improved"):
    st.markdown("---")
    st.subheader("Model Improved — What would you like to do?")

    save_col, deploy_col = st.columns(2)

    with save_col:
        final_model = result["final_model"]
        model_bytes = io.BytesIO()
        pickle.dump(final_model.model, model_bytes)
        model_bytes.seek(0)

        st.download_button(
            label="Save Healed Model (.pkl)",
            data=model_bytes,
            file_name="cognis_healed_model.pkl",
            mime="application/octet-stream"
        )

        if st.session_state.X is not None:
            npz_bytes = io.BytesIO()
            np.savez(npz_bytes, X=st.session_state.X, y=st.session_state.y)
            npz_bytes.seek(0)

            st.download_button(
                label="Save Dataset (.npz)",
                data=npz_bytes,
                file_name="cognis_dataset.npz",
                mime="application/octet-stream"
            )

    with deploy_col:
        st.markdown("#### Deploy your model")
        st.markdown(
            """
            **Streamlit Community Cloud** — wrap your model in a Streamlit app and deploy for free.
            [share.streamlit.io](https://share.streamlit.io)

            **Hugging Face Spaces** — upload your `.pkl` and a Gradio or Streamlit app to a Space.
            [huggingface.co/spaces](https://huggingface.co/spaces)

            **Render** — deploy a FastAPI wrapper as a REST API on the free tier.
            [render.com](https://render.com)
            """
        )

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
            f"<div style='display:flex;justify-content:flex-start;margin-bottom:10px;'>"
            f"<div style='background-color:#1f2937;padding:10px 14px;border-radius:10px;"
            f"max-width:70%;color:white;'>&#9881; {typed}</div></div>",
            unsafe_allow_html=True
        )
        time.sleep(0.01)

st.markdown("---")
chat_box = st.container(border=True)
with chat_box:
    for role, message in st.session_state.chat:
        if role == "user":
            st.markdown(
                f"<div style='display:flex;justify-content:flex-end;margin-bottom:10px;'>"
                f"<div style='background-color:#2563eb;color:white;padding:10px 14px;"
                f"border-radius:10px;max-width:70%;'>&#128100; {message}</div></div>",
                unsafe_allow_html=True
            )
        else:
            type_writer_chat(message)
            time.sleep(0.2)
