import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pickle
import numpy as np
import time
import io

from core.cognis import Cognis
from core.strategy_memory import StrategyMemory
from core.model_interface import ModelInterface
from utils.config import THRESHOLDS

st.set_page_config(page_title="Cognis AI", layout="wide")
st.markdown("<style>.block-container{padding-bottom:50px;}</style>", unsafe_allow_html=True)

st.title("Cognis — Self-Healing AI System")
st.caption("Upload your classification model and dataset. Cognis will diagnose, fix, and heal your model automatically.")
st.link_button("View on GitHub", url="https://github.com/Anjana2035/Cognis")
st.markdown("---")

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
MEMORY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cognis_strategy_memory.json")

if "strategy_memory" not in st.session_state:
    
    if os.path.exists(MEMORY_LOG_PATH):
        with open(MEMORY_LOG_PATH, "r", encoding="utf-8") as f:
            st.session_state.strategy_memory = StrategyMemory.from_json(f.read())
    else:
        st.session_state.strategy_memory = StrategyMemory()


def _persist_strategy_memory():
    
    try:
        with open(MEMORY_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(st.session_state.strategy_memory.to_json())
    except OSError as e:
        st.warning(f"Could not save strategy log: {e}")

uploaded_model    = st.file_uploader("Upload your classification model (.pkl)", type=["pkl"])
uploaded_data     = st.file_uploader("Upload dataset (.npz / .csv / .xlsx)", type=["npz", "csv", "xlsx"])
uploaded_baseline = st.file_uploader("Upload baseline dataset (optional)", type=["npz", "csv", "xlsx"])

with st.expander("Dataset Format"):
    st.markdown("""
    - `.npz` — must contain keys `X` and `y`
    - `.csv` / `.xlsx` — last column is treated as the target label
    - Cognis supports **classification only**. Regression models/targets are rejected.
    """)

with st.expander("Why upload a baseline dataset?"):
    st.markdown("""
    The **baseline dataset** represents what *healthy* model behaviour looks like.
    Cognis uses it to compute reference metrics (accuracy, entropy, confidence
    distribution, class distribution) against which the current evaluation data
    is compared.

    **If you skip it**, Cognis uses a hardcoded synthetic healthy-model baseline:
    85% accuracy, balanced class distribution, and well-calibrated confidence scores.
    This gives the monitor a real reference point so it can still detect degradation.

    **When you should supply one:**
    - You have a held-out slice of data the model was originally validated on.
    - You are simulating drift by evaluating on a shifted dataset.
    - You want monitoring signals (confidence drift, class imbalance) calibrated
      to your specific model's actual healthy behaviour.
    """)

with st.expander("Experience memory"):
    st.markdown(
        "Strategy win-rates accumulate across every diagnosis run in this "
        "browser session (not just within one run's 3 attempts)."
    )
    if st.button("Reset strategy memory"):
        st.session_state.strategy_memory = StrategyMemory()
        _persist_strategy_memory()
        st.success("Strategy memory reset.")
    st.download_button(
        "Download strategy log (.json)",
        data=st.session_state.strategy_memory.to_json(),
        file_name="cognis_strategy_memory.json",
        mime="application/json",
    )

col1, col2 = st.columns(2)
with col1:
    run_button = st.button("Run Diagnosis")
with col2:
    abort_button = st.button("Abort")

if abort_button:
    st.session_state.abort = True

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

def build_synthetic_baseline(X_test, y_test):
    n_classes = len(np.unique(y_test))
    n_synth   = 500
    rng       = np.random.default_rng(42)

    y_baseline = np.array(
        [k for k in range(n_classes)] * (n_synth // n_classes + 1)
    )[:n_synth].astype(float)

    X_baseline = np.zeros((n_synth, X_test.shape[1]))
    return X_baseline, y_baseline


if run_button and uploaded_model and uploaded_data:
    st.session_state.abort = False
    st.session_state.chat  = []
    st.session_state.result = None

    patched = False  
    _orig_evaluate = ModelInterface.evaluate

    try:
        model = pickle.load(uploaded_model)
        X, y  = load_data(uploaded_data)

        st.session_state.X = X
        st.session_state.y = y

        if uploaded_baseline:
            X_baseline, y_baseline = load_data(uploaded_baseline)
            st.info("Using separate baseline dataset.")
        else:
            X_baseline, y_baseline = build_synthetic_baseline(X, y)

            n_classes  = len(np.unique(y))
            n_synth    = len(y_baseline)
            rng        = np.random.default_rng(42)
            base_fill  = 0.05 / max(n_classes - 1, 1)
            synth_probs = np.full((n_synth, n_classes), base_fill)

            for i, cls in enumerate(y_baseline):
                if rng.random() < 0.85:
                    synth_probs[i, int(cls)] = 0.90
                else:
                    wrong = (int(cls) + 1) % n_classes
                    synth_probs[i, wrong]    = 0.90
            synth_probs /= synth_probs.sum(axis=1, keepdims=True)

            def _synthetic_evaluate(self_mi, X_in, y_in):
                if X_in.shape[0] == n_synth:
                    y_pred = np.argmax(synth_probs, axis=1)
                    return {
                        "y_true":        y_in,
                        "y_pred":        y_pred,
                        "probabilities": synth_probs
                    }
                return _orig_evaluate(self_mi, X_in, y_in)

            ModelInterface.evaluate = _synthetic_evaluate
            patched = True

            st.warning(
                "No baseline uploaded. Using hardcoded healthy-model baseline "
                "(85% accuracy, balanced classes, well-calibrated). "
                "For more accurate monitoring, upload a real baseline dataset."
            )

        with st.spinner("Initialising Cognis..."):
            cognis = Cognis(
                model=model,
                X_baseline=X_baseline,
                y_baseline=y_baseline,
                thresholds=THRESHOLDS,
                max_iters=3,
                temperature=1.5,
                memory=st.session_state.strategy_memory,  
            )

        if patched:
            ModelInterface.evaluate = _orig_evaluate
            patched = False

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
            before_acc = round(result["initial_accuracy"], 4)
            after_acc  = round(result["final_accuracy"], 4)
            msg = (
                f"Healing complete. Accuracy improved from {before_acc} to {after_acc}. "
                f"You can save or deploy the healed model below."
            )
        else:
            msg = (
                "Cognis attempted all available strategies but could not improve the model. "
                "Manual review recommended."
            )

        st.session_state.chat.append(("bot", msg))
        _persist_strategy_memory()

    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if patched:
            ModelInterface.evaluate = _orig_evaluate

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


result = st.session_state.result

if result and result.get("strategy_memory"):
    st.markdown("---")
    st.subheader("Experience-Based Learning — Strategy Win Rates")
    st.caption(
        "Cognis tracks which healing strategies work for each issue type, "
        "accumulated across every run in this session. Strategies are "
        "ranked by historical win-rate so proven approaches are tried first."
    )

    memory_summary = result["strategy_memory"]

    if not memory_summary:
        st.info("No strategies were attempted in this session.")
    else:
        for issue, strategies in memory_summary.items():
            if not strategies:
                continue 
            st.markdown(f"**{issue.replace('_', ' ').title()}**")
            cols = st.columns(len(strategies))
            for col, (strategy_name, stats) in zip(cols, strategies.items()):
                win_rate = stats["win_rate"]
                rate_display = f"{win_rate:.0%}" if isinstance(win_rate, float) else win_rate
                col.metric(
                    label=strategy_name.replace("_", " ").title(),
                    value=rate_display,
                    delta=f"{stats['wins']}/{stats['attempts']} wins"
                )

if result:
    st.markdown("---")
    st.subheader("Save or Deploy Your Model")

    improved = result.get("improved", False)
    if improved:
        st.success("Cognis improved this model. Download the healed version below.")
    else:
        st.info(
            "Accuracy did not improve, but you can still download the current "
            "model state or explore deployment options."
        )

    save_col, deploy_col = st.columns(2)

    with save_col:
        st.markdown("#### 💾 Save Model & Dataset")

        final_model = result["final_model"]
        model_bytes = io.BytesIO()
        pickle.dump(final_model.model, model_bytes)
        model_bytes.seek(0)

        st.download_button(
            label="Download Healed Model (.pkl)",
            data=model_bytes,
            file_name="cognis_healed_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

        if st.session_state.X is not None:
            npz_bytes = io.BytesIO()
            np.savez(npz_bytes, X=st.session_state.X, y=st.session_state.y)
            npz_bytes.seek(0)

            st.download_button(
                label="Download Dataset (.npz)",
                data=npz_bytes,
                file_name="cognis_dataset.npz",
                mime="application/octet-stream",
                use_container_width=True,
            )

    with deploy_col:
        st.markdown("#### 🚀 Deploy Your Model")
        st.markdown(
            """
| Platform | Best For | Link |
|---|---|---|
| **Streamlit Community Cloud** | Streamlit apps, free hosting | [share.streamlit.io](https://share.streamlit.io) |
| **Hugging Face Spaces** | Gradio / Streamlit, model cards | [huggingface.co/spaces](https://huggingface.co/spaces) |
| **Render** | FastAPI / REST API, free tier | [render.com](https://render.com) |
| **Railway** | Docker / Python apps, simple setup | [railway.app](https://railway.app) |
| **Google Cloud Run** | Scalable containers, pay-per-use | [cloud.google.com/run](https://cloud.google.com/run) |
| **AWS Lambda** | Serverless inference, low latency | [aws.amazon.com/lambda](https://aws.amazon.com/lambda) |
            """
        )
        st.caption(
            "Tip: wrap your `.pkl` in a FastAPI or Flask app, "
            "then deploy via Docker on any platform above."
        )
