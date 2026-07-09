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
uploaded_model    = st.file_uploader("Upload your model (.pkl)", type=["pkl"])
uploaded_data     = st.file_uploader("Upload dataset (.npz / .csv / .xlsx)", type=["npz", "csv", "xlsx"])
uploaded_baseline = st.file_uploader("Upload baseline dataset (optional)", type=["npz", "csv", "xlsx"])


with st.expander("Dataset Format"):
    st.markdown("""
    - `.npz` — must contain keys `X` and `y`
    - `.csv` / `.xlsx` — last column is treated as the target label
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
# SYNTHETIC BASELINE BUILDER
# =========================
def build_synthetic_baseline(X_test, y_test):
    """
    Build a synthetic (X_baseline, y_baseline) that represents a healthy
    model — 85% accuracy, balanced classes, well-calibrated probabilities.

    This is used when no baseline dataset is uploaded.  It gives the
    HealthMonitor a real reference point so all 6 monitoring signals
    can still fire correctly, instead of comparing test data against
    itself (which makes all deltas zero and detection impossible).

    Parameters
    ----------
    X_test : ndarray  — the uploaded evaluation data (used only for shape)
    y_test : ndarray  — used to infer number of classes

    Returns
    -------
    X_baseline : ndarray  (n_synth, n_features)  — zeros, not used by monitor
    y_baseline : ndarray  (n_synth,)             — balanced class labels
    """
    n_classes = len(np.unique(y_test))
    n_synth   = 500
    rng       = np.random.default_rng(42)

    # Balanced class labels cycling through all classes
    y_baseline = np.array(
        [k for k in range(n_classes)] * (n_synth // n_classes + 1)
    )[:n_synth].astype(float)

    # X is irrelevant — HealthMonitor never trains on baseline X, it only
    # uses it through model.evaluate() to get probs.  We pass zeros.
    X_baseline = np.zeros((n_synth, X_test.shape[1]))

    return X_baseline, y_baseline


# =========================
# RUN COGNIS
# =========================
if run_button and uploaded_model and uploaded_data:
    st.session_state.abort = False
    st.session_state.chat  = []
    st.session_state.result = None

    try:
        model = pickle.load(uploaded_model)
        X, y  = load_data(uploaded_data)

        st.session_state.X = X
        st.session_state.y = y

        if uploaded_baseline:
            X_baseline, y_baseline = load_data(uploaded_baseline)
            st.info("Using separate baseline dataset.")
        else:
            # ------------------------------------------------------------------
            # Hardcoded synthetic baseline.
            # Represents a healthy, well-performing model so that the monitor
            # always has a genuine reference point even without a real baseline.
            #
            # How it works:
            #   1. We generate 500 balanced synthetic labels (equal classes).
            #   2. Cognis runs model.evaluate(X_baseline, y_baseline) internally.
            #   3. The model's real predictions on the zero X_baseline won't be
            #      meaningful — but that's fine because we patch ModelInterface
            #      below to intercept the baseline evaluation call and return
            #      our handcrafted healthy probability matrix instead.
            #   4. From step 3, HealthMonitor stores:
            #        - baseline accuracy ≈ 0.85
            #        - baseline entropy  → low (confident model)
            #        - baseline probs    → tight distributions around correct class
            #        - baseline class dist → balanced (equal per class)
            #   5. All subsequent monitoring compares the real test batch
            #      against these stored healthy-model reference values.
            #
            # IMPORTANT: this baseline is ONLY used by HealthMonitor to decide
            # whether the live data looks degraded relative to a healthy
            # reference distribution. It is never used to judge whether
            # healing "improved" the model — that comparison is always made
            # against this run's own starting accuracy (see cognis.py's
            # initial_accuracy / final_accuracy fields), never against this
            # synthetic 85% number.
            # ------------------------------------------------------------------
            X_baseline, y_baseline = build_synthetic_baseline(X, y)

            # Build the synthetic probability matrix:
            # 85% of samples → high confidence (0.90) on correct class
            # 15% of samples → high confidence on wrong class (simulates errors)
            n_classes  = len(np.unique(y))
            n_synth    = len(y_baseline)
            rng        = np.random.default_rng(42)
            base_fill  = 0.05 / max(n_classes - 1, 1)
            synth_probs = np.full((n_synth, n_classes), base_fill)

            for i, cls in enumerate(y_baseline):
                if rng.random() < 0.85:
                    synth_probs[i, int(cls)] = 0.90        # correct + confident
                else:
                    wrong = (int(cls) + 1) % n_classes
                    synth_probs[i, wrong]    = 0.90        # wrong + confident (15%)
            synth_probs /= synth_probs.sum(axis=1, keepdims=True)

            # Patch ModelInterface so the baseline evaluate() call returns
            # our synthetic probs instead of running the model on zero inputs.
            # We restore the original method right after Cognis.__init__().
            from core.model_interface import ModelInterface

            _orig_evaluate = ModelInterface.evaluate

            def _synthetic_evaluate(self_mi, X_in, y_in):
                # Only intercept the baseline call (identified by matching shape)
                if X_in.shape[0] == n_synth:
                    y_pred = np.argmax(synth_probs, axis=1)
                    return {
                        "y_true":         y_in,
                        "y_pred":         y_pred,
                        "probabilities":  synth_probs
                    }
                return _orig_evaluate(self_mi, X_in, y_in)

            ModelInterface.evaluate = _synthetic_evaluate

            st.warning(
                "No baseline uploaded — using hardcoded healthy-model baseline "
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
                temperature=1.5
            )

        # Restore original evaluate after baseline init is done
        if not uploaded_baseline:
            ModelInterface.evaluate = _orig_evaluate

        st.session_state.chat.insert(0, ("user", "Run diagnosis on uploaded model"))

        result = cognis.start_diagnosis(X, y)
        st.session_state.result = result

        for step_data in result["history"]:
            st.session_state.chat.append(("bot", step_data["explanation"]))

        status   = result["final_status"]
        improved = result.get("improved", False)

        # ------------------------------------------------------------------
        # FIX: use result["initial_accuracy"] / result["final_accuracy"]
        # instead of result["baseline_metrics"]["accuracy"] (that's the
        # SYNTHETIC/uploaded reference dataset's accuracy — e.g. 0.974 —
        # not this run's starting accuracy) and instead of
        # result["history"][-1]["monitoring_after"] (that's whatever the
        # LAST attempt scored, even if that attempt was rolled back).
        # cognis.py now computes both fields off the same live-data
        # evaluations it uses internally to decide promote/rollback/best,
        # so they always describe the model actually being handed back.
        # ------------------------------------------------------------------
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

    except Exception as e:
        st.error(f"Error: {str(e)}")

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


# =========================
# STRATEGY MEMORY PANEL  (Objective 4)
# =========================
result = st.session_state.result

if result and result.get("strategy_memory"):
    st.markdown("---")
    st.subheader("Experience-Based Learning — Strategy Win Rates")
    st.caption(
        "Cognis tracks which healing strategies work for each issue type. "
        "On each attempt, strategies are ranked by their historical win-rate "
        "so proven approaches are tried first."
    )

    memory_summary = result["strategy_memory"]

    if not memory_summary:
        st.info("No strategies were attempted in this session.")
    else:
        for issue, strategies in memory_summary.items():
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

# =========================
# SAVE & DEPLOY PANEL
# =========================
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
