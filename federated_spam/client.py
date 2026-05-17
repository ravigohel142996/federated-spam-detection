"""Streamlit demo client for a federated spam detection research prototype.

This app focuses on visualization and simulation quality while keeping
implementation simple and beginner-friendly.
"""

from __future__ import annotations

import random
import time
from datetime import datetime

import streamlit as st

# ------------------------------
# Basic page setup
# ------------------------------
st.set_page_config(
    page_title="Federated Spam Security Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# Custom styling for a modern dashboard feel.
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        color: #f9fafb;
    }
    .status-card {
        border: 1px solid #2b3648;
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        background-color: #111827;
    }
    .status-title {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .green { background-color: #1b4332; color: #b7f7cf; }
    .yellow { background-color: #5f3f00; color: #ffd166; }
    .red { background-color: #7f1d1d; color: #fecaca; }
    .log-box {
        border: 1px solid #2b3648;
        border-radius: 12px;
        padding: 0.75rem;
        height: 350px;
        overflow-y: auto;
        background-color: #0f172a;
        font-family: monospace;
        font-size: 0.8rem;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Session state for persistent UI
# ------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "client_status" not in st.session_state:
    st.session_state.client_status = {
        "H1": "normal",
        "H2": "normal",
        "H3": "normal",
        "H4": "normal",
    }
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "prediction" not in st.session_state:
    st.session_state.prediction = "No prediction yet"


# ------------------------------
# Helper functions
# ------------------------------
def append_log(message: str) -> None:
    """Add a timestamped line to the live log panel."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")
    # Keep only latest logs to avoid unlimited growth.
    st.session_state.logs = st.session_state.logs[-120:]


def render_status(client_id: str, status: str) -> None:
    """Render one client connection/status card with color coding."""
    color_class = {"normal": "green", "processing": "yellow", "spam": "red"}.get(status, "green")
    label = {
        "normal": "CONNECTED",
        "processing": "PROCESSING",
        "spam": "SPAM PATTERN",
    }.get(status, "CONNECTED")

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">Client {client_id}</div>
            <span class="status-pill {color_class}">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_from_prediction(is_spam: bool) -> str:
    """Map prediction output to dashboard status naming."""
    return "spam" if is_spam else "normal"


# ------------------------------
# Header / controls
# ------------------------------
st.title("🛡️ Federated AI Security Monitoring Dashboard")
st.caption("Research prototype demo • Federated spam detection workflow simulation")

with st.sidebar:
    st.subheader("Simulation Controls")
    input_message = st.text_area(
        "Incoming message to inspect",
        value="Congratulations! You won a free crypto reward. Click now.",
        height=100,
    )
    rounds = st.slider("Federated rounds", min_value=1, max_value=5, value=3)
    run_simulation = st.button("▶ Run Federated Simulation", type="primary")


# ------------------------------
# Layout containers
# ------------------------------
left_col, mid_col, right_col = st.columns([1.1, 1.2, 1.5])

with left_col:
    st.subheader("Client Connectivity")
    for client_name in ["H1", "H2", "H3", "H4"]:
        render_status(client_name, st.session_state.client_status[client_name])

with mid_col:
    st.subheader("Federated Learning Steps")
    step_box = st.empty()
    round_progress = st.progress(0, text="Waiting to start simulation")
    aggregation_progress = st.progress(0, text="Aggregation idle")

with right_col:
    st.subheader("Live System Log")
    log_panel = st.empty()


# ------------------------------
# Simulation flow
# ------------------------------
if run_simulation:
    # Step 1: initialize
    append_log("Server initialized and waiting for client updates")
    step_box.info("Step 1/5: Initializing server and preparing model")

    # Animate overall rounds progress.
    for round_idx in range(1, rounds + 1):
        # Step 2: distribute global model.
        step_box.info(f"Step 2/5: Distributing global model for round {round_idx}")
        append_log(f"Round {round_idx}: global model distributed to H1-H4")

        # Step 3: local client training (simulated).
        step_box.warning(f"Step 3/5: Local training in progress (round {round_idx})")
        for client_name in ["H1", "H2", "H3", "H4"]:
            st.session_state.client_status[client_name] = "processing"
            append_log(f"{client_name} started local training")

        for frame in range(10):
            round_percent = int(((round_idx - 1) + (frame + 1) / 10) / rounds * 100)
            round_progress.progress(round_percent, text=f"Federated round progress: {round_percent}%")
            time.sleep(0.08)

        # Step 4: aggregation from clients to server.
        step_box.warning(f"Step 4/5: Aggregating client updates on server (round {round_idx})")
        append_log(f"Round {round_idx}: server started secure aggregation")
        for frame in range(10):
            aggregation_percent = int((frame + 1) * 10)
            aggregation_progress.progress(
                aggregation_percent,
                text=f"Aggregation H1-H4 → Server: {aggregation_percent}%",
            )
            time.sleep(0.07)

        # Return all clients to normal at end of each round.
        for client_name in ["H1", "H2", "H3", "H4"]:
            st.session_state.client_status[client_name] = "normal"
            append_log(f"{client_name} sent model update successfully")

    # Step 5: run final spam prediction with confidence.
    step_box.success("Step 5/5: Running global model inference")
    spam_keywords = ["free", "winner", "win", "urgent", "click", "crypto", "prize", "offer"]
    score = sum(1 for keyword in spam_keywords if keyword in input_message.lower())
    # Keep logic simple: confidence is a lightweight rule-based estimate.
    confidence = min(95, 30 + score * 12 + random.randint(0, 8))
    is_spam = confidence >= 60

    st.session_state.confidence = confidence
    st.session_state.prediction = "Spam detected" if is_spam else "Normal message"
    final_status = status_from_prediction(is_spam)

    # Reflect final prediction state on one client for visual incident signal.
    st.session_state.client_status["H3"] = final_status

    append_log("Global model inference completed")
    append_log(f"Prediction: {st.session_state.prediction} ({confidence}% confidence)")


# ------------------------------
# Refresh live status + logs
# ------------------------------
with left_col:
    st.subheader("Client Connectivity")
    for client_name in ["H1", "H2", "H3", "H4"]:
        render_status(client_name, st.session_state.client_status[client_name])

with right_col:
    log_text = "\n".join(st.session_state.logs) if st.session_state.logs else "[No events yet]"
    log_panel.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)


# ------------------------------
# Prediction output card
# ------------------------------
st.subheader("Spam Detection Output")

if st.session_state.confidence is None:
    st.info("Run the simulation to generate a prediction confidence score.")
else:
    is_spam = st.session_state.prediction == "Spam detected"
    pill_class = "red" if is_spam else "green"
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">Prediction Result</div>
            <span class="status-pill {pill_class}">{st.session_state.prediction.upper()}</span>
            <div style="margin-top: 0.6rem;">Confidence: <b>{st.session_state.confidence}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
