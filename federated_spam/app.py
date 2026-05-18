"""Streamlit dashboard for the federated spam detection demo."""

from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from model import (
    CLIENT_IDS,
    aggregate_client_updates,
    build_model,
    client_update,
    format_percent,
    load_spam_dataset,
    score_message,
    split_dataset_for_clients,
)


DATASET_PATH = Path(__file__).parent / "dataset" / "spam.csv"


def _initialize_state() -> None:
    if "global_state" not in st.session_state:
        st.session_state.global_state = build_model()["global_state"]
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    if "round_number" not in st.session_state:
        st.session_state.round_number = 0
    if "threat_level" not in st.session_state:
        st.session_state.threat_level = "LOW"
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "latest_client_updates" not in st.session_state:
        st.session_state.latest_client_updates = []


def _inject_styles() -> None:
    css = """
        <style>
        /* Broad selectors to support multiple Streamlit versions */
        [data-testid="stAppViewContainer"] .stApp,
        .stApp,
        .reportview-container .stApp {
            background:
                radial-gradient(circle at top left, rgba(211, 218, 227, 0.72), transparent 28%),
                radial-gradient(circle at 85% 15%, rgba(193, 203, 216, 0.55), transparent 24%),
                linear-gradient(180deg, #f5f7fb 0%, #edf1f6 100%) !important;
            color: #0f172a !important;
        }

        [data-testid="stAppViewContainer"] .main .block-container,
        .main .block-container,
        .reportview-container .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 1320px !important;
            box-sizing: border-box !important;
        }

        /* rest of original CSS preserved */
        """

    # store CSS for embedding into components.html blocks
    global _INJECTED_CSS
    _INJECTED_CSS = css

    try:
        st.markdown(css + "</style>", unsafe_allow_html=True)
    except Exception:
        # ignore; we'll include CSS inside components.html where needed
        pass


def _escape_line(value: object) -> str:
    return html.escape(str(value), quote=False)


def _append_log(message: str) -> None:
    st.session_state.activity_log.append(message)


def _render_sidebar() -> None:
    threat = st.session_state.get("threat_level", "LOW")
    threat_class = {
        "HIGH": "warn",
        "MEDIUM": "info",
        "LOW": "good",
    }.get(threat, "neutral")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-shell">
            <div class="sidebar-title">Federated Control Room</div>
            <div class="sidebar-panel">
                <div class="sidebar-label">FL Server Status</div>
                <div class="sidebar-value accent">Online</div>
                <div class="sidebar-separator"></div>
                <div class="sidebar-label">Active Clients</div>
                <div class="sidebar-value">4</div>
            </div>
            <div class="sidebar-panel">
                <div class="sidebar-label">Aggregation Round</div>
                <div class="sidebar-value">{st.session_state.round_number}</div>
                <div class="sidebar-separator"></div>
                <div class="sidebar-label">Threat Level</div>
                <div class="sidebar-value {threat_class}">{html.escape(threat)}</div>
            </div>
            <div class="sidebar-panel">
                <div class="sidebar-label">Research Notes</div>
                <div class="muted-copy">A lightweight federated learning demo for SMS spam classification with a clean dashboard-first presentation.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    html_block = """
        <div class="hero-shell">
            <div class="eyebrow">Federated learning research demo</div>
            <h1 class="page-title">Federated Spam Detection</h1>
            <p class="page-subtitle">
                A minimal, premium dashboard for demonstrating client-side learning, secure aggregation,
                and live spam inference. The interface is tuned for clarity, spacing, and professional presentation.
            </p>
        </div>
        """
    components.html((_INJECTED_CSS or "") + html_block, height=160, scrolling=False)


def _render_overview_metrics() -> None:
    model_state = st.session_state.global_state
    metrics = [
        ("Global signal", f"{model_state.mean():.2f}", "Aggregated model strength from all clients"),
        ("Active clients", "4", "Shards participating in the current round"),
        ("Round", str(st.session_state.round_number), "Latest federated aggregation cycle"),
        ("Threat state", st.session_state.get("threat_level", "LOW"), "Current inference severity"),
    ]

    cards = []
    for title, value, note in metrics:
        cards.append(
            f"""
            <div class="metric-card">
                <div>
                    <div class="metric-title">{html.escape(title)}</div>
                    <div class="metric-value">{html.escape(value)}</div>
                </div>
                <div class="metric-note">{html.escape(note)}</div>
            </div>
            """
        )

    html_block = '<div class="metric-grid">' + "".join(cards) + "</div>"
    components.html((_INJECTED_CSS or "") + html_block, height=240, scrolling=False)


def _render_client_cards(client_updates: list[dict]) -> None:
    cards = []
    for update in client_updates:
        parameter_vector = update["parameters"]
        client_id = html.escape(str(update["client_id"]))
        samples = int(update["sample_count"])
        stats = [
            ("Spam signal", parameter_vector[0]),
            ("Keyword density", parameter_vector[1]),
            ("Normalized length", parameter_vector[2]),
            ("Sync score", parameter_vector[3]),
        ]
        rows = []
        for label, value in stats:
            width = int(max(0.0, min(float(value), 1.0)) * 100)
            rows.append(
                f"""
                <div class="stat-row">
                    <div class="panel-row">
                        <div class="stat-label">{html.escape(label)}</div>
                        <div class="stat-value">{value:.2f}</div>
                    </div>
                    <div class="bar-track"><div class="bar-fill" style="width:{width}%"></div></div>
                </div>
                """
            )

        cards.append(
            f"""
            <div class="client-card">
                <div>
                    <div class="client-head">
                        <div class="client-name">Client {client_id}</div>
                        <div class="client-samples">{samples} samples</div>
                    </div>
                    <div class="client-stats">{''.join(rows)}</div>
                </div>
            </div>
            """
        )

    html_block = '<div class="client-grid">' + "".join(cards) + "</div>"
    components.html((_INJECTED_CSS or "") + html_block, height=340, scrolling=True)


def _render_result_section(result: dict) -> None:
    prediction = str(result["prediction"])
    prediction_class = "spam" if prediction == "SPAM" else "ham"
    confidence = int(result["confidence"])
    spam_probability = float(result["spam_probability"])
    ham_probability = float(result["ham_probability"])
    matches = result.get("matches", [])
    keyword_tokens = ", ".join(matches) if matches else "No keyword matches"

    result_html = f"""
    <div class="result-card">
        <div class="result-header">
            <div>
                <div class="eyebrow" style="margin-bottom:0.35rem;">Detection Result</div>
                <div class="prediction-badge {prediction_class}">{html.escape(prediction)}</div>
            </div>
            <div class="pill {'warn' if result['threat_level'] == 'HIGH' else 'info' if result['threat_level'] == 'MEDIUM' else 'good'}">Threat {html.escape(str(result['threat_level']))}</div>
        </div>

        <div class="result-summary">
            <div class="confidence-value">{confidence}%</div>
            <div class="confidence-copy">Confidence score for the current message classification.</div>
            <div class="thin-progress"><span style="width:{confidence}%"></span></div>
        </div>

        <div class="result-metrics">
            <div class="meta-card">
                <div class="panel-label">Spam probability</div>
                <div class="panel-value">{format_percent(spam_probability)}</div>
                <div class="thin-progress"><span style="width:{int(spam_probability * 100)}%"></span></div>
            </div>
            <div class="meta-card">
                <div class="panel-label">Ham probability</div>
                <div class="panel-value">{format_percent(ham_probability)}</div>
                <div class="thin-progress"><span style="width:{int(ham_probability * 100)}%"></span></div>
            </div>
            <div class="meta-card">
                <div class="panel-label">Matched keywords</div>
                <div class="panel-value">{len(matches)}</div>
                <div class="muted-copy">{html.escape(keyword_tokens)}</div>
            </div>
        </div>
    </div>
    """

    workflow_steps = [
        "Client H1-H4 synchronized",
        "Local updates encrypted",
        "Server aggregation complete",
        f"Global model tuned to {st.session_state.global_state.mean():.2f}",
        f"Prediction: {prediction}",
    ]
    workflow_html = "".join(
        f"""
        <div class="timeline-step"><strong>Stage {index}</strong>{html.escape(step)}</div>
        """
        for index, step in enumerate(workflow_steps, start=1)
    )

    keyword_pills = "".join(
        f'<span class="pill neutral">{html.escape(keyword)}</span>' for keyword in matches
    ) or '<span class="pill neutral">No suspicious tokens</span>'

    detail_html = f"""
    <div class="workflow-card">
        <div class="workflow-row">
            <div>
                <div class="eyebrow" style="margin-bottom:0.35rem;">Workflow</div>
                <p class="workflow-copy">Clients send local updates to the FL server, the server aggregates them, and the global model scores the message.</p>
            </div>
            <div class="pill info">Federated pipeline</div>
        </div>
        <div class="workflow-timeline">{workflow_html}</div>
    </div>

    <div class="meta-card" style="margin-top:1rem;">
        <div class="panel-row">
            <div class="panel-label">Matched keywords</div>
            <div class="pill neutral">{len(matches)} hits</div>
        </div>
        <div class="keyword-list">{keyword_pills}</div>
    </div>
    """

    html_block = '<div class="result-grid">' + result_html + detail_html + "</div>"
    components.html((_INJECTED_CSS or "") + html_block, height=520, scrolling=True)


def _render_logs() -> None:
    log_lines = st.session_state.activity_log[-24:]
    log_text = "\n".join(f"• {line}" for line in log_lines) or "Waiting for a detection run..."
    html_block = f"""
        <div class="feed-card">
            <div class="panel-row">
                <div>
                    <div class="eyebrow" style="margin-bottom:0.35rem;">Live Activity Feed</div>
                    <div class="surface-title" style="color:#dbe6f2;">Terminal-style event stream</div>
                </div>
                <div class="pill neutral">Streaming log</div>
            </div>
            <div class="feed-window">{_escape_line(log_text)}</div>
        </div>
        """
    components.html((_INJECTED_CSS or "") + html_block, height=320, scrolling=True)


def _run_demo(message: str) -> dict:
    data = load_spam_dataset(DATASET_PATH)
    shards = split_dataset_for_clients(data, CLIENT_IDS)

    st.session_state.round_number += 1
    _append_log(f"Round {st.session_state.round_number}: message received")

    progress = st.progress(0, text="Initializing federated workflow")
    status_placeholder = st.empty()

    client_updates: list[dict] = []
    steps = [
        "Client H1 connected",
        "Client H2 connected",
        "Client H3 connected",
        "Client H4 connected",
        "Local training started",
        "Encrypting parameters",
        "Sending updates to FL server",
        "Federated aggregation running",
        "Global model synchronized",
        "Spam prediction generated",
    ]

    with st.status("Federated workflow running", expanded=True) as status:
        for index, step in enumerate(steps, start=1):
            status.update(label=step)
            progress.progress(int(index / len(steps) * 100), text=step)
            status_placeholder.markdown(f"<div class='muted-copy'>{html.escape(step)}</div>", unsafe_allow_html=True)
            _append_log(step)
            time.sleep(0.22)

            if step.startswith("Client"):
                client_id = step.split()[1]
                client_updates.append(client_update(shards[client_id], client_id))

        aggregated_state = aggregate_client_updates(client_updates)
        st.session_state.global_state = aggregated_state

        result = score_message(message, aggregated_state)
        st.session_state.threat_level = result["threat_level"]
        st.session_state.latest_result = result
        st.session_state.latest_client_updates = client_updates
        status.update(label="Detection complete", state="complete")
        progress.progress(100, text="Workflow complete")

    _append_log(f"Final prediction: {result['prediction']} with {result['confidence']}% confidence")
    return result


def main() -> None:
    st.set_page_config(
        page_title="Federated Spam Detection Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_state()
    _inject_styles()
    _render_sidebar()

    _render_header()
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 0.95], gap="large")
    with left:
        st.markdown(
            """
            <div class="surface-card soft">
                <div class="surface-head">
                    <div class="surface-title">Message Input</div>
                    <p class="body-copy">Enter a message to evaluate with the federated baseline. The layout is designed to stay compact and readable on both desktop and mobile.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
        message = st.text_area(
            "Message to classify",
            placeholder="Example: You won free bitcoin reward",
            height=130,
            label_visibility="collapsed",
        )
        button_left, button_center, button_right = st.columns([1, 1.15, 1])
        with button_center:
            run_clicked = st.button("Run Detection", type="primary", use_container_width=True)
    with right:
        st.markdown(
            """
            <div class="surface-card soft">
                <div class="surface-head">
                    <div class="surface-title">Research Summary</div>
                    <p class="body-copy">A clean demo of client-side learning, aggregation, and prediction scoring. The presentation favors clarity over visual noise.</p>
                </div>
                <div style="padding:0 1.1rem 1.1rem;">
                    <div class="workflow-track">
                        <div class="timeline-step"><strong>Model</strong>Keyword baseline with federated-style client updates.</div>
                        <div class="timeline-step"><strong>Flow</strong>Clients H1-H4 train locally and share compact parameter vectors.</div>
                        <div class="timeline-step"><strong>Output</strong>Spam probability, threat level, and matched keyword trace.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    _render_overview_metrics()
    st.markdown("<div class='section-shell'></div>", unsafe_allow_html=True)

    if not message.strip():
        st.info("Enter a message to start the federated detection workflow.")
    elif run_clicked:
        result = _run_demo(message)
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        _render_client_cards(st.session_state.latest_client_updates)
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        _render_result_section(result)
    elif st.session_state.latest_result is not None:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        _render_client_cards(st.session_state.latest_client_updates)
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        _render_result_section(st.session_state.latest_result)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    _render_logs()


if __name__ == "__main__":
    main()