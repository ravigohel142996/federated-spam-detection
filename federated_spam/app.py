"""Streamlit dashboard for the federated spam detection demo."""

from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st

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
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(211, 218, 227, 0.72), transparent 28%),
                radial-gradient(circle at 85% 15%, rgba(193, 203, 216, 0.55), transparent 24%),
                linear-gradient(180deg, #f5f7fb 0%, #edf1f6 100%);
            color: #0f172a;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111f 0%, #0b1628 45%, #0d1b2d 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        .sidebar-shell {
            padding: 0.25rem 0.25rem 0.5rem;
            color: #e5eef8;
        }

        .sidebar-title {
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 1rem;
        }

        .sidebar-panel {
            background: rgba(10, 18, 32, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 18px 38px rgba(2, 6, 23, 0.3);
        }

        .sidebar-label {
            font-size: 0.72rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .sidebar-value {
            font-size: 1.18rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.1;
        }

        .sidebar-value.accent {
            color: #9fd3ff;
        }

        .sidebar-value.good {
            color: #86efac;
        }

        .sidebar-value.info {
            color: #93c5fd;
        }

        .sidebar-value.warn {
            color: #fda4af;
        }

        .sidebar-value.neutral {
            color: #e2e8f0;
        }

        .sidebar-separator {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.25), transparent);
            margin: 0.9rem 0;
        }

        .hero-shell {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 28px;
            box-shadow: 0 22px 54px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(18px);
            padding: 1.35rem 1.4rem;
        }

        .eyebrow {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.45rem;
        }

        .page-title {
            margin: 0;
            font-size: clamp(1.9rem, 4vw, 3.2rem);
            line-height: 1.02;
            letter-spacing: -0.04em;
            color: #0f172a;
        }

        .page-subtitle {
            margin-top: 0.9rem;
            max-width: 72ch;
            color: #475569;
            font-size: 1rem;
            line-height: 1.65;
        }

        .surface-card {
            background: rgba(9, 17, 30, 0.96);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 24px;
            box-shadow: 0 24px 48px rgba(15, 23, 42, 0.12);
            color: #e5eef8;
            overflow: hidden;
        }

        .surface-card.soft {
            background: rgba(255, 255, 255, 0.78);
            color: #0f172a;
        }

        .surface-head {
            padding: 1rem 1.1rem 0.3rem;
        }

        .surface-title {
            margin: 0;
            font-size: 0.9rem;
            font-weight: 700;
            color: #e2e8f0;
            letter-spacing: 0.03em;
        }

        .surface-card.soft .surface-title,
        .surface-card.soft .body-copy,
        .surface-card.soft .stat-label,
        .surface-card.soft .stat-value,
        .surface-card.soft .muted-copy,
        .surface-card.soft .workflow-copy,
        .surface-card.soft .chip,
        .surface-card.soft .panel-label,
        .surface-card.soft .panel-value {
            color: #0f172a;
        }

        .body-copy {
            color: #cbd5e1;
            line-height: 1.6;
            margin: 0.55rem 0 0;
            font-size: 0.95rem;
        }

        .muted-copy {
            color: #64748b;
            margin: 0;
            line-height: 1.5;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
        }

        .metric-card,
        .client-card,
        .result-card,
        .feed-card,
        .workflow-card,
        .meta-card {
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        }

        .metric-card {
            background: rgba(9, 17, 30, 0.97);
            color: #e5eef8;
            padding: 1rem 1rem 0.95rem;
            min-height: 168px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .metric-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 0.7rem;
        }

        .metric-value {
            font-size: 1.95rem;
            line-height: 1;
            font-weight: 700;
            letter-spacing: -0.04em;
            margin-bottom: 0.3rem;
            overflow-wrap: anywhere;
        }

        .metric-note {
            font-size: 0.92rem;
            color: #cbd5e1;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .client-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
        }

        .client-card {
            background: linear-gradient(180deg, #0c1728 0%, #0a1322 100%);
            color: #e5eef8;
            min-height: 244px;
            padding: 1rem 1rem 1.05rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .client-head,
        .result-header,
        .panel-row,
        .workflow-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .client-name {
            font-size: 1.02rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            overflow-wrap: anywhere;
        }

        .client-samples {
            font-size: 0.8rem;
            color: #94a3b8;
            white-space: nowrap;
        }

        .client-stats {
            display: grid;
            gap: 0.72rem;
            margin-top: 1rem;
        }

        .stat-row {
            display: grid;
            gap: 0.34rem;
        }

        .stat-row .panel-row {
            align-items: baseline;
        }

        .stat-label,
        .panel-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
        }

        .stat-value,
        .panel-value {
            font-size: 0.88rem;
            font-weight: 650;
            color: #f8fafc;
        }

        .bar-track {
            height: 7px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.16);
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #6ba8ff 0%, #8ce3c6 100%);
        }

        .section-shell {
            margin-top: 1.2rem;
        }

        .result-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 1rem;
        }

        .result-card,
        .workflow-card,
        .feed-card,
        .meta-card {
            background: rgba(255, 255, 255, 0.8);
            padding: 1.05rem 1.05rem 1rem;
        }

        .result-card {
            min-height: 100%;
        }

        .prediction-badge,
        .pill,
        .chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 0.42rem 0.7rem;
            white-space: nowrap;
        }

        .prediction-badge.spam {
            background: rgba(239, 68, 68, 0.12);
            color: #b91c1c;
        }

        .prediction-badge.ham {
            background: rgba(34, 197, 94, 0.12);
            color: #15803d;
        }

        .pill.info {
            background: rgba(59, 130, 246, 0.1);
            color: #1d4ed8;
        }

        .pill.neutral {
            background: rgba(100, 116, 139, 0.12);
            color: #475569;
        }

        .pill.good {
            background: rgba(16, 185, 129, 0.12);
            color: #047857;
        }

        .pill.warn {
            background: rgba(245, 158, 11, 0.14);
            color: #b45309;
        }

        .result-summary {
            display: grid;
            gap: 0.45rem;
            margin-top: 1rem;
        }

        .confidence-value {
            font-size: clamp(2.2rem, 4vw, 3.3rem);
            line-height: 0.95;
            letter-spacing: -0.05em;
            font-weight: 750;
            color: #0f172a;
        }

        .confidence-copy {
            color: #475569;
            font-size: 0.94rem;
        }

        .thin-progress {
            width: 100%;
            height: 7px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.18);
            overflow: hidden;
        }

        .thin-progress > span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 100%);
        }

        .result-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }

        .meta-card {
            display: grid;
            gap: 0.75rem;
        }

        .workflow-copy {
            color: #334155;
            line-height: 1.6;
            margin: 0.15rem 0 0;
            overflow-wrap: anywhere;
        }

        .workflow-track {
            display: grid;
            gap: 0.7rem;
            margin-top: 1rem;
        }

        .workflow-node {
            background: rgba(15, 23, 42, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
            color: #0f172a;
            font-size: 0.9rem;
            font-weight: 600;
            overflow-wrap: anywhere;
        }

        .keyword-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }

        .feed-card {
            background: #08111d;
            color: #d9e3ef;
        }

        .feed-window {
            margin-top: 0.9rem;
            background: #050b14;
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            max-height: 280px;
            overflow-y: auto;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.86rem;
            line-height: 1.65;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            color: #cbd5e1;
        }

        .feed-line {
            display: block;
        }

        .workflow-timeline {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 1rem;
        }

        .timeline-step {
            padding: 0.8rem 0.75rem;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #0f172a;
            font-size: 0.83rem;
            line-height: 1.35;
            min-height: 72px;
        }

        .timeline-step strong {
            display: block;
            margin-bottom: 0.35rem;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
        }

        .run-wrap {
            display: flex;
            justify-content: center;
            margin-top: 0.35rem;
        }

        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 18px;
            color: #0f172a;
            padding: 1rem 1rem 0.95rem;
            line-height: 1.6;
            font-size: 0.98rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
        }

        .stTextArea textarea:focus {
            border-color: rgba(37, 99, 235, 0.45);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }

        .stButton > button {
            background: linear-gradient(135deg, #102a43 0%, #1f4b7a 100%);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 999px;
            padding: 0.8rem 1.5rem;
            min-height: 3rem;
            font-size: 0.95rem;
            font-weight: 700;
            box-shadow: 0 18px 30px rgba(15, 23, 42, 0.16);
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
            box-shadow: 0 22px 34px rgba(15, 23, 42, 0.2);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, #2563eb 0%, #14b8a6 100%);
            border-radius: 999px;
        }

        .stProgress [role="progressbar"] {
            height: 7px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.16);
        }

        @media (max-width: 1100px) {
            .metric-grid,
            .client-grid,
            .workflow-timeline {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .result-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 760px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }

            .hero-shell,
            .surface-card.soft,
            .result-card,
            .workflow-card,
            .feed-card,
            .meta-card,
            .client-card,
            .metric-card {
                border-radius: 20px;
            }

            .metric-grid,
            .client-grid,
            .result-metrics,
            .workflow-timeline {
                grid-template-columns: 1fr;
            }

            .page-title {
                font-size: 1.8rem;
            }

            .client-card,
            .metric-card {
                min-height: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    st.markdown(
        """
        <div class="hero-shell">
            <div class="eyebrow">Federated learning research demo</div>
            <h1 class="page-title">Federated Spam Detection</h1>
            <p class="page-subtitle">
                A minimal, premium dashboard for demonstrating client-side learning, secure aggregation,
                and live spam inference. The interface is tuned for clarity, spacing, and professional presentation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    st.markdown('<div class="metric-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


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

    st.markdown('<div class="client-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


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

    st.markdown(
        '<div class="result-grid">' + result_html + detail_html + "</div>",
        unsafe_allow_html=True,
    )


def _render_logs() -> None:
    log_lines = st.session_state.activity_log[-24:]
    log_text = "\n".join(f"• {line}" for line in log_lines) or "Waiting for a detection run..."
    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )


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