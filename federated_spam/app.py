"""Modern Streamlit dashboard for the federated spam detection demo."""

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
        :root {
            --bg-primary: #eef2f7;
            --bg-secondary: #f7f9fc;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-border: rgba(148, 163, 184, 0.18);
            --surface-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
            --card-bg: linear-gradient(180deg, #102033 0%, #0d1b2b 100%);
            --card-border: rgba(148, 163, 184, 0.16);
            --card-shadow: 0 20px 40px rgba(2, 8, 23, 0.16);
            --text-main: #0f172a;
            --text-soft: #475569;
            --text-muted: #64748b;
            --text-on-dark: #e5edf7;
            --text-on-dark-soft: #a8b6c9;
            --accent: #5b8def;
        }

        html, body, [class*="css"] {
            font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.92), transparent 28%),
                radial-gradient(circle at top right, rgba(191, 219, 254, 0.20), transparent 22%),
                linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
            color: var(--text-main);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .main .block-container {
            max-width: 1320px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1726 0%, #132238 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] .st-emotion-cache-10trblm,
        [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
            color: #e5edf7;
        }

        [data-testid="stSidebarNav"] {
            display: none;
        }

        .sidebar-shell {
            padding-top: 0.25rem;
        }

        .sidebar-kicker {
            color: #8ea4c3;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        .sidebar-title {
            color: #f8fbff;
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 1.1rem;
        }

        .sidebar-panel {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 20px;
            padding: 1rem 1rem 0.15rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            margin-bottom: 0.95rem;
        }

        .sidebar-metric {
            padding-bottom: 0.85rem;
            margin-bottom: 0.85rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }

        .sidebar-metric:last-child {
            margin-bottom: 0;
            border-bottom: 0;
        }

        .sidebar-label {
            color: #8ea4c3;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .sidebar-value {
            color: #f8fbff;
            font-size: 1.42rem;
            font-weight: 600;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }

        .tone-good { color: #73d6a8; }
        .tone-warn { color: #f3c46d; }
        .tone-danger { color: #ff8a8a; }
        .tone-accent { color: #8eb7ff; }

        .sidebar-note {
            color: #9db0c8;
            font-size: 0.86rem;
            line-height: 1.6;
            margin: 0;
        }

        .hero-card,
        .surface-card,
        .empty-state,
        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--surface-border);
            box-shadow: var(--surface-shadow);
            backdrop-filter: blur(18px);
        }

        .hero-card {
            border-radius: 28px;
            padding: 1.5rem 1.55rem;
            margin-bottom: 1.1rem;
            position: relative;
            overflow: hidden;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            top: -3.5rem;
            right: -2rem;
            width: 13rem;
            height: 13rem;
            background: radial-gradient(circle, rgba(91, 141, 239, 0.14), transparent 70%);
            pointer-events: none;
        }

        div[data-testid="stForm"] {
            border-radius: 24px;
            padding: 1.2rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr);
            gap: 1.4rem;
            align-items: end;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.02;
            letter-spacing: -0.05em;
            color: var(--text-main);
            font-weight: 700;
        }

        .hero-copy {
            color: var(--text-soft);
            font-size: 1rem;
            line-height: 1.75;
            max-width: 50rem;
            margin: 0.9rem 0 0;
        }

        .hero-meta-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(244,247,252,0.86));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 22px;
            padding: 1rem 1.05rem;
        }

        .hero-meta-label {
            color: var(--text-muted);
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-meta-value {
            color: var(--text-main);
            font-size: 1.7rem;
            font-weight: 650;
            letter-spacing: -0.04em;
            margin-bottom: 0.4rem;
        }

        .hero-meta-copy {
            color: var(--text-soft);
            font-size: 0.88rem;
            line-height: 1.6;
            margin: 0;
        }

        .surface-card {
            border-radius: 24px;
            padding: 1.25rem;
            height: 100%;
        }

        .card-kicker {
            font-size: 0.72rem;
            color: var(--text-muted);
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .card-title {
            color: var(--text-main);
            font-size: 1.12rem;
            font-weight: 650;
            letter-spacing: -0.03em;
            margin-bottom: 0.45rem;
        }

        .card-copy,
        .research-list,
        .empty-copy {
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.72;
            margin: 0;
        }

        .research-list {
            list-style: none;
            padding: 0;
            margin: 1rem 0 0;
        }

        .research-list li {
            display: flex;
            gap: 0.8rem;
            align-items: flex-start;
            padding: 0.7rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.14);
        }

        .research-list li:first-child {
            border-top: 0;
            padding-top: 0;
        }

        .research-index {
            color: var(--accent);
            font-weight: 650;
            min-width: 1.5rem;
        }

        .section-label {
            color: var(--text-muted);
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 1.6rem 0 0.65rem;
        }

        .section-title {
            color: var(--text-main);
            font-size: 1.3rem;
            font-weight: 650;
            letter-spacing: -0.03em;
            margin: 0 0 0.25rem;
        }

        .section-copy {
            color: var(--text-soft);
            font-size: 0.93rem;
            line-height: 1.7;
            margin: 0 0 1rem;
        }

        .client-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 0.9rem;
        }

        .client-card,
        .result-card,
        .workflow-card,
        .activity-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            box-shadow: var(--card-shadow);
            color: var(--text-on-dark);
        }

        .client-card {
            border-radius: 24px;
            padding: 1.1rem;
            min-height: 280px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 1rem;
            overflow: hidden;
        }

        .client-top {
            display: flex;
            justify-content: space-between;
            gap: 0.9rem;
            align-items: flex-start;
        }

        .client-name {
            font-size: 1rem;
            font-weight: 650;
            color: #f8fbff;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }

        .client-subtitle {
            color: var(--text-on-dark-soft);
            font-size: 0.82rem;
            line-height: 1.5;
        }

        .client-badge {
            color: #dce8f7;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            padding: 0.38rem 0.72rem;
            font-size: 0.75rem;
            white-space: nowrap;
        }

        .client-stats {
            display: grid;
            gap: 0.78rem;
        }

        .metric-line {
            display: grid;
            gap: 0.35rem;
        }

        .metric-head {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
        }

        .metric-label {
            color: #c8d5e6;
            font-size: 0.79rem;
            line-height: 1.4;
            min-width: 0;
            flex: 1;
        }

        .metric-value {
            color: #f8fbff;
            font-size: 0.82rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .mini-track,
        .meter-track,
        .result-progress {
            width: 100%;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
        }

        .mini-track,
        .meter-track {
            height: 6px;
        }

        .result-progress {
            height: 7px;
        }

        .mini-fill,
        .meter-fill,
        .result-fill {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #7eb3ff 0%, #5b8def 100%);
        }

        .result-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.9fr);
            gap: 1rem;
            margin-top: 1rem;
        }

        .result-card,
        .workflow-card,
        .activity-card {
            border-radius: 26px;
            padding: 1.2rem;
        }

        .result-top {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1rem;
        }

        .result-kicker,
        .workflow-kicker,
        .activity-kicker {
            color: #8ea4c3;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .prediction-badge,
        .tone-pill,
        .keyword-pill,
        .workflow-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            font-weight: 600;
            line-height: 1;
        }

        .prediction-badge {
            padding: 0.58rem 0.92rem;
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .prediction-spam {
            color: #fff6f6;
            background: rgba(239, 107, 107, 0.16);
            border: 1px solid rgba(239, 107, 107, 0.26);
        }

        .prediction-ham {
            color: #effff7;
            background: rgba(53, 179, 126, 0.14);
            border: 1px solid rgba(53, 179, 126, 0.24);
        }

        .tone-pill {
            padding: 0.52rem 0.84rem;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .pill-good {
            color: #eafff5;
            background: rgba(53, 179, 126, 0.16);
            border: 1px solid rgba(53, 179, 126, 0.22);
        }

        .pill-warn {
            color: #fff7df;
            background: rgba(224, 168, 72, 0.15);
            border: 1px solid rgba(224, 168, 72, 0.24);
        }

        .pill-danger {
            color: #fff2f2;
            background: rgba(239, 107, 107, 0.16);
            border: 1px solid rgba(239, 107, 107, 0.24);
        }

        .result-title {
            color: #f8fbff;
            font-size: 1.24rem;
            font-weight: 650;
            letter-spacing: -0.03em;
            margin: 0 0 0.35rem;
        }

        .result-copy,
        .workflow-copy,
        .activity-copy {
            color: var(--text-on-dark-soft);
            font-size: 0.9rem;
            line-height: 1.7;
            margin: 0;
        }

        .result-confidence {
            display: grid;
            gap: 0.8rem;
            padding: 1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .confidence-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
        }

        .confidence-value {
            color: #f8fbff;
            font-size: clamp(2.1rem, 3vw, 2.9rem);
            font-weight: 700;
            letter-spacing: -0.06em;
            line-height: 1;
        }

        .result-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.9rem;
        }

        .probability-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 20px;
            padding: 0.95rem;
        }

        .probability-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.55rem;
        }

        .probability-label {
            color: #c8d5e6;
            font-size: 0.8rem;
            line-height: 1.45;
        }

        .probability-value {
            color: #f8fbff;
            font-size: 1.2rem;
            font-weight: 650;
            letter-spacing: -0.03em;
        }

        .workflow-card {
            display: grid;
            gap: 1rem;
        }

        .workflow-flow {
            display: grid;
            gap: 0.75rem;
        }

        .workflow-step {
            display: grid;
            grid-template-columns: 38px minmax(0, 1fr);
            gap: 0.8rem;
            align-items: start;
            padding: 0.82rem 0;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .workflow-step:first-child {
            border-top: 0;
            padding-top: 0;
        }

        .workflow-index {
            width: 38px;
            height: 38px;
            border-radius: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.82rem;
            font-weight: 650;
            color: #f8fbff;
            background: rgba(91, 141, 239, 0.18);
            border: 1px solid rgba(91, 141, 239, 0.24);
        }

        .workflow-step-title {
            color: #f8fbff;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.22rem;
        }

        .workflow-step-copy {
            color: var(--text-on-dark-soft);
            font-size: 0.84rem;
            line-height: 1.6;
        }

        .keyword-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .keyword-pill,
        .workflow-pill {
            color: #dce8f7;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 0.46rem 0.72rem;
            font-size: 0.78rem;
        }

        .activity-card {
            margin-top: 1rem;
        }

        .activity-window {
            margin-top: 0.95rem;
            max-height: 270px;
            overflow-y: auto;
            padding: 1rem;
            border-radius: 18px;
            background: rgba(2, 8, 23, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.12);
            color: #d6e2f0;
            font-family: "SFMono-Regular", "JetBrains Mono", "Consolas", monospace;
            font-size: 0.82rem;
            line-height: 1.75;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .empty-state {
            border-radius: 26px;
            padding: 1.35rem;
            margin-top: 1rem;
        }

        .empty-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(240px, 0.9fr);
            gap: 1rem;
        }

        .empty-stats {
            display: grid;
            gap: 0.8rem;
        }

        .empty-stat {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.12);
        }

        .empty-stat-label {
            color: var(--text-muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.3rem;
        }

        .empty-stat-value {
            color: var(--text-main);
            font-size: 1.25rem;
            font-weight: 650;
            letter-spacing: -0.03em;
        }

        .stTextArea textarea {
            min-height: 150px !important;
            border-radius: 18px !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            background: rgba(255, 255, 255, 0.88) !important;
            color: var(--text-main) !important;
            line-height: 1.7 !important;
            font-size: 0.97rem !important;
            padding: 1rem 1.05rem !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03);
        }

        .stTextArea textarea:focus {
            border-color: rgba(91, 141, 239, 0.5) !important;
            box-shadow: 0 0 0 1px rgba(91, 141, 239, 0.2) !important;
        }

        div[data-testid="stTextArea"] label {
            color: var(--text-soft) !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }

        div[data-testid="stButton"],
        div[data-testid="stFormSubmitButton"] {
            display: flex;
            justify-content: center;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            max-width: 260px;
            min-height: 50px;
            border-radius: 999px;
            border: 0;
            color: #f8fbff;
            font-size: 0.98rem;
            font-weight: 650;
            letter-spacing: -0.01em;
            background: linear-gradient(135deg, #6a93f7 0%, #507fe7 100%);
            box-shadow: 0 14px 28px rgba(80, 127, 231, 0.26);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        }

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.03);
            box-shadow: 0 18px 30px rgba(80, 127, 231, 0.28);
        }

        div[data-testid="stButton"] > button:focus,
        div[data-testid="stFormSubmitButton"] > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(91, 141, 239, 0.22), 0 18px 30px rgba(80, 127, 231, 0.28);
        }

        .stProgress > div > div {
            height: 0.38rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.22);
        }

        .stProgress > div > div > div {
            border-radius: 999px;
            background: linear-gradient(90deg, #7eb3ff 0%, #5b8def 100%);
        }

        [data-testid="stStatusWidget"] {
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);
        }

        .stAlert {
            border-radius: 18px;
        }

        @media (max-width: 1180px) {
            .client-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .result-grid,
            .hero-grid,
            .empty-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 760px) {
            .main .block-container {
                padding-top: 1.2rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .hero-card,
            .surface-card,
            .result-card,
            .workflow-card,
            .activity-card,
            .empty-state,
            .client-card,
            div[data-testid="stForm"] {
                border-radius: 22px;
            }

            .client-grid,
            .result-metric-grid {
                grid-template-columns: 1fr;
            }

            .result-top,
            .confidence-row,
            .probability-head,
            .client-top,
            .metric-head {
                flex-direction: column;
                align-items: flex-start;
            }

            div[data-testid="stButton"] > button,
            div[data-testid="stFormSubmitButton"] > button {
                max-width: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _append_log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.activity_log.append(f"[{timestamp}] {message}")


def _tone_class(level: str) -> str:
    return {
        "LOW": "pill-good",
        "MEDIUM": "pill-warn",
        "HIGH": "pill-danger",
    }.get(level, "pill-good")


def _sidebar_value_tone(level: str) -> str:
    return {
        "LOW": "tone-good",
        "MEDIUM": "tone-warn",
        "HIGH": "tone-danger",
    }.get(level, "tone-good")


def _load_client_snapshot() -> list[dict]:
    if st.session_state.latest_client_updates:
        return st.session_state.latest_client_updates

    frame = load_spam_dataset(DATASET_PATH)
    shards = split_dataset_for_clients(frame, CLIENT_IDS)
    return [client_update(shards[client_id], client_id) for client_id in CLIENT_IDS]


def _render_sidebar() -> None:
    threat_level = str(st.session_state.threat_level)
    threat_tone = _sidebar_value_tone(threat_level)
    st.sidebar.markdown(
        f"""
        <div class="sidebar-shell">
            <div class="sidebar-kicker">Research console</div>
            <div class="sidebar-title">Federated Spam Detection</div>

            <div class="sidebar-panel">
                <div class="sidebar-metric">
                    <div class="sidebar-label">FL Server Status</div>
                    <div class="sidebar-value tone-good">Online</div>
                </div>
                <div class="sidebar-metric">
                    <div class="sidebar-label">Active Clients</div>
                    <div class="sidebar-value tone-accent">4</div>
                </div>
            </div>

            <div class="sidebar-panel">
                <div class="sidebar-metric">
                    <div class="sidebar-label">Aggregation Round</div>
                    <div class="sidebar-value">{st.session_state.round_number}</div>
                </div>
                <div class="sidebar-metric">
                    <div class="sidebar-label">Threat Level</div>
                    <div class="sidebar-value {threat_tone}">{_escape(threat_level)}</div>
                </div>
            </div>

            <div class="sidebar-panel">
                <p class="sidebar-note">
                    Lightweight federated research dashboard for local client updates,
                    aggregation health, and message classification output.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    global_signal = float(st.session_state.global_state.mean())
    st.markdown(
        f"""
        <section class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Federated learning research demo</div>
                    <h1 class="hero-title">Minimal dashboard for distributed spam detection</h1>
                    <p class="hero-copy">
                        A polished research-facing UI that highlights local client updates,
                        model aggregation, and live inference without visual clutter.
                    </p>
                </div>
                <div class="hero-meta-card">
                    <div class="hero-meta-label">Global signal</div>
                    <div class="hero-meta-value">{global_signal:.2f}</div>
                    <p class="hero-meta-copy">
                        The current aggregated state from four simulated clients, updated every detection round.
                    </p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_client_cards(client_updates: list[dict]) -> None:
    cards: list[str] = []
    stat_labels = [
        "Spam signal",
        "Keyword density",
        "Normalized length",
        "Sync score",
    ]

    for update in client_updates:
        rows: list[str] = []
        for label, value in zip(stat_labels, update["parameters"], strict=True):
            metric_value = float(value)
            width = max(0, min(int(metric_value * 100), 100))
            rows.append(
                f"""
                <div class="metric-line">
                    <div class="metric-head">
                        <div class="metric-label">{_escape(label)}</div>
                        <div class="metric-value">{metric_value:.2f}</div>
                    </div>
                    <div class="mini-track"><span class="mini-fill" style="width:{width}%"></span></div>
                </div>
                """
            )

        cards.append(
            f"""
            <div class="client-card">
                <div class="client-top">
                    <div>
                        <div class="client-name">Client {_escape(update['client_id'])}</div>
                        <div class="client-subtitle">Compact local metrics from the current shard update.</div>
                    </div>
                    <div class="client-badge">{int(update['sample_count'])} samples</div>
                </div>
                <div class="client-stats">{''.join(rows)}</div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="section-label">Client cards</div>
        <h2 class="section-title">Distributed client summary</h2>
        <p class="section-copy">
            Four equal-height client cards show the latest local signals without overflow or spacing issues.
        </p>
        <div class="client-grid">{''.join(cards)}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_result() -> None:
    st.markdown(
        f"""
        <div class="section-label">Detection result</div>
        <h2 class="section-title">Inference analytics</h2>
        <p class="section-copy">Run a message through the workflow to populate the prediction panel.</p>
        <div class="empty-state">
            <div class="empty-grid">
                <div>
                    <div class="card-kicker">Awaiting input</div>
                    <div class="card-title">Dashboard output will appear here</div>
                    <p class="empty-copy">
                        The result area is prepared to show prediction badge, confidence, threat level,
                        probability split, workflow progression, and matched keywords in one clean view.
                    </p>
                </div>
                <div class="empty-stats">
                    <div class="empty-stat">
                        <div class="empty-stat-label">Threat state</div>
                        <div class="empty-stat-value">{_escape(st.session_state.threat_level)}</div>
                    </div>
                    <div class="empty-stat">
                        <div class="empty-stat-label">Aggregation round</div>
                        <div class="empty-stat-value">{st.session_state.round_number}</div>
                    </div>
                    <div class="empty-stat">
                        <div class="empty-stat-label">Global signal</div>
                        <div class="empty-stat-value">{float(st.session_state.global_state.mean()):.2f}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_section(result: dict) -> None:
    prediction = str(result["prediction"])
    prediction_class = "prediction-spam" if prediction == "SPAM" else "prediction-ham"
    threat_level = str(result["threat_level"])
    matches = result.get("matches", [])
    confidence = int(result["confidence"])
    spam_probability = float(result["spam_probability"])
    ham_probability = float(result["ham_probability"])
    keyword_component = float(result.get("keyword_component", 0.0))
    workflow_steps = [
        (
            "Client shards synchronized",
            "H1-H4 summarize local data privately before sharing compact update vectors.",
        ),
        (
            "Federated aggregation completed",
            f"Round {st.session_state.round_number} updates merged into a global signal of {float(st.session_state.global_state.mean()):.2f}.",
        ),
        (
            "Inference scored",
            f"Keyword engine identified {len(matches)} suspicious token{'s' if len(matches) != 1 else ''} in the submitted message.",
        ),
        (
            "Decision published",
            f"Message classified as {prediction} with {confidence}% confidence and {threat_level} threat level.",
        ),
    ]

    keyword_pills = "".join(
        f'<span class="keyword-pill">{_escape(keyword)}</span>' for keyword in matches
    ) or '<span class="keyword-pill">No suspicious keywords matched</span>'

    workflow_html = "".join(
        f"""
        <div class="workflow-step">
            <div class="workflow-index">{index}</div>
            <div>
                <div class="workflow-step-title">{_escape(title)}</div>
                <div class="workflow-step-copy">{_escape(copy)}</div>
            </div>
        </div>
        """
        for index, (title, copy) in enumerate(workflow_steps, start=1)
    )

    st.markdown(
        f"""
        <div class="section-label">Detection result</div>
        <h2 class="section-title">Inference analytics</h2>
        <p class="section-copy">A cleaner result surface for prediction, confidence, workflow, and keyword evidence.</p>

        <div class="result-grid">
            <div class="result-card">
                <div class="result-top">
                    <div>
                        <div class="result-kicker">Prediction output</div>
                        <div class="prediction-badge {prediction_class}">{_escape(prediction)}</div>
                    </div>
                    <div class="tone-pill {_tone_class(threat_level)}">Threat {_escape(threat_level)}</div>
                </div>

                <div class="result-confidence">
                    <div class="confidence-row">
                        <div>
                            <div class="result-title">Confidence</div>
                            <p class="result-copy">Probability-weighted confidence for the final classification.</p>
                        </div>
                        <div class="confidence-value">{confidence}%</div>
                    </div>
                    <div class="result-progress"><span class="result-fill" style="width:{confidence}%"></span></div>
                </div>

                <div class="result-metric-grid">
                    <div class="probability-card">
                        <div class="probability-head">
                            <div class="probability-label">Spam probability</div>
                            <div class="probability-value">{_escape(format_percent(spam_probability))}</div>
                        </div>
                        <div class="meter-track"><span class="meter-fill" style="width:{int(spam_probability * 100)}%"></span></div>
                    </div>

                    <div class="probability-card">
                        <div class="probability-head">
                            <div class="probability-label">Ham probability</div>
                            <div class="probability-value">{_escape(format_percent(ham_probability))}</div>
                        </div>
                        <div class="meter-track"><span class="meter-fill" style="width:{int(ham_probability * 100)}%"></span></div>
                    </div>

                    <div class="probability-card">
                        <div class="probability-head">
                            <div class="probability-label">Threat level</div>
                            <div class="probability-value">{_escape(threat_level)}</div>
                        </div>
                        <div class="meter-track"><span class="meter-fill" style="width:{int(spam_probability * 100)}%"></span></div>
                    </div>

                    <div class="probability-card">
                        <div class="probability-head">
                            <div class="probability-label">Keyword signal</div>
                            <div class="probability-value">{_escape(format_percent(keyword_component))}</div>
                        </div>
                        <div class="meter-track"><span class="meter-fill" style="width:{int(keyword_component * 100)}%"></span></div>
                    </div>
                </div>
            </div>

            <div class="workflow-card">
                <div>
                    <div class="workflow-kicker">Workflow visualization</div>
                    <div class="result-title">Federated decision path</div>
                    <p class="workflow-copy">
                        A lightweight trace from client synchronization through server aggregation and final scoring.
                    </p>
                </div>

                <div class="workflow-flow">{workflow_html}</div>

                <div>
                    <div class="workflow-kicker">Matched keywords</div>
                    <div class="keyword-wrap">{keyword_pills}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_logs() -> None:
    log_lines = st.session_state.activity_log[-24:]
    log_text = "\n".join(log_lines) if log_lines else "[idle] Waiting for a detection run..."
    st.markdown(
        f"""
        <div class="section-label">Live activity feed</div>
        <h2 class="section-title">Terminal stream</h2>
        <p class="section-copy">Minimal event log with monospace typography and a scrollable dark surface.</p>
        <div class="activity-card">
            <div class="activity-kicker">Live activity feed</div>
            <div class="result-title">Federated runtime events</div>
            <p class="activity-copy">Client updates, aggregation steps, and prediction events are streamed here.</p>
            <div class="activity-window">{html.escape(log_text, quote=False)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_demo(message: str) -> None:
    frame = load_spam_dataset(DATASET_PATH)
    shards = split_dataset_for_clients(frame, CLIENT_IDS)

    st.session_state.round_number += 1
    _append_log(f"round {st.session_state.round_number}: message queued for analysis")

    progress = st.progress(0, text="Initializing federated workflow")
    client_updates: list[dict] = []
    aggregated_state = st.session_state.global_state
    result: dict | None = None

    steps = [
        ("Client H1 summarized local shard", "client", "H1"),
        ("Client H2 summarized local shard", "client", "H2"),
        ("Client H3 summarized local shard", "client", "H3"),
        ("Client H4 summarized local shard", "client", "H4"),
        ("Server aggregated encrypted updates", "aggregate", None),
        ("Global model scored the submitted message", "score", None),
        ("Dashboard published the detection result", "complete", None),
    ]

    with st.status("Running federated detection", expanded=True) as status:
        for index, (label, phase, client_id) in enumerate(steps, start=1):
            status.update(label=label, state="running")
            progress.progress(int(index / len(steps) * 100), text=label)
            _append_log(label)

            if phase == "client" and client_id is not None:
                update = client_update(shards[client_id], client_id)
                client_updates.append(update)
                _append_log(f"{client_id}: {int(update['sample_count'])} samples summarized")
            elif phase == "aggregate":
                aggregated_state = aggregate_client_updates(client_updates)
                st.session_state.global_state = aggregated_state
                _append_log(f"global state updated to {float(aggregated_state.mean()):.2f}")
            elif phase == "score":
                result = score_message(message, aggregated_state)
                _append_log(
                    f"prediction {result['prediction']} at {result['confidence']}% confidence"
                )

            time.sleep(0.16)

        status.update(label="Detection complete", state="complete")

    progress.progress(100, text="Workflow complete")

    if result is None:
        result = score_message(message, aggregated_state)

    st.session_state.latest_result = result
    st.session_state.latest_client_updates = client_updates
    st.session_state.threat_level = result["threat_level"]


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

    top_left, top_right = st.columns([1.18, 0.82], gap="large")

    with top_left:
        with st.form("detection_form"):
            st.markdown(
                """
                <div class="card-kicker">Detection workspace</div>
                <div class="card-title">Analyze a message with the federated baseline</div>
                <p class="card-copy">
                    Enter a message to inspect how the four-client simulation updates the shared model and produces a prediction.
                </p>
                """,
                unsafe_allow_html=True,
            )
            message = st.text_area(
                "Message to classify",
                key="message_input",
                placeholder="Example: Congratulations, click now to claim your free reward.",
                label_visibility="collapsed",
            )
            button_left, button_center, button_right = st.columns([1, 1.2, 1])
            with button_center:
                run_clicked = st.form_submit_button("Run Detection", use_container_width=True)

    with top_right:
        st.markdown(
            """
            <div class="surface-card">
                <div class="card-kicker">Research summary</div>
                <div class="card-title">What the dashboard emphasizes</div>
                <p class="card-copy">
                    The interface stays intentionally minimal while surfacing the elements professors and reviewers care about most.
                </p>
                <ul class="research-list">
                    <li><span class="research-index">01</span><span>Client H1-H4 metrics remain compact, readable, and evenly aligned.</span></li>
                    <li><span class="research-index">02</span><span>The result panel combines prediction, confidence, workflow, and evidence in one clean surface.</span></li>
                    <li><span class="research-index">03</span><span>The activity feed keeps the demo grounded in a real engineering workflow.</span></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_clicked:
        if message.strip():
            _run_demo(message.strip())
        else:
            st.warning("Enter a message before running the federated detection workflow.")

    _render_client_cards(_load_client_snapshot())

    if st.session_state.latest_result is None:
        _render_empty_result()
    else:
        _render_result_section(st.session_state.latest_result)

    _render_logs()


if __name__ == "__main__":
    main()
