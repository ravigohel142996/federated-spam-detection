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
MAX_VISIBLE_LOG_LINES = 24

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_dataset_frame():
    return load_spam_dataset(DATASET_PATH)


@st.cache_data(show_spinner=False)
def _load_dataset_shards():
    return split_dataset_for_clients(_load_dataset_frame(), CLIENT_IDS)


@st.cache_data(show_spinner=False)
def _build_default_client_snapshot() -> list[dict]:
    shards = _load_dataset_shards()
    return [client_update(shards[cid], cid) for cid in CLIENT_IDS]


@st.cache_data(show_spinner=False)
def _load_dataset_context() -> dict:
    frame = _load_dataset_frame()
    shards = _load_dataset_shards()
    total_messages = int(len(frame))
    spam_count = int(frame["is_spam"].sum())
    ham_count = int(total_messages - spam_count)
    spam_rate = float(spam_count / total_messages) if total_messages else 0.0
    average_length = float(frame["message_length"].mean()) if total_messages else 0.0
    return {
        "total_messages": total_messages,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_rate": spam_rate,
        "average_length": average_length,
        "client_count": len(shards),
    }


@st.cache_data(show_spinner=False)
def _load_dataset_preview(rows: int = 12):
    frame = _load_dataset_frame()
    return frame.loc[:, ["label", "message"]].head(rows).rename(
        columns={"label": "Label", "message": "Message"}
    )


def _initialize_state() -> None:
    defaults = {
        "global_state": build_model()["global_state"],
        "activity_log": [],
        "round_number": 0,
        "threat_level": "LOW",
        "latest_result": None,
        "latest_client_updates": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "default_client_snapshot" not in st.session_state:
        st.session_state.default_client_snapshot = _build_default_client_snapshot()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _esc(value: object) -> str:
    """HTML-escape a value for safe inline injection."""
    return html.escape(str(value), quote=False)


def _append_log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.activity_log.append(f"[{timestamp}] {message}")


def _threat_pill_class(level: str) -> str:
    return {"LOW": "pill-good", "MEDIUM": "pill-warn", "HIGH": "pill-danger"}.get(
        level, "pill-neutral"
    )


def _threat_tone_class(level: str) -> str:
    return {"LOW": "tone-good", "MEDIUM": "tone-warn", "HIGH": "tone-danger"}.get(
        level, "tone-neutral"
    )


def _load_client_snapshot() -> list[dict]:
    return st.session_state.latest_client_updates or st.session_state.default_client_snapshot


# ---------------------------------------------------------------------------
# Global styles  (one block, no duplicates)
# ---------------------------------------------------------------------------


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Variables ─────────────────────────────────────────────────── */
        :root {
            --bg:          #F5F7FB;
            --surface:     #ffffff;
            --surface-b:   rgba(148,163,184,0.18);
            --surface-sh:  0 2px 12px rgba(15,23,42,0.07), 0 1px 3px rgba(15,23,42,0.04);
            --dark-bg:     linear-gradient(160deg,#0f1e2e 0%,#0d1b2a 100%);
            --dark-b:      rgba(148,163,184,0.13);
            --dark-sh:     0 4px 20px rgba(2,8,23,0.18);
            --txt:         #0f172a;
            --txt-soft:    #475569;
            --txt-muted:   #64748b;
            --txt-dk:      #e5edf7;
            --txt-dk-soft: #a8b6c9;
            --accent:      #4f7fe4;
            --radius-lg:   20px;
            --radius-md:   14px;
            --radius-sm:   10px;
        }

        /* ── Base font ──────────────────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                         "SF Pro Display", sans-serif !important;
        }

        /* ── App background ─────────────────────────────────────────────── */
        .stApp {
            background-color: var(--bg) !important;
            color: var(--txt);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .main .block-container {
            max-width: 1300px;
            padding: 2rem 1.25rem 3rem;
        }

        /* ── Sidebar ────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1726 0%, #132238 100%) !important;
            border-right: 1px solid rgba(148,163,184,0.10);
        }

        [data-testid="stSidebar"] * {
            color: #e5edf7;
        }

        [data-testid="stSidebarNav"] { display: none; }

        .sb-kicker {
            color: #8ea4c3;
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin: 0 0 0.5rem;
        }

        .sb-title {
            color: #f8fbff;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            line-height: 1.3;
            margin: 0 0 1.25rem;
        }

        .sb-desc {
            color: #7e96b4;
            font-size: 0.82rem;
            line-height: 1.65;
            margin: 0 0 1.1rem;
        }

        .sb-panel {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148,163,184,0.10);
            border-radius: var(--radius-md);
            padding: 0.85rem 0.9rem 0.1rem;
            margin-bottom: 0.85rem;
        }

        .sb-row {
            padding-bottom: 0.75rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid rgba(148,163,184,0.10);
        }

        .sb-row:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }

        .sb-label {
            color: #7e96b4;
            font-size: 0.68rem;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .sb-value {
            color: #f8fbff;
            font-size: 1.3rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .sb-footer {
            color: #566e8a;
            font-size: 0.75rem;
            line-height: 1.6;
            margin-top: 1rem;
        }

        .tone-good   { color: #5ed4a0 !important; }
        .tone-warn   { color: #f0c060 !important; }
        .tone-danger { color: #f07070 !important; }
        .tone-accent { color: #7fb3ff !important; }

        /* ── Hero card ──────────────────────────────────────────────────── */
        .hero-wrap {
            background: var(--surface);
            border: 1px solid var(--surface-b);
            border-radius: var(--radius-lg);
            box-shadow: var(--surface-sh);
            padding: 1.5rem 1.6rem 1.4rem;
            margin-bottom: 1.25rem;
        }

        .hero-inner {
            display: grid;
            grid-template-columns: 1fr 260px;
            gap: 1.5rem;
            align-items: end;
        }

        .hero-eyebrow {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 0 0 0.75rem;
        }

        .hero-title {
            font-size: clamp(1.7rem, 2.8vw, 2.5rem);
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 1.08;
            color: var(--txt);
            margin: 0 0 0.8rem;
        }

        .hero-sub {
            color: var(--txt-soft);
            font-size: 0.97rem;
            line-height: 1.72;
            margin: 0;
            max-width: 46rem;
        }

        .signal-card {
            background: #f8fafd;
            border: 1px solid var(--surface-b);
            border-radius: var(--radius-md);
            padding: 1rem 1.1rem 0.9rem;
        }

        .signal-label {
            color: var(--txt-muted);
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 0 0 0.3rem;
        }

        .signal-value {
            color: var(--txt);
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.05em;
            line-height: 1;
            margin: 0 0 0.45rem;
        }

        .signal-note {
            color: var(--txt-soft);
            font-size: 0.83rem;
            line-height: 1.55;
            margin: 0;
        }

        /* ── Input workspace (light card wrapping Streamlit form) ────────── */
        .workspace-card {
            background: var(--surface);
            border: 1px solid var(--surface-b);
            border-radius: var(--radius-lg);
            box-shadow: var(--surface-sh);
            padding: 1.25rem 1.3rem 1.1rem;
            height: 100%;
        }

        .card-eyebrow {
            color: var(--txt-muted);
            font-size: 0.7rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 0 0 0.45rem;
        }

        .card-heading {
            color: var(--txt);
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin: 0 0 0.35rem;
        }

        .card-body {
            color: var(--txt-soft);
            font-size: 0.9rem;
            line-height: 1.7;
            margin: 0 0 1rem;
        }

        /* ── Research summary card ──────────────────────────────────────── */
        .research-card {
            background: var(--surface);
            border: 1px solid var(--surface-b);
            border-radius: var(--radius-lg);
            box-shadow: var(--surface-sh);
            padding: 1.25rem 1.3rem;
            height: 100%;
        }

        .research-list {
            list-style: none;
            padding: 0;
            margin: 0.9rem 0 0;
        }

        .research-list li {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.65rem 0;
            border-top: 1px solid rgba(148,163,184,0.12);
            color: var(--txt-soft);
            font-size: 0.88rem;
            line-height: 1.6;
        }

        .research-list li:first-child {
            border-top: none;
            padding-top: 0;
        }

        .ri {
            color: var(--accent);
            font-weight: 600;
            font-size: 0.8rem;
            min-width: 1.4rem;
            padding-top: 0.05rem;
        }

        /* ── Section headers ────────────────────────────────────────────── */
        .sec-label {
            color: var(--txt-muted);
            font-size: 0.7rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 1.8rem 0 0.5rem;
        }

        .sec-title {
            color: var(--txt);
            font-size: 1.2rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            margin: 0 0 0.2rem;
        }

        .sec-sub {
            color: var(--txt-soft);
            font-size: 0.88rem;
            line-height: 1.65;
            margin: 0 0 0.9rem;
        }

        /* ── Client cards ───────────────────────────────────────────────── */
        .client-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
        }

        .client-card {
            background: var(--dark-bg);
            border: 1px solid var(--dark-b);
            border-radius: var(--radius-lg);
            box-shadow: var(--dark-sh);
            padding: 1rem 1.05rem;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }

        .cc-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.75rem;
        }

        .cc-name {
            color: #f8fbff;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin: 0 0 0.15rem;
        }

        .cc-sub {
            color: var(--txt-dk-soft);
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .cc-badge {
            color: #cdd9eb;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            font-size: 0.72rem;
            white-space: nowrap;
            flex-shrink: 0;
        }

        .cc-stats { display: grid; gap: 0.65rem; }

        .m-row {
            display: grid;
            gap: 0.28rem;
        }

        .m-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .m-label {
            color: #b0c4d8;
            font-size: 0.76rem;
        }

        .m-val {
            color: #f8fbff;
            font-size: 0.77rem;
            font-weight: 600;
        }

        .bar-track {
            width: 100%;
            height: 5px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #7eb3ff 0%, #4f7fe4 100%);
        }

        /* ── Result section ─────────────────────────────────────────────── */
        .result-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.85fr);
            gap: 0.85rem;
        }

        .result-card, .workflow-card {
            background: var(--dark-bg);
            border: 1px solid var(--dark-b);
            border-radius: var(--radius-lg);
            box-shadow: var(--dark-sh);
            padding: 1.15rem 1.2rem;
        }

        .dk-kicker {
            color: #7e96b4;
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 0 0 0.5rem;
        }

        .result-top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .pred-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.5rem 1rem;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .pred-spam {
            color: #ffe4e4;
            background: rgba(239,107,107,0.14);
            border: 1px solid rgba(239,107,107,0.24);
        }

        .pred-ham {
            color: #e4fff2;
            background: rgba(53,179,126,0.12);
            border: 1px solid rgba(53,179,126,0.22);
        }

        .threat-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.42rem 0.8rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .pill-good   { color: #e4fff2; background: rgba(53,179,126,0.14); border: 1px solid rgba(53,179,126,0.22); }
        .pill-warn   { color: #fff8e1; background: rgba(224,168,72,0.13); border: 1px solid rgba(224,168,72,0.22); }
        .pill-danger { color: #fff2f2; background: rgba(239,107,107,0.14); border: 1px solid rgba(239,107,107,0.22); }
        .pill-neutral { color: #e8f0fb; background: rgba(148,163,184,0.13); border: 1px solid rgba(148,163,184,0.22); }

        .conf-block {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: var(--radius-md);
            padding: 0.9rem 1rem;
            margin-bottom: 0.85rem;
        }

        .conf-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.75rem;
            margin-bottom: 0.6rem;
        }

        .conf-label {
            color: #f8fbff;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .conf-sub {
            color: var(--txt-dk-soft);
            font-size: 0.8rem;
            margin: 0;
        }

        .conf-pct {
            color: #f8fbff;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.05em;
            line-height: 1;
        }

        .conf-track {
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            overflow: hidden;
        }

        .conf-fill {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #7eb3ff 0%, #4f7fe4 100%);
        }

        .prob-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
        }

        .prob-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: var(--radius-md);
            padding: 0.75rem 0.8rem;
        }

        .prob-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.45rem;
        }

        .prob-label { color: #a8b6c9; font-size: 0.76rem; }

        .prob-val {
            color: #f8fbff;
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .meter-track {
            width: 100%;
            height: 5px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            overflow: hidden;
        }

        .meter-fill {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #7eb3ff 0%, #4f7fe4 100%);
        }

        /* ── Workflow card ───────────────────────────────────────────────── */
        .workflow-card { display: flex; flex-direction: column; gap: 0.9rem; }

        .wf-title {
            color: #f8fbff;
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin: 0 0 0.2rem;
        }

        .wf-sub {
            color: var(--txt-dk-soft);
            font-size: 0.82rem;
            line-height: 1.6;
            margin: 0;
        }

        .wf-steps { display: flex; flex-direction: column; }

        .wf-step {
            display: grid;
            grid-template-columns: 32px minmax(0, 1fr);
            gap: 0.7rem;
            align-items: start;
            padding: 0.7rem 0;
            border-top: 1px solid rgba(255,255,255,0.07);
        }

        .wf-step:first-child { border-top: none; padding-top: 0; }

        .wf-num {
            width: 32px;
            height: 32px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 600;
            color: #f8fbff;
            background: rgba(79,127,228,0.18);
            border: 1px solid rgba(79,127,228,0.28);
            flex-shrink: 0;
        }

        .wf-step-title {
            color: #f8fbff;
            font-size: 0.87rem;
            font-weight: 600;
            margin: 0 0 0.18rem;
        }

        .wf-step-body {
            color: var(--txt-dk-soft);
            font-size: 0.8rem;
            line-height: 1.55;
        }

        .kw-section { margin-top: 0.1rem; }

        .kw-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.45rem; }

        .kw-pill {
            color: #cdd9eb;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            font-size: 0.75rem;
        }

        /* ── Empty result ────────────────────────────────────────────────── */
        .empty-panel {
            background: var(--surface);
            border: 1px solid var(--surface-b);
            border-radius: var(--radius-lg);
            box-shadow: var(--surface-sh);
            padding: 1.3rem 1.4rem;
        }

        .empty-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) 220px;
            gap: 1rem;
        }

        .empty-heading {
            color: var(--txt);
            font-size: 1rem;
            font-weight: 600;
            margin: 0 0 0.35rem;
        }

        .empty-body {
            color: var(--txt-soft);
            font-size: 0.88rem;
            line-height: 1.7;
            margin: 0;
        }

        .empty-stats { display: grid; gap: 0.6rem; }

        .empty-stat {
            background: #f8fafd;
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: var(--radius-sm);
            padding: 0.75rem 0.8rem;
        }

        .es-label {
            color: var(--txt-muted);
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin: 0 0 0.22rem;
        }

        .es-value {
            color: var(--txt);
            font-size: 1.15rem;
            font-weight: 600;
            letter-spacing: -0.03em;
        }

        /* ── Terminal stream ─────────────────────────────────────────────── */
        .terminal-card {
            background: var(--dark-bg);
            border: 1px solid var(--dark-b);
            border-radius: var(--radius-lg);
            box-shadow: var(--dark-sh);
            padding: 1.15rem 1.2rem;
        }

        .terminal-win {
            margin-top: 0.8rem;
            max-height: 260px;
            overflow-y: auto;
            background: rgba(2,8,23,0.5);
            border: 1px solid rgba(148,163,184,0.10);
            border-radius: var(--radius-sm);
            padding: 0.85rem 0.9rem;
            font-family: ui-monospace, "SF Mono", "SFMono-Regular",
                         "JetBrains Mono", "Consolas", monospace;
            font-size: 0.8rem;
            line-height: 1.75;
            color: #c8d8ea;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* ── Streamlit widget overrides ─────────────────────────────────── */
        .stTextArea textarea {
            min-height: 140px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(148,163,184,0.22) !important;
            background: #fafbfd !important;
            color: var(--txt) !important;
            font-size: 0.95rem !important;
            line-height: 1.65 !important;
            padding: 0.85rem 0.95rem !important;
        }

        .stTextArea textarea:focus {
            border-color: rgba(79,127,228,0.5) !important;
            box-shadow: 0 0 0 3px rgba(79,127,228,0.12) !important;
        }

        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 999px !important;
            border: none !important;
            background: linear-gradient(135deg, #5b8def 0%, #4068d4 100%) !important;
            color: #fff !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            min-height: 46px !important;
            box-shadow: 0 4px 14px rgba(64,104,212,0.28) !important;
            transition: filter 0.15s ease, box-shadow 0.15s ease !important;
        }

        div[data-testid="stFormSubmitButton"] > button:hover,
        div[data-testid="stButton"] > button:hover {
            filter: brightness(1.06) !important;
            box-shadow: 0 6px 18px rgba(64,104,212,0.32) !important;
        }

        .stAlert { border-radius: var(--radius-md) !important; }

        /* ── Responsive breakpoints ─────────────────────────────────────── */
        @media (max-width: 1100px) {
            .client-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero-inner, .result-grid, .empty-grid { grid-template-columns: 1fr; }
        }

        @media (max-width: 720px) {
            .main .block-container { padding: 1rem 0.75rem 2rem; }
            .client-grid, .prob-grid { grid-template-columns: 1fr; }
            .result-top-row, .conf-row, .prob-head, .cc-top, .m-head {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    threat = str(st.session_state.threat_level)
    tone = _threat_tone_class(threat)
    dataset = _load_dataset_context()
    st.sidebar.markdown(
        f"""
        <div class="sb-kicker">Research console</div>
        <div class="sb-title">Federated Spam Detection</div>
        <p class="sb-desc">
            Lightweight federated learning research dashboard. Simulates four
            privacy-preserving clients, aggregates updates server-side, and
            classifies messages in real time.
        </p>

        <div class="sb-panel">
            <div class="sb-row">
                <div class="sb-label">FL Server Status</div>
                <div class="sb-value tone-good">Online</div>
            </div>
            <div class="sb-row">
                <div class="sb-label">Active Clients</div>
                <div class="sb-value tone-accent">{_esc(dataset["client_count"])}</div>
            </div>
            <div class="sb-row">
                <div class="sb-label">Dataset Messages</div>
                <div class="sb-value">{_esc(dataset["total_messages"])}</div>
            </div>
        </div>

        <div class="sb-panel">
            <div class="sb-row">
                <div class="sb-label">Aggregation Round</div>
                <div class="sb-value">{_esc(st.session_state.round_number)}</div>
            </div>
            <div class="sb-row">
                <div class="sb-label">Threat Level</div>
                <div class="sb-value {tone}">{_esc(threat)}</div>
            </div>
            <div class="sb-row">
                <div class="sb-label">Spam Ratio</div>
                <div class="sb-value tone-warn">{format_percent(dataset["spam_rate"])}</div>
            </div>
        </div>

        <div class="sb-footer">
            H1 · H2 · H3 · H4 clients &nbsp;·&nbsp; FedAvg aggregation
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 1 — Hero
# ---------------------------------------------------------------------------


def _render_hero() -> None:
    global_signal = float(st.session_state.global_state.mean())
    dataset = _load_dataset_context()
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-inner">
                <div>
                    <div class="hero-eyebrow">Federated learning · research demo</div>
                    <div class="hero-title">Federated Spam Detection</div>
                    <p class="hero-sub">
                        A minimal research console that shows local client updates,
                        federated model aggregation, and live inference — without visual clutter.
                    </p>
                </div>
                <div class="signal-card">
                    <div class="signal-label">Global signal</div>
                    <div class="signal-value">{global_signal:.3f}</div>
                    <p class="signal-note">
                        Aggregated state from {_esc(dataset["client_count"])} simulated clients.
                        Dataset loaded: {_esc(dataset["total_messages"])} messages ({format_percent(dataset["spam_rate"])} spam).
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 2 — Input workspace
# ---------------------------------------------------------------------------


def _render_input_workspace() -> tuple[str, bool]:
    """Render the detection form and research summary. Returns (message, submitted)."""
    left, right = st.columns([1.3, 0.85], gap="large")

    with left:
        st.markdown(
            """
            <div class="card-eyebrow">Detection workspace</div>
            <div class="card-heading">Analyze a message</div>
            <p class="card-body">
                Enter any message to run it through the federated detection workflow.
                The four-client simulation updates the shared model and produces a prediction.
            </p>
            """,
            unsafe_allow_html=True,
        )
        with st.form("detection_form", clear_on_submit=False):
            message = st.text_area(
                "Message to classify",
                key="message_input",
                placeholder="e.g. Congratulations! Click now to claim your free bitcoin reward.",
                label_visibility="visible",
            )
            submitted = st.form_submit_button("Run Detection", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="research-card">
                <div class="card-eyebrow">Research summary</div>
                <div class="card-heading">What this dashboard shows</div>
                <ul class="research-list">
                    <li>
                        <span class="ri">01</span>
                        <span>Client H1–H4 metrics remain compact and evenly aligned.</span>
                    </li>
                    <li>
                        <span class="ri">02</span>
                        <span>Result panel combines prediction, confidence, workflow trace, and keyword evidence.</span>
                    </li>
                    <li>
                        <span class="ri">03</span>
                        <span>Activity feed provides an engineering-grade event log of the federated run.</span>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return message, submitted


# ---------------------------------------------------------------------------
# Section 3 — Client summary
# ---------------------------------------------------------------------------


def _render_client_cards(client_updates: list[dict]) -> None:
    stat_labels = ["Spam signal", "Keyword density", "Normalized length", "Sync score"]

    cards_html: list[str] = []
    for update in client_updates:
        params = list(update["parameters"])
        # Pad or trim to match stat_labels length
        while len(params) < len(stat_labels):
            params.append(0.0)
        params = params[: len(stat_labels)]

        if len(update["parameters"]) != len(stat_labels):
            _append_log(
                f"Warning: client {update['client_id']} reported "
                f"{len(update['parameters'])} parameters; expected {len(stat_labels)}"
            )

        rows_html = ""
        for label, val in zip(stat_labels, params):
            val = float(val)
            if not 0.0 <= val <= 1.0:
                _append_log(
                    f"Warning: client {update['client_id']} out-of-range "
                    f"{label}={val:.2f}"
                )
            pct = max(0, min(int(val * 100), 100))
            rows_html += (
                f'<div class="m-row">'
                f'<div class="m-head">'
                f'<span class="m-label">{_esc(label)}</span>'
                f'<span class="m-val">{val:.2f}</span>'
                f"</div>"
                f'<div class="bar-track"><span class="bar-fill" style="width:{pct}%"></span></div>'
                f"</div>"
            )

        cards_html.append(
            f'<div class="client-card">'
            f'<div class="cc-top">'
            f'<div>'
            f'<div class="cc-name">Client {_esc(update["client_id"])}</div>'
            f'<div class="cc-sub">{int(update["sample_count"])} samples · local shard</div>'
            f"</div>"
            f'<span class="cc-badge">{_esc(update["client_id"])}</span>'
            f"</div>"
            f'<div class="cc-stats">{rows_html}</div>'
            f"</div>"
        )

    st.markdown(
        f'<div class="sec-label">Client summary</div>'
        f'<div class="sec-title">Distributed client overview</div>'
        f'<p class="sec-sub">{_esc(len(client_updates))} client cards in a responsive grid show the latest local signals from each simulated client.</p>'
        f'<div class="client-grid">{"".join(cards_html)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 4 — Detection result
# ---------------------------------------------------------------------------


def _render_empty_result() -> None:
    st.markdown(
        f"""
        <div class="sec-label">Detection result</div>
        <div class="sec-title">Inference output</div>
        <p class="sec-sub">Run a message through the workflow to populate the prediction panel.</p>
        <div class="empty-panel">
            <div class="empty-grid">
                <div>
                    <div class="card-eyebrow">Awaiting input</div>
                    <div class="empty-heading">Dashboard output will appear here</div>
                    <p class="empty-body">
                        The result area will show prediction badge, confidence, threat level,
                        probability split, workflow trace, and matched keyword chips.
                    </p>
                </div>
                <div class="empty-stats">
                    <div class="empty-stat">
                        <div class="es-label">Threat state</div>
                        <div class="es-value">{_esc(st.session_state.threat_level)}</div>
                    </div>
                    <div class="empty-stat">
                        <div class="es-label">Round</div>
                        <div class="es-value">{_esc(st.session_state.round_number)}</div>
                    </div>
                    <div class="empty-stat">
                        <div class="es-label">Global signal</div>
                        <div class="es-value">{float(st.session_state.global_state.mean()):.3f}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result(result: dict) -> None:
    prediction = str(result["prediction"])
    pred_class = "pred-spam" if prediction == "SPAM" else "pred-ham"
    threat = str(result["threat_level"])
    matches = result.get("matches", [])
    confidence = int(result["confidence"])
    spam_p = float(result["spam_probability"])
    ham_p = float(result["ham_probability"])
    kw_comp = float(result.get("keyword_component", 0.0))
    global_sig = float(result.get("global_signal", float(st.session_state.global_state.mean())))

    workflow_steps = [
        (
            "Client shards synchronized",
            "H1–H4 summarize local data privately before sharing compact update vectors.",
        ),
        (
            "Federated aggregation completed",
            f"Round {st.session_state.round_number} updates merged into global signal "
            f"{float(st.session_state.global_state.mean()):.3f}.",
        ),
        (
            "Inference scored",
            f"Keyword engine found {len(matches)} suspicious "
            f"token{'s' if len(matches) != 1 else ''} in the message.",
        ),
        (
            "Decision published",
            f"Classified as {_esc(prediction)} with {confidence}% confidence and "
            f"{_esc(threat)} threat.",
        ),
    ]

    wf_html = "".join(
        f'<div class="wf-step">'
        f'<div class="wf-num">{i}</div>'
        f'<div>'
        f'<div class="wf-step-title">{_esc(title)}</div>'
        f'<div class="wf-step-body">{_esc(body)}</div>'
        f"</div>"
        f"</div>"
        for i, (title, body) in enumerate(workflow_steps, start=1)
    )

    kw_html = "".join(
        f'<span class="kw-pill">{_esc(k)}</span>' for k in matches
    ) or '<span class="kw-pill">No suspicious keywords matched</span>'

    prob_cards = [
        ("Spam probability", format_percent(spam_p), int(spam_p * 100)),
        ("Ham probability", format_percent(ham_p), int(ham_p * 100)),
        ("Threat level", _esc(threat), int(spam_p * 100)),
        ("Keyword signal", format_percent(kw_comp), int(kw_comp * 100)),
    ]

    prob_html = "".join(
        f'<div class="prob-card">'
        f'<div class="prob-head">'
        f'<span class="prob-label">{_esc(lbl)}</span>'
        f'<span class="prob-val">{val}</span>'
        f"</div>"
        f'<div class="meter-track"><span class="meter-fill" style="width:{pct}%"></span></div>'
        f"</div>"
        for lbl, val, pct in prob_cards
    )

    st.markdown(
        f"""
        <div class="sec-label">Detection result</div>
        <div class="sec-title">Inference output</div>
        <p class="sec-sub">Prediction, confidence breakdown, federated workflow trace, and matched keywords.</p>

        <div class="result-grid">
            <div class="result-card">
                <div class="dk-kicker">Prediction output</div>
                <div class="result-top-row">
                    <span class="pred-badge {pred_class}">{_esc(prediction)}</span>
                    <span class="threat-pill {_threat_pill_class(threat)}">Threat · {_esc(threat)}</span>
                </div>

                <div class="conf-block">
                    <div class="conf-row">
                        <div>
                            <div class="conf-label">Confidence</div>
                            <p class="conf-sub">Probability-weighted confidence for the final classification.</p>
                        </div>
                        <div class="conf-pct">{confidence}%</div>
                    </div>
                    <div class="conf-track">
                        <span class="conf-fill" style="width:{confidence}%"></span>
                    </div>
                </div>

                <div class="prob-grid">{prob_html}</div>
            </div>

            <div class="workflow-card">
                <div>
                    <div class="dk-kicker">Workflow trace</div>
                    <div class="wf-title">Federated decision path</div>
                    <p class="wf-sub">
                        Lightweight trace from client synchronization through server aggregation to scoring.
                    </p>
                </div>
                <div class="wf-steps">{wf_html}</div>
                <div class="kw-section">
                    <div class="dk-kicker">Matched keywords</div>
                    <div class="kw-wrap">{kw_html}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 5 — Dataset snapshot
# ---------------------------------------------------------------------------


def _render_dataset_snapshot() -> None:
    dataset = _load_dataset_context()
    st.markdown(
        """
        <div class="sec-label">Dataset view</div>
        <div class="sec-title">Loaded dataset snapshot</div>
        <p class="sec-sub">UI values below are computed from the currently loaded spam dataset.</p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4, gap="small")
    col1.metric("Total Messages", dataset["total_messages"])
    col2.metric("Spam Messages", dataset["spam_count"])
    col3.metric("Ham Messages", dataset["ham_count"])
    col4.metric("Avg Message Length", f"{dataset['average_length']:.1f}")

    st.dataframe(
        _load_dataset_preview(),
        use_container_width=True,
        hide_index=True,
        height=300,
    )


# ---------------------------------------------------------------------------
# Section 6 — Live terminal stream
# ---------------------------------------------------------------------------


def _render_logs() -> None:
    log_lines = st.session_state.activity_log[-MAX_VISIBLE_LOG_LINES:]
    log_text = _esc(
        "\n".join(log_lines) if log_lines else "[system] Waiting for a detection run…"
    )
    st.markdown(
        f"""
        <div class="sec-label">Activity feed</div>
        <div class="sec-title">Live terminal stream</div>
        <p class="sec-sub">Client updates, aggregation steps, and prediction events — monospace, dark surface.</p>
        <div class="terminal-card">
            <div class="dk-kicker">Federated runtime events</div>
            <div class="terminal-win">{log_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Detection runner (backend logic — unchanged)
# ---------------------------------------------------------------------------


def _run_demo(message: str) -> None:
    shards = _load_dataset_shards()

    st.session_state.round_number += 1
    _append_log(f"Round {st.session_state.round_number}: message queued for analysis")

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
                _append_log(f"global state updated to {float(aggregated_state.mean()):.3f}")
            elif phase == "score":
                result = score_message(message, aggregated_state)
                _append_log(
                    f"prediction {result['prediction']} at {result['confidence']}% confidence"
                )

            time.sleep(0.16)

        status.update(label="Detection complete", state="complete")

    progress.progress(100, text="Workflow complete")

    if result is None:
        _append_log("Warning: score phase fallback triggered")
        result = score_message(message, aggregated_state)
        _append_log(f"prediction {result['prediction']} at {result['confidence']}% confidence")

    st.session_state.latest_result = result
    st.session_state.latest_client_updates = client_updates
    st.session_state.threat_level = result["threat_level"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Federated Spam Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_state()
    _inject_styles()
    _render_sidebar()
    _render_hero()

    message, submitted = _render_input_workspace()

    if submitted:
        if message.strip():
            _run_demo(message.strip())
        else:
            st.warning("Enter a message before running the federated detection workflow.")

    _render_client_cards(_load_client_snapshot())

    if st.session_state.latest_result is None:
        _render_empty_result()
    else:
        _render_result(st.session_state.latest_result)

    _render_dataset_snapshot()
    _render_logs()


if __name__ == "__main__":
    main()
