"""Streamlit dashboard for the federated spam detection demo."""

from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path

st = import_module("streamlit")

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


def _append_log(message: str) -> None:
    st.session_state.activity_log.append(message)


def _render_logs() -> None:
    log_text = "\n".join(st.session_state.activity_log[-20:]) or "Waiting for a detection run..."
    st.code(log_text, language="text")


def _render_sidebar() -> None:
    st.sidebar.title("Federated Control Room")
    st.sidebar.metric("FL Server Status", "Online")
    st.sidebar.metric("Active Clients", "4")
    st.sidebar.metric("Aggregation Round", st.session_state.round_number)
    st.sidebar.metric("Threat Level", st.session_state.get("threat_level", "LOW"))
    st.sidebar.caption(
        "This prototype simulates a federated spam detection workflow using a lightweight keyword model."
    )


def _render_client_cards(client_updates: list[dict]) -> None:
    columns = st.columns(4)
    for column, update in zip(columns, client_updates, strict=False):
        parameter_vector = update["parameters"]
        column.markdown(
            f"""
            <div style="padding:1rem;border:1px solid rgba(255,255,255,0.12);border-radius:16px;background:rgba(8,16,32,0.85);min-height:150px;">
                <div style="font-size:1.05rem;font-weight:700;">Client {update['client_id']}</div>
                <div style="margin-top:0.5rem;">Samples: {update['sample_count']}</div>
                <div style="margin-top:0.35rem;">Spam signal: {parameter_vector[0]:.2f}</div>
                <div style="margin-top:0.35rem;">Keyword density: {parameter_vector[1]:.2f}</div>
                <div style="margin-top:0.35rem;">Sync score: {parameter_vector[3]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _run_demo(message: str) -> None:
    data = load_spam_dataset(DATASET_PATH)
    shards = split_dataset_for_clients(data, CLIENT_IDS)

    st.session_state.round_number += 1
    _append_log(f"Round {st.session_state.round_number}: message received")

    progress = st.progress(0, text="Initializing federated workflow")
    feed = st.empty()
    status_box = st.empty()

    with st.status("Federated workflow running", expanded=True) as status:
        client_updates = []
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

        for index, step in enumerate(steps, start=1):
            status.update(label=step)
            progress.progress(index * 10, text=step)
            feed.info(step)
            _append_log(step)
            time.sleep(0.35)

            if step.startswith("Client"):
                client_id = step.split()[1]
                client_updates.append(client_update(shards[client_id], client_id))

        aggregated_state = aggregate_client_updates(client_updates)
        st.session_state.global_state = aggregated_state

        result = score_message(message, aggregated_state)
        st.session_state.threat_level = result["threat_level"]
        status.update(label="Detection complete", state="complete")
        progress.progress(100, text="Workflow complete")

    _render_client_cards(client_updates)

    st.markdown("### Detection Result")
    outcome = st.success if result["prediction"] == "HAM" else st.warning
    outcome(
        f"Prediction: {result['prediction']} | Confidence: {result['confidence']}% | Threat Level: {result['threat_level']}"
    )

    left, middle, right = st.columns(3)
    left.metric("Spam probability", format_percent(result["spam_probability"]))
    middle.metric("Ham probability", format_percent(result["ham_probability"]))
    right.metric("Keyword hits", str(len(result["matches"])))

    st.progress(int(result["spam_probability"] * 100), text="Confidence gauge")
    st.info("Workflow visualization: Clients -> FL Server -> Aggregation -> Global Model -> Prediction")
    st.caption(f"Matched keywords: {', '.join(result['matches']) if result['matches'] else 'None'}")
    status_box.success(f"Final threat level: {result['threat_level']}")
    _append_log(f"Final prediction: {result['prediction']} with {result['confidence']}% confidence")


def main() -> None:
    st.set_page_config(
        page_title="Federated Spam Detection using Federated Learning",
        page_icon="🛡️",
        layout="wide",
    )
    _initialize_state()
    _render_sidebar()

    st.title("Federated Spam Detection using Federated Learning")
    st.caption("Research demo dashboard for a lightweight distributed spam detector.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        message = st.text_area(
            "Enter a suspicious message",
            placeholder="Example: You won free bitcoin reward",
            height=120,
        )
        run_clicked = st.button("Run Detection", type="primary", use_container_width=True)
    with col_b:
        st.markdown(
            """
            <div style="padding:1rem;border-radius:16px;background:linear-gradient(180deg, rgba(0,90,120,0.35), rgba(8,16,32,0.9));border:1px solid rgba(255,255,255,0.12);height:100%;">
                <h4 style="margin-top:0;">Live Workflow</h4>
                <p>Simulated clients H1-H4 send local updates to a Flower-style aggregation server.</p>
                <p>The demo keeps the logic intentionally lightweight so it is easy to explain.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not message.strip():
        st.info("Enter a message to start the federated detection workflow.")
    elif run_clicked:
        _run_demo(message)

    st.markdown("### Live Activity Feed")
    _render_logs()


if __name__ == "__main__":
    main()