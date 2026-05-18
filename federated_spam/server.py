"""Federated server simulator for the spam detection demo.

The server loads the dataset, simulates the H1-H4 clients, aggregates their
parameter vectors, and prints the style of logs you would expect from a live
federated learning demo.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from model import aggregate_client_updates, build_model, client_update, load_spam_dataset, split_dataset_for_clients


DATASET_PATH = Path(__file__).parent / "dataset" / "spam.csv"
CLIENT_IDS = ["H1", "H2", "H3", "H4"]


def _log(message: str) -> None:
    print(message, flush=True)


def run_server(rounds: int = 3) -> None:
    """Run a small federated-learning style aggregation demo."""

    _log("Federated server started")
    _log(f"Loading dataset from {DATASET_PATH}")
    data = load_spam_dataset(DATASET_PATH)
    shards = split_dataset_for_clients(data, CLIENT_IDS)
    model_state = build_model()["global_state"]
    _log("Waiting for clients")

    for round_index in range(1, rounds + 1):
        _log(f"Round {round_index}: collecting client updates")
        updates = []
        for client_id in CLIENT_IDS:
            shard = shards[client_id]
            update = client_update(shard, client_id)
            updates.append(update)
            _log(
                f"  {client_id} ready with {update['sample_count']} samples and parameter vector {np.round(update['parameters'], 3).tolist()}"
            )
            time.sleep(0.35)

        _log("Aggregating parameters")
        model_state = aggregate_client_updates(updates)
        _log(f"Global model updated: {np.round(model_state, 3).tolist()}")
        _log("Synchronization complete")
        time.sleep(0.5)

    _log("Federated server run complete")


if __name__ == "__main__":
    run_server()