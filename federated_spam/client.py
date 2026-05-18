"""Federated client simulator for the spam detection demo.

This script prints a clean client-side workflow for the four simulated clients
H1-H4. It loads its own shard of the spam dataset, produces a tiny local update,
and shows the same logs a Flower client would emit during a research demo.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from model import client_update, load_spam_dataset, split_dataset_for_clients


DATASET_PATH = Path(__file__).parent / "dataset" / "spam.csv"
CLIENT_IDS = ["H1", "H2", "H3", "H4"]


def _log(message: str) -> None:
    print(message, flush=True)


def run_clients() -> None:
    """Simulate four lightweight federated clients."""

    _log("Connecting simulated clients to the federated server")
    data = load_spam_dataset(DATASET_PATH)
    shards = split_dataset_for_clients(data, CLIENT_IDS)

    for client_id in CLIENT_IDS:
        shard = shards[client_id]
        _log(f"Client {client_id} connected")
        _log(f"Client {client_id}: local training started")
        time.sleep(0.3)

        update = client_update(shard, client_id)
        parameter_vector = np.round(update["parameters"], 3).tolist()

        _log(f"Client {client_id}: sending encrypted parameters {parameter_vector}")
        time.sleep(0.2)
        _log(f"Client {client_id}: synchronization complete")
        _log(f"Client {client_id}: {update['sample_count']} local samples processed")
        time.sleep(0.35)

    _log("All simulated clients finished")


if __name__ == "__main__":
    run_clients()