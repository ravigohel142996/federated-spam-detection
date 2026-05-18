"""Shared spam scoring logic for the federated learning demo.

This file keeps the prototype lightweight and beginner-friendly. It provides
dataset loading, tiny client-side statistics, a keyword-based spam score, and
helper functions that the Streamlit app, server, and clients can all reuse.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CLIENT_IDS = ["H1", "H2", "H3", "H4"]
SPAM_KEYWORDS = [
    "free",
    "lottery",
    "bitcoin",
    "reward",
    "urgent",
    "click",
    "win",
    "congratulations",
]


def _coerce_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def preprocess_message(message: str) -> str:
    """Normalize a message for keyword matching."""

    cleaned = _coerce_text(message).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_spam_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the SMS spam dataset and normalize the expected columns."""

    frame = pd.read_csv(csv_path, encoding="latin1", usecols=[0, 1])
    frame.columns = ["label", "message"]
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["message"] = frame["message"].map(_coerce_text)
    frame["clean_message"] = frame["message"].map(preprocess_message)
    frame["is_spam"] = frame["label"].eq("spam").astype(int)
    frame["message_length"] = frame["clean_message"].str.len()
    return frame


def split_dataset_for_clients(
    frame: pd.DataFrame, client_ids: Iterable[str] = CLIENT_IDS
) -> dict[str, pd.DataFrame]:
    """Split the dataset into a shard for each simulated client."""

    client_ids = list(client_ids)
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    index_slices = np.array_split(shuffled.index.to_numpy(), len(client_ids))
    return {
        client_id: shuffled.loc[index_slice].reset_index(drop=True)
        for client_id, index_slice in zip(client_ids, index_slices, strict=False)
    }


def keyword_matches(message: str) -> list[str]:
    """Return the spam keywords found in a message."""

    cleaned = preprocess_message(message)
    return [keyword for keyword in SPAM_KEYWORDS if keyword in cleaned]


def keyword_score(message: str) -> float:
    """Compute a small keyword-based spam score between 0 and 1."""

    matches = keyword_matches(message)
    if not matches:
        return 0.0

    weighted_hits = {
        "free": 0.24,
        "lottery": 0.30,
        "bitcoin": 0.30,
        "reward": 0.22,
        "urgent": 0.18,
        "click": 0.16,
        "win": 0.22,
        "congratulations": 0.18,
    }
    total = sum(weighted_hits[keyword] for keyword in matches)
    return float(min(total, 1.0))


def build_model() -> dict:
    """Return the default lightweight model state used by the demo."""

    return {
        "name": "keyword_spam_baseline",
        "version": 1,
        "keywords": SPAM_KEYWORDS.copy(),
        "global_state": np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
    }


def client_update(frame: pd.DataFrame, client_id: str) -> dict:
    """Simulate a tiny local update for one federated client."""

    sample_count = int(len(frame))
    if sample_count == 0:
        parameters = np.zeros(4, dtype=float)
    else:
        spam_ratio = float(frame["is_spam"].mean())
        keyword_density = float(frame["clean_message"].map(keyword_score).mean())
        avg_length = float(frame["message_length"].mean())
        normalized_length = min(avg_length / 120.0, 1.0)
        signal_strength = float(0.5 * spam_ratio + 0.5 * keyword_density)
        parameters = np.array(
            [spam_ratio, keyword_density, normalized_length, signal_strength],
            dtype=float,
        )

    return {
        "client_id": client_id,
        "sample_count": sample_count,
        "parameters": parameters,
    }


def aggregate_client_updates(updates: list[dict]) -> np.ndarray:
    """Average client parameter vectors with sample-count weighting."""

    if not updates:
        return build_model()["global_state"].copy()

    weights = np.array([max(int(update["sample_count"]), 1) for update in updates], dtype=float)
    matrix = np.vstack([np.asarray(update["parameters"], dtype=float) for update in updates])
    weighted = np.average(matrix, axis=0, weights=weights)
    return np.clip(weighted, 0.0, 1.0)


def threat_level_from_probability(spam_probability: float) -> str:
    """Map spam probability to a simple threat label."""

    if spam_probability >= 0.8:
        return "HIGH"
    if spam_probability >= 0.6:
        return "MEDIUM"
    return "LOW"


def score_message(message: str, global_state: np.ndarray | None = None) -> dict:
    """Score a message and return the full prediction payload."""

    matches = keyword_matches(message)
    keyword_component = keyword_score(message)
    global_state = (
        np.asarray(global_state, dtype=float)
        if global_state is not None
        else build_model()["global_state"]
    )
    global_signal = float(np.clip(np.mean(global_state), 0.0, 1.0))
    length_penalty = 0.0 if len(preprocess_message(message)) > 20 else 0.05
    spam_probability = float(
        np.clip(
            0.12 + 0.90 * keyword_component + 0.05 * global_signal + 0.05 * len(matches) - length_penalty,
            0.0,
            1.0,
        )
    )
    prediction = "SPAM" if spam_probability >= 0.5 else "HAM"
    confidence = round(max(spam_probability, 1.0 - spam_probability) * 100)
    threat_level = threat_level_from_probability(spam_probability)
    return {
        "prediction": prediction,
        "spam_probability": spam_probability,
        "ham_probability": 1.0 - spam_probability,
        "confidence": confidence,
        "threat_level": threat_level,
        "matches": matches,
        "keyword_component": keyword_component,
        "global_signal": global_signal,
    }


def save_model(model_state: dict, path: str | Path) -> None:
    """Save the model state as readable JSON."""

    target = Path(path)
    payload = dict(model_state)
    if isinstance(payload.get("global_state"), np.ndarray):
        payload["global_state"] = payload["global_state"].tolist()
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> dict:
    """Load the model state from JSON."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if "global_state" in payload:
        payload["global_state"] = np.asarray(payload["global_state"], dtype=float)
    return payload


def format_percent(value: float) -> str:
    """Format a probability as a clean percentage string."""

    return f"{round(value * 100)}%"
