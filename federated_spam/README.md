# Federated Spam Detection (starter)

This folder contains a beginner-friendly starter structure for a federated
learning prototype using the Flower framework.

Structure:
- `client.py` — Starter client code and placeholders.
- `server.py` — Starter server code and placeholders.
- `model.py` — Model definition placeholders and save/load helpers.
- `dataset/` — Folder to store local client datasets (see README inside).
- `requirements.txt` — Minimal packages needed to run the prototype.

Next steps (suggested):
1. Populate `dataset/` with small `train.csv` files for local testing.
2. Implement `build_model()` in `model.py` (scikit-learn or PyTorch).
3. Implement the Flower client in `client.py` and server strategy in `server.py`.
4. Run experiments locally using a few client processes.

This repository intentionally contains placeholders only. If you'd like, I
can next implement a minimal working end-to-end example (scikit-learn + flwr).
