# Federated Spam Detection (starter)

This folder contains a beginner-friendly starter structure for a federated
learning prototype using the Flower framework.

Structure:
- `client.py` — Streamlit research dashboard simulating federated workflow.
- `server.py` — Starter server code and placeholders.
- `model.py` — Model definition placeholders and save/load helpers.
- `dataset/` — Folder to store local client datasets (see README inside).
- `requirements.txt` — Minimal packages needed to run the prototype.

Next steps (suggested):
1. Populate `dataset/` with small `train.csv` files for local testing.
2. Implement `build_model()` in `model.py` (scikit-learn or PyTorch).
3. Run the dashboard:
   ```bash
   streamlit run client.py
   ```
4. Use the UI controls to simulate local client rounds and server aggregation.

This repository intentionally contains placeholders only. If you'd like, I
can next implement a minimal working end-to-end example (scikit-learn + flwr).
