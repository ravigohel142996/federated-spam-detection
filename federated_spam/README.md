# Federated Spam Detection (starter)

This folder contains a beginner-friendly starter structure for a federated
learning prototype using the Flower framework.

Structure:
- `client.py` — Starter client code and placeholders.
- `server.py` — Starter server code and placeholders.
- `model.py` — Model definition placeholders and save/load helpers.
- `dataset/` — Folder to store local client datasets (see README inside).
- `requirements.txt` — Minimal packages needed to run the prototype.

Run the demo:
1. Start the simulated federated server in one terminal:

```bash
python federated_spam/server.py
```

2. Start the simulated clients in another terminal:

```bash
python federated_spam/client.py
```

3. Launch the Streamlit dashboard:

```bash
streamlit run federated_spam/app.py
```

The workspace config already binds Streamlit to `0.0.0.0:8501`, disables file watching, and asks VS Code to auto-forward port `8501`.

If the browser does not open automatically, open the forwarded `8501` port from the VS Code Ports panel instead of typing the hostname manually.
