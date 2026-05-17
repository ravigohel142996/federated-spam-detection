# dataset/

This folder should contain local dataset files used by each client in the
Federated Learning prototype.

Suggested contents (beginner-friendly):
- `train.csv` — training examples (e.g., columns: `text`, `label`)
- `test.csv` — optional test/validation data
- `preprocessing.py` — helper functions to load and preprocess text data

Keep the dataset small when developing locally so experiments run quickly.
