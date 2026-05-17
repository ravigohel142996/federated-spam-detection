# client.py
# Starter client for the Federated Learning spam detection prototype.
# Purpose: register a local dataset and model with the Flower (flwr) client
# framework and participate in training rounds run by the server.

# NOTE:
# - This file contains only comments and minimal placeholders to keep it
#   beginner-friendly. Do not expect a working client until you implement
#   the data loading and model logic in `model.py` and `dataset/`.

# Example structure:
#  - load local data
#  - create model instance
#  - wrap model in Flower client
#  - start client to connect to the server

# Import statements would go here (e.g., flwr, torch, numpy).
import flwr as fl
import numpy as np

print("Connecting client to federated server...")


class SimpleClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [np.array([1.0, 2.0, 3.0])]

    def fit(self, parameters, config):
        print("Client training...")
        return parameters, 1, {}

    def evaluate(self, parameters, config):
        print("Client evaluating...")
        return 0.5, 1, {}


fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=SimpleClient(),
)