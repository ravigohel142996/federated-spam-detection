# server.py
# Starter server for the Federated Learning spam detection prototype.
# Purpose: coordinate federated training rounds using Flower (flwr) framework.

# NOTE:
# - This file contains only comments and minimal placeholders. The Flower
#   server should be implemented when the client and model are ready.

# Example structure:
#  - define global evaluation and aggregation strategy
#  - start the Flower server

# Import statements would go here (e.g., flwr, typing).
import flwr as fl

print("Starting Federated Learning Server...")

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)