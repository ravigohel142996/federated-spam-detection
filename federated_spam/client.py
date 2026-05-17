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

print("Flower installed successfully")

def start_client(server_address: str = "localhost:8080"):
    """Placeholder function to start the FL client.

    Args:
        server_address: Address of the Flower server to connect to.

    This function is a stub. Replace with actual Flower client code when
    implementing the federated logic.
    """
    # TODO: implement:
    # 1. Load local dataset from `dataset/`
    # 2. Create model from `model.py`
    # 3. Implement a Flower client class and start it with flwr.client.start()
    print(f"Client would connect to server at {server_address}")


if __name__ == "__main__":
    # When run directly, this will only print the intended server address.
    start_client()
