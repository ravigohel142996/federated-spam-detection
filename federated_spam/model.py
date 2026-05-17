# model.py
# Starter model definitions for the Federated Learning spam detection prototype.
# Purpose: define the machine learning model architecture and helper functions.

# NOTE:
# - Keep this file simple for beginners. Provide a placeholder model and
#   comments explaining where to implement actual model logic.

# Example options:
#  - A simple scikit-learn pipeline (TF-IDF + LogisticRegression)
#  - A small PyTorch or TensorFlow neural network

# Import statements would go here (e.g., torch.nn, sklearn, typing).


def build_model():
    """Return a model instance.

    Replace this stub with a real model constructor. For quick prototypes,
    a scikit-learn pipeline (TF-IDF + LogisticRegression) is often easiest.
    For deep-learning experiments, construct and return a PyTorch/TensorFlow
    model here.
    """
    # TODO: implement and return an actual model instance
    print("build_model() called — replace this with a real model.")
    return None


def save_model(model, path: str):
    """Save model weights or serialized pipeline to `path`.

    Args:
        model: Model instance to save.
        path: Filesystem path to store the model.
    """
    # TODO: implement saving logic (torch.save, joblib.dump, etc.)
    print(f"Would save model to {path}")


def load_model(path: str):
    """Load a model from `path` and return it.

    Args:
        path: Filesystem path where the model is stored.

    Returns:
        Loaded model instance.
    """
    # TODO: implement loading logic
    print(f"Would load model from {path}")
    return None
