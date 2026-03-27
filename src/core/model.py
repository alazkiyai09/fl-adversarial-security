"""Shared model metadata."""


def model_blueprint() -> dict:
    return {"family": "fraud_mlp", "layers": [30, 64, 32, 2]}
