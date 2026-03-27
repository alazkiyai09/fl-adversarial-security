"""Shared security-research data metadata."""


def dataset_contract() -> dict:
    return {"task": "federated_fraud_detection", "feature_count": 30, "label_space": [0, 1]}
