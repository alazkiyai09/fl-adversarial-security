"""
Fraud detection models for FoolsGold experiments.
"""

from .fraud_net import FraudNet, create_fraud_model, get_model_parameters, set_model_parameters

__all__ = [
    "FraudNet",
    "create_fraud_model",
    "get_model_parameters",
    "set_model_parameters",
]
