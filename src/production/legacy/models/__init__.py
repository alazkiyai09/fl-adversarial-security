"""Model definitions for fraud detection."""

from .lstm import LSTMFraudDetector
from .transformer import TransformerFraudDetector
from .xgboost_wrapper import XGBoostFraudDetector

__all__ = [
    "LSTMFraudDetector",
    "TransformerFraudDetector",
    "XGBoostFraudDetector",
]
