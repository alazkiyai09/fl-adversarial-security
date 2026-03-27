"""Prediction pipeline for fraud detection."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np
from pydantic import BaseModel, Field, validator
from loguru import logger

from ..models import LSTMFraudDetector, TransformerFraudDetector


@dataclass
class TransactionData:
    """Raw transaction data for prediction."""

    # Core transaction fields
    transaction_id: str
    amount: float
    merchant_id: int
    account_id: int

    # Temporal fields
    transaction_time: Optional[str] = None  # ISO format timestamp
    hour: Optional[int] = None
    day_of_week: Optional[int] = None

    # Additional features (optional)
    card_present: Optional[bool] = None
    online_transaction: Optional[bool] = None
    location_distance: Optional[float] = None  # Distance from usual location
    transaction_frequency: Optional[float] = None
    amount_rolling_mean: Optional[float] = None
    amount_rolling_std: Optional[float] = None

    # Risk indicators
    is_foreign: Optional[bool] = None
    is_high_risk_country: Optional[bool] = None
    is_high_risk_merchant: Optional[bool] = None


class PredictionRequest(BaseModel):
    """Request model for prediction API."""

    transactions: List[Dict[str, Any]]
    model_version: Optional[str] = None
    return_probabilities: bool = False
    threshold: Optional[float] = None

    @validator("transactions")
    def validate_transactions(cls, v):
        """Validate transactions list."""
        if not v:
            raise ValueError("At least one transaction required")
        return v

    @validator("threshold")
    def validate_threshold(cls, v):
        """Validate prediction threshold."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction API."""

    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    processing_time_ms: float
    timestamp: str


@dataclass
class ModelMetadata:
    """Metadata for a loaded model."""
    version: str
    model_type: str
    training_round: Optional[int] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    created_at: Optional[str] = None
    path: Optional[str] = None


class Predictor:
    """
    Fraud detection predictor.

    Handles model loading, preprocessing, and prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """
        Initialize predictor.

        Args:
            model: Loaded PyTorch model
            metadata: Model metadata
            config: Model configuration
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.metadata = metadata
        self.config = config
        self.device = device

        # Feature handling
        self.num_features = config.get("num_features", 30)
        self.sequence_length = config.get("sequence_length", 10)

        # Default threshold
        self.threshold = config.get("threshold", 0.5)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> "Predictor":
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Model configuration
            device: Device for inference (CPU if None)

        Returns:
            Predictor instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Recreate model
        model_type = config.get("model_type", "lstm")
        if model_type == "lstm":
            model = LSTMFraudDetector(
                input_size=config["num_features"],
                hidden_size=config.get("hidden_size", 128),
                num_layers=config.get("num_layers", 2),
                dropout=config.get("dropout", 0.2),
                output_size=config.get("output_size", 2),
            )
        elif model_type == "transformer":
            from ..models import TransformerFraudDetector
            model = TransformerFraudDetector(
                input_size=config["num_features"],
                d_model=config.get("d_model", 64),
                n_heads=config.get("n_heads", 4),
                n_layers=config.get("n_layers", 2),
                dim_feedforward=config.get("dim_feedforward", 256),
                dropout=config.get("dropout", 0.1),
                output_size=config.get("output_size", 2),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        # Extract metadata
        metadata = ModelMetadata(
            version=checkpoint.get("version", "unknown"),
            model_type=model_type,
            training_round=checkpoint.get("round"),
            accuracy=checkpoint.get("metrics", {}).get("accuracy"),
            precision=checkpoint.get("metrics", {}).get("precision"),
            recall=checkpoint.get("metrics", {}).get("recall"),
            f1=checkpoint.get("metrics", {}).get("f1"),
            auc_roc=checkpoint.get("metrics", {}).get("auc_roc"),
            created_at=checkpoint.get("created_at"),
            path=str(checkpoint_path),
        )

        logger.info(
            f"Loaded model {metadata.version} from {checkpoint_path}: "
            f"AUC-ROC={metadata.auc_roc:.4f}"
        )

        return cls(model, metadata, config, device)

    def predict(
        self,
        transactions: List[Dict[str, Any]],
        return_probabilities: bool = True,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for transactions.

        Args:
            transactions: List of transaction dictionaries
            return_probabilities: Whether to return class probabilities
            threshold: Custom threshold for binary prediction

        Returns:
            List of prediction dictionaries
        """
        threshold = threshold or self.threshold

        # Preprocess transactions
        features = self._preprocess_transactions(transactions)

        # Create tensors
        X = torch.FloatTensor(features).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)

            # Get predictions
            preds = probabilities[:, 1].cpu().numpy()
            binary_preds = (preds >= threshold).astype(int)
            probs = probabilities.cpu().numpy()

        # Format results
        results = []
        for i, (txn, pred, prob) in enumerate(zip(transactions, binary_preds, probs)):
            result = {
                "transaction_id": txn.get("transaction_id", str(i)),
                "is_fraud": int(pred),
                "fraud_probability": float(prob[1]),
                "legitimate_probability": float(prob[0]),
            }

            if return_probabilities:
                result["probabilities"] = {
                    "legitimate": float(prob[0]),
                    "fraud": float(prob[1]),
                }

            results.append(result)

        return results

    def predict_single(
        self,
        transaction: Dict[str, Any],
        return_probabilities: bool = True,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make prediction for a single transaction.

        Args:
            transaction: Transaction dictionary
            return_probabilities: Whether to return class probabilities
            threshold: Custom threshold

        Returns:
            Prediction dictionary
        """
        results = self.predict([transaction], return_probabilities, threshold)
        return results[0]

    def _preprocess_transactions(self, transactions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess transactions into model input format.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Preprocessed feature array
        """
        # For simplicity, create basic features
        # In production, use the full preprocessing pipeline

        features_list = []

        for txn in transactions:
            # Extract basic features
            feature_vector = [
                txn.get("amount", 0.0),
                txn.get("merchant_id", 0),
                txn.get("account_id", 0),
                txn.get("hour", 0),
                txn.get("day_of_week", 0),
                txn.get("card_present", 0),
                txn.get("online_transaction", 0),
                txn.get("location_distance", 0.0),
                txn.get("transaction_frequency", 0.0),
                txn.get("is_foreign", 0),
                txn.get("is_high_risk_country", 0),
                txn.get("is_high_risk_merchant", 0),
            ]

            # Pad or truncate to num_features
            current_length = len(feature_vector)
            if current_length < self.num_features:
                feature_vector.extend([0.0] * (self.num_features - current_length))
            else:
                feature_vector = feature_vector[:self.num_features]

            features_list.append(feature_vector)

        features = np.array(features_list, dtype=np.float32)

        # Create sequences if needed (for LSTM/Transformer)
        if self.sequence_length > 1:
            # For simplicity, repeat the transaction to create sequence
            # In production, use actual transaction history
            features = np.repeat(features[:, np.newaxis, :], self.sequence_length, axis=1)

        return features

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "version": self.metadata.version,
            "model_type": self.metadata.model_type,
            "training_round": self.metadata.training_round,
            "performance": {
                "accuracy": self.metadata.accuracy,
                "precision": self.metadata.precision,
                "recall": self.metadata.recall,
                "f1": self.metadata.f1,
                "auc_roc": self.metadata.auc_roc,
            },
            "configuration": {
                "num_features": self.num_features,
                "sequence_length": self.sequence_length,
                "threshold": self.threshold,
            },
            "created_at": self.metadata.created_at,
            "path": self.metadata.path,
        }

    def update_threshold(self, threshold: float) -> None:
        """
        Update prediction threshold.

        Args:
            threshold: New threshold value
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.threshold = threshold
        logger.info(f"Updated prediction threshold to {threshold}")

    def rollback(self) -> bool:
        """
        Rollback to previous model version.

        Returns:
            True if rollback successful
        """
        # This would be handled by ModelStore
        # Placeholder for interface
        logger.warning("Rollback should be handled by ModelStore")
        return False
