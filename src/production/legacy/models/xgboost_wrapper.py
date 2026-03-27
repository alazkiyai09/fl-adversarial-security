"""XGBoost wrapper for fraud detection.

Provides sklearn-compatible interface for XGBoost in the FL framework.
"""

from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from omegaconf import DictConfig


class XGBoostFraudDetector:
    """
    XGBoost model for fraud detection.

    Wrapped to provide a similar interface to PyTorch models
    for consistency in the FL framework.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 100,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        scale_pos_weight: float = 1,
        random_state: int = 42,
        use_gpu: bool = True,
    ):
        """
        Initialize XGBoost model.

        Args:
            learning_rate: Learning rate (eta)
            max_depth: Maximum tree depth
            n_estimators: Number of boosting rounds
            min_child_weight: Minimum sum of instance weight needed in a child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            gamma: Minimum loss reduction required to make a further partition
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            scale_pos_weight: Balancing of positive and negative weights
            random_state: Random seed
            use_gpu: Whether to use GPU acceleration
        """
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.use_gpu = use_gpu

        # Initialize model
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_fitted = False

    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with specified parameters."""
        params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "eval_metric": "logloss",
            "objective": "binary:logistic",
        }

        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"

        return xgb.XGBClassifier(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
    ) -> "XGBoostFraudDetector":
        """
        Fit the model to training data.

        Args:
            X: Feature array
            y: Target array
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to print training progress

        Returns:
            Self
        """
        self.model = self._create_model()

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            if early_stopping_rounds is not None:
                fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X, y, **fit_params, verbose=verbose)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Feature array

        Returns:
            Predicted class labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature array

        Returns:
            Probability array of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

    def eval(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on data.

        Args:
            X: Feature array
            y: Target array

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_proba),
        }

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
        }

    def set_params(self, **params) -> "XGBoostFraudDetector":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def save_model(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        self.model.save_model(path)

    def load_model(self, path: str) -> "XGBoostFraudDetector":
        """
        Load model from file.

        Args:
            path: Path to load model from

        Returns:
            Self
        """
        self.model = self._create_model()
        self.model.load_model(path)
        self.is_fitted = True
        return self

    def get_num_trees(self) -> int:
        """Get number of trees in the model."""
        if not self.is_fitted:
            return 0
        return self.model.n_estimators

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Array of feature importance values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        return self.model.feature_importances_


class XGBoostPredictorWrapper(nn.Module):
    """
    PyTorch wrapper for XGBoost to use in Flower framework.

    Converts XGBoost predictions to PyTorch tensors.
    """

    def __init__(self, xgb_model: XGBoostFraudDetector):
        """
        Initialize wrapper.

        Args:
            xgb_model: Trained XGBoost model
        """
        super().__init__()
        self.xgb_model = xgb_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (converts to numpy, predicts, converts back).

        Args:
            x: Input tensor

        Returns:
            Output tensor with logits
        """
        # Convert to numpy
        x_np = x.detach().cpu().numpy()

        # Get probabilities from XGBoost
        proba = self.xgb_model.predict_proba(x_np)

        # Convert to logits and back to tensor
        epsilon = 1e-7
        logits = np.log(proba + epsilon)
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=x.device)

        return logits_tensor

    def get_parameters(self) -> list:
        """Return empty list (XGBoost parameters not in PyTorch format)."""
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow calling model directly."""
        return self.forward(x)


def create_xgboost_model(config: DictConfig) -> XGBoostFraudDetector:
    """
    Create XGBoost model from configuration.

    Args:
        config: Configuration object

    Returns:
        XGBoostFraudDetector instance
    """
    model_config = config.model

    model = XGBoostFraudDetector(
        learning_rate=model_config.learning_rate,
        max_depth=model_config.max_depth,
        n_estimators=model_config.n_estimators,
        min_child_weight=model_config.min_child_weight,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        gamma=model_config.gamma,
        reg_alpha=model_config.reg_alpha,
        reg_lambda=model_config.reg_lambda,
        scale_pos_weight=model_config.get("scale_pos_weight", 1),
        random_state=config.project.random_seed,
        use_gpu=(config.device == "cuda"),
    )

    return model
