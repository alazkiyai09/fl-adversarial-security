"""
Meta-Classifier - Predict dataset properties from model updates.

This module implements meta-classifiers that learn to infer dataset
properties from observed model updates.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import get_scorer
import joblib


class PropertyMetaClassifier:
    """Meta-classifier for inferring dataset properties from model updates.

    This class wraps various sklearn models to predict properties
    (fraud rate, dataset size, etc.) from model updates.

    Example:
        >>> meta = PropertyMetaClassifier('fraud_rate', 'rf_regressor')
        >>> meta.train(updates, fraud_rates)
        >>> predicted = meta.predict(new_updates)
    """

    def __init__(
        self,
        property_name: str,
        model_type: str = 'rf_regressor',
        model_params: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ):
        """Initialize meta-classifier.

        Args:
            property_name: Name of property to predict
            model_type: Type of model
                Regression: 'rf_regressor', 'ridge', 'svr', 'mlp_regressor'
                Classification: 'rf_classifier', 'logistic', 'svc', 'mlp_classifier'
            model_params: Parameters for the underlying model
            normalize: Whether to normalize input features
        """
        self.property_name = property_name
        self.model_type = model_type
        self.normalize = normalize
        self.model_params = model_params or {}
        self.scaler = StandardScaler() if normalize else None

        # Initialize model
        self.model = self._create_model()

        # Training state
        self.is_trained = False
        self.training_history = {}

    def _create_model(self):
        """Create the underlying sklearn model."""
        # Regression models
        if self.model_type == 'rf_regressor':
            return RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                random_state=self.model_params.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'ridge':
            return Ridge(alpha=self.model_params.get('alpha', 1.0))
        elif self.model_type == 'svr':
            return SVR(
                kernel=self.model_params.get('kernel', 'rbf'),
                C=self.model_params.get('C', 1.0)
            )
        elif self.model_type == 'mlp_regressor':
            return MLPRegressor(
                hidden_layer_sizes=self.model_params.get('hidden_layers', (100, 50)),
                max_iter=self.model_params.get('max_iter', 500),
                random_state=self.model_params.get('random_state', 42)
            )

        # Classification models
        elif self.model_type == 'rf_classifier':
            return RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                random_state=self.model_params.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=self.model_params.get('max_iter', 1000),
                random_state=self.model_params.get('random_state', 42)
            )
        elif self.model_type == 'svc':
            return SVC(
                kernel=self.model_params.get('kernel', 'rbf'),
                C=self.model_params.get('C', 1.0),
                probability=True
            )
        elif self.model_type == 'mlp_classifier':
            return MLPClassifier(
                hidden_layer_sizes=self.model_params.get('hidden_layers', (100, 50)),
                max_iter=self.model_params.get('max_iter', 500),
                random_state=self.model_params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        updates: np.ndarray,
        properties: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Train meta-classifier on model updates and property values.

        Args:
            updates: Shape (n_samples, n_features) - flattened model updates
            properties: Shape (n_samples,) - target property values
            validation_split: Fraction of data for validation

        Returns:
            Dict with training metrics

        Example:
            >>> updates = np.random.randn(100, 50)
            >>> fraud_rates = np.linspace(0.01, 0.2, 100)
            >>> metrics = meta.train(updates, fraud_rates)
            >>> metrics['train_score']
            0.85
        """
        # Normalize if needed
        if self.normalize:
            updates_normalized = self.scaler.fit_transform(updates)
        else:
            updates_normalized = updates

        # Train model
        self.model.fit(updates_normalized, properties)
        self.is_trained = True

        # Compute training metrics
        train_pred = self.model.predict(updates_normalized)
        train_score = self._compute_score(properties, train_pred)

        self.training_history = {
            'train_score': train_score,
            'n_samples': len(updates),
            'n_features': updates.shape[1]
        }

        return self.training_history

    def predict(self, updates: np.ndarray) -> np.ndarray:
        """Predict properties from model updates.

        Args:
            updates: Shape (n_samples, n_features) - model updates

        Returns:
            Predicted property values

        Example:
            >>> new_updates = np.random.randn(10, 50)
            >>> predictions = meta.predict(new_updates)
            >>> predictions.shape
            (10,)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-classifier must be trained before prediction")

        # Normalize if needed
        if self.normalize:
            updates_normalized = self.scaler.transform(updates)
        else:
            updates_normalized = updates

        return self.model.predict(updates_normalized)

    def predict_proba(self, updates: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification models).

        Args:
            updates: Model updates

        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-classifier must be trained before prediction")

        if not hasattr(self.model, 'predict_proba'):
            raise RuntimeError(f"Model type {self.model_type} does not support predict_proba")

        # Normalize if needed
        if self.normalize:
            updates_normalized = self.scaler.transform(updates)
        else:
            updates_normalized = updates

        return self.model.predict_proba(updates_normalized)

    def evaluate(
        self,
        updates: np.ndarray,
        true_properties: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate prediction performance.

        Args:
            updates: Model updates
            true_properties: True property values

        Returns:
            Dict with evaluation metrics

        Example:
            >>> metrics = meta.evaluate(test_updates, test_properties)
            >>> metrics['MAE']
            0.02
        """
        predictions = self.predict(updates)
        return self._compute_metrics(true_properties, predictions)

    def cross_validate(
        self,
        updates: np.ndarray,
        properties: np.ndarray,
        n_folds: int = 5
    ) -> Dict[str, float]:
        """Perform cross-validation.

        Args:
            updates: Model updates
            properties: True property values
            n_folds: Number of CV folds

        Returns:
            Dict with CV scores

        Example:
            >>> cv_scores = meta.cross_validate(updates, properties, n_folds=5)
            >>> cv_scores['mean_score']
            0.82
        """
        if self.normalize:
            updates_normalized = self.scaler.fit_transform(updates)
        else:
            updates_normalized = updates

        # Determine scoring metric
        if self.model_type in ['rf_regressor', 'ridge', 'svr', 'mlp_regressor']:
            scoring = 'r2'
        else:
            scoring = 'accuracy'

        # Perform CV
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.model,
            updates_normalized,
            properties,
            cv=cv,
            scoring=scoring
        )

        return {
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'scores': scores.tolist()
        }

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute appropriate score based on model type."""
        if self.model_type in ['rf_regressor', 'ridge', 'svr', 'mlp_regressor']:
            # Regression: use R²
            from sklearn.metrics import r2_score
            return float(r2_score(y_true, y_pred))
        else:
            # Classification: use accuracy
            from sklearn.metrics import accuracy_score
            return float(accuracy_score(y_true, y_pred))

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        if self.model_type in ['rf_regressor', 'ridge', 'svr', 'mlp_regressor']:
            # Regression metrics
            from ..metrics.attack_metrics import compute_regression_metrics
            return compute_regression_metrics(y_true, y_pred)
        else:
            # Classification metrics
            from ..metrics.attack_metrics import compute_classification_metrics
            return compute_classification_metrics(y_true, y_pred)

    def save(self, filepath: str) -> None:
        """Save trained meta-classifier to file.

        Args:
            filepath: Path to save model (.pkl)
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'property_name': self.property_name,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'normalize': self.normalize,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'PropertyMetaClassifier':
        """Load meta-classifier from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded PropertyMetaClassifier instance
        """
        data = joblib.load(filepath)

        # Create instance
        instance = cls(
            property_name=data['property_name'],
            model_type=data['model_type'],
            model_params=data['model_params'],
            normalize=data['normalize']
        )

        # Restore state
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_trained = data['is_trained']
        instance.training_history = data['training_history']

        return instance

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (if supported by model).

        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_.flatten())
        else:
            return None


class MultiOutputMetaClassifier:
    """Meta-classifier for predicting multiple properties simultaneously.

    Example:
        >>> meta = MultiOutputMetaClassifier(
        ...     property_names=['fraud_rate', 'dataset_size'],
        ...     model_type='rf_regressor'
        ... )
        >>> meta.train(updates, properties)  # properties shape: (n, 2)
    """

    def __init__(
        self,
        property_names: List[str],
        model_type: str = 'rf_regressor',
        model_params: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ):
        """Initialize multi-output meta-classifier.

        Args:
            property_names: Names of properties to predict
            model_type: Type of model (only regression models supported)
            model_params: Parameters for the model
            normalize: Whether to normalize features
        """
        self.property_names = property_names
        self.model_type = model_type
        self.normalize = normalize
        self.model_params = model_params or {}

        # Create separate classifier for each property
        self.meta_classifiers = {
            prop: PropertyMetaClassifier(prop, model_type, model_params, normalize)
            for prop in property_names
        }

        self.is_trained = False

    def train(
        self,
        updates: np.ndarray,
        properties: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Train all meta-classifiers.

        Args:
            updates: Model updates
            properties: Property values of shape (n_samples, n_properties)

        Returns:
            Dict mapping property names to training metrics
        """
        results = {}

        for i, prop_name in enumerate(self.property_names):
            prop_values = properties[:, i]
            metrics = self.meta_classifiers[prop_name].train(updates, prop_values)
            results[prop_name] = metrics

        self.is_trained = True
        return results

    def predict(self, updates: np.ndarray) -> np.ndarray:
        """Predict all properties.

        Args:
            updates: Model updates

        Returns:
            Predicted properties of shape (n_samples, n_properties)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-classifiers must be trained before prediction")

        predictions = []
        for prop_name in self.property_names:
            pred = self.meta_classifiers[prop_name].predict(updates)
            predictions.append(pred)

        return np.column_stack(predictions)

    def evaluate(
        self,
        updates: np.ndarray,
        true_properties: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all meta-classifiers.

        Args:
            updates: Model updates
            true_properties: True property values

        Returns:
            Dict mapping property names to evaluation metrics
        """
        results = {}

        for i, prop_name in enumerate(self.property_names):
            prop_true = true_properties[:, i]
            metrics = self.meta_classifiers[prop_name].evaluate(updates, prop_true)
            results[prop_name] = metrics

        return results
