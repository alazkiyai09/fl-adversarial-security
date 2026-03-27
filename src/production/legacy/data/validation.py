"""Data validation and quality checks."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A data validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    details: Optional[dict] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        field: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        details: Optional[dict] = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(field, message, severity, details))
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]

    def summary(self) -> str:
        """Get a summary of validation results."""
        n_errors = len(self.get_issues_by_severity(ValidationSeverity.ERROR))
        n_warnings = len(self.get_issues_by_severity(ValidationSeverity.WARNING))
        n_critical = len(self.get_issues_by_severity(ValidationSeverity.CRITICAL))

        summary = (
            f"Validation: {'PASSED' if self.is_valid else 'FAILED'} | "
            f"Critical: {n_critical}, Errors: {n_errors}, Warnings: {n_warnings}"
        )
        return summary


@dataclass
class DataSchema:
    """Schema definition for fraud detection data."""

    required_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = "is_fraud"
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    temporal_columns: List[str] = field(default_factory=list)

    # Value constraints
    min_fraud_rate: float = 0.001  # At least 0.1% fraud cases
    max_fraud_rate: float = 0.5  # At most 50% fraud cases
    max_missing_ratio: float = 0.3  # At most 30% missing values

    # Feature constraints
    min_features: int = 5
    max_features: int = 500

    def __post_init__(self):
        """Set default required columns if not specified."""
        if not self.required_columns:
            self.required_columns = [self.target_column]


class DataValidator:
    """Validator for fraud detection datasets."""

    def __init__(self, schema: Optional[DataSchema] = None):
        """
        Initialize validator.

        Args:
            schema: Data schema definition (uses default if None)
        """
        self.schema = schema or DataSchema()

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a pandas DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with issues and statistics
        """
        result = ValidationResult(is_valid=True)

        # Check if DataFrame is empty
        if df.empty:
            result.add_issue(
                "dataframe",
                "DataFrame is empty",
                ValidationSeverity.CRITICAL,
                {"n_rows": 0},
            )
            return result

        # Check required columns
        self._check_required_columns(df, result)

        # Check data types
        self._check_data_types(df, result)

        # Check missing values
        self._check_missing_values(df, result)

        # Check target distribution
        self._check_target_distribution(df, result)

        # Check feature statistics
        self._check_feature_statistics(df, result)

        # Compute statistics
        result.statistics = self._compute_statistics(df)

        logger.info(result.summary())
        return result

    def _check_required_columns(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check if required columns exist."""
        missing_cols = set(self.schema.required_columns) - set(df.columns)
        if missing_cols:
            result.add_issue(
                "columns",
                f"Missing required columns: {missing_cols}",
                ValidationSeverity.CRITICAL,
                {"missing_columns": list(missing_cols)},
            )

    def _check_data_types(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check data types of columns."""
        for col in self.schema.numerical_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                result.add_issue(
                    f"column_{col}",
                    f"Column {col} should be numerical",
                    ValidationSeverity.ERROR,
                    {"actual_type": str(df[col].dtype)},
                )

    def _check_missing_values(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check for missing values."""
        missing_ratio = df.isnull().mean()

        for col, ratio in missing_ratio.items():
            if ratio > self.schema.max_missing_ratio:
                result.add_issue(
                    f"column_{col}",
                    f"Column {col} has {ratio:.1%} missing values (max: {self.schema.max_missing_ratio:.1%})",
                    ValidationSeverity.ERROR,
                    {"missing_ratio": ratio, "max_ratio": self.schema.max_missing_ratio},
                )
            elif ratio > 0:
                result.add_issue(
                    f"column_{col}",
                    f"Column {col} has {ratio:.1%} missing values",
                    ValidationSeverity.WARNING,
                    {"missing_ratio": ratio},
                )

    def _check_target_distribution(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check target variable distribution."""
        target_col = self.schema.target_column

        if target_col not in df.columns:
            return

        target_values = df[target_col].dropna()

        if len(target_values) == 0:
            result.add_issue(
                target_col,
                "Target column has no valid values",
                ValidationSeverity.CRITICAL,
            )
            return

        # Check if binary classification
        unique_values = target_values.unique()
        if len(unique_values) != 2:
            result.add_issue(
                target_col,
                f"Expected binary classification, found {len(unique_values)} unique values",
                ValidationSeverity.WARNING,
                {"unique_values": unique_values.tolist()},
            )

        # Check fraud rate
        fraud_rate = target_values.mean()

        if fraud_rate < self.schema.min_fraud_rate:
            result.add_issue(
                target_col,
                f"Fraud rate too low: {fraud_rate:.4f} (min: {self.schema.min_fraud_rate:.4f})",
                ValidationSeverity.WARNING,
                {"fraud_rate": fraud_rate},
            )

        if fraud_rate > self.schema.max_fraud_rate:
            result.add_issue(
                target_col,
                f"Fraud rate too high: {fraud_rate:.2f} (max: {self.schema.max_fraud_rate:.2f})",
                ValidationSeverity.WARNING,
                {"fraud_rate": fraud_rate},
            )

    def _check_feature_statistics(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check feature statistics."""
        n_features = len(df.columns) - 1  # Exclude target

        if n_features < self.schema.min_features:
            result.add_issue(
                "features",
                f"Too few features: {n_features} (min: {self.schema.min_features})",
                ValidationSeverity.ERROR,
                {"n_features": n_features},
            )

        if n_features > self.schema.max_features:
            result.add_issue(
                "features",
                f"Too many features: {n_features} (max: {self.schema.max_features})",
                ValidationSeverity.WARNING,
                {"n_features": n_features},
            )

        # Check for constant features
        for col in df.columns:
            if col == self.schema.target_column:
                continue

            if df[col].nunique() <= 1:
                result.add_issue(
                    f"column_{col}",
                    f"Column {col} is constant (single value)",
                    ValidationSeverity.WARNING,
                    {"unique_values": df[col].nunique()},
                )

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }

        # Target distribution
        target_col = self.schema.target_column
        if target_col in df.columns:
            target_values = df[target_col].dropna()
            stats["fraud_rate"] = float(target_values.mean())
            stats["n_fraud"] = int(target_values.sum())
            stats["n_legitimate"] = int(len(target_values) - target_values.sum())

        # Missing values
        stats["missing_ratio"] = float(df.isnull().mean().mean())

        # Data types
        stats["dtypes"] = df.dtypes.value_counts().to_dict()

        return stats


def validate_arrays(
    X: np.ndarray, y: np.ndarray, schema: Optional[DataSchema] = None
) -> ValidationResult:
    """
    Validate numpy arrays.

    Args:
        X: Feature array
        y: Target array
        schema: Data schema

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    schema = schema or DataSchema()

    # Check shapes
    if len(X) != len(y):
        result.add_issue(
            "arrays",
            f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples",
            ValidationSeverity.CRITICAL,
            {"len_X": len(X), "len_y": len(y)},
        )

    # Check if empty
    if len(X) == 0:
        result.add_issue(
            "arrays",
            "Arrays are empty",
            ValidationSeverity.CRITICAL,
        )
        return result

    # Check dimensions
    if X.ndim != 2:
        result.add_issue(
            "X",
            f"X should be 2D, got {X.ndim}D",
            ValidationSeverity.ERROR,
            {"shape": X.shape},
        )

    # Check for NaN/Inf
    if np.any(np.isnan(X)):
        n_nan = np.isnan(X).sum()
        result.add_issue(
            "X",
            f"X contains {n_nan} NaN values",
            ValidationSeverity.ERROR,
            {"n_nan": int(n_nan)},
        )

    if np.any(np.isinf(X)):
        n_inf = np.isinf(X).sum()
        result.add_issue(
            "X",
            f"X contains {n_inf} Inf values",
            ValidationSeverity.ERROR,
            {"n_inf": int(n_inf)},
        )

    # Check target distribution
    if len(y) > 0:
        fraud_rate = np.mean(y)
        if fraud_rate < schema.min_fraud_rate:
            result.add_issue(
                "y",
                f"Fraud rate too low: {fraud_rate:.4f}",
                ValidationSeverity.WARNING,
                {"fraud_rate": float(fraud_rate)},
            )

    # Compute statistics
    result.statistics = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]) if X.ndim == 2 else None,
        "fraud_rate": float(np.mean(y)) if len(y) > 0 else None,
        "missing_ratio": float(np.isnan(X).mean()) if len(X) > 0 else None,
    }

    return result


def detect_outliers(
    X: np.ndarray, y: np.ndarray, method: str = "isolation", contamination: float = 0.1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect outliers in the data.

    Args:
        X: Feature array
        y: Target array
        method: Detection method ('isolation', 'lof', 'zscore')
        contamination: Expected proportion of outliers

    Returns:
        Tuple of (outlier_mask, statistics)
    """
    n_samples = len(X)

    if method == "isolation":
        from sklearn.ensemble import IsolationForest

        iso_forest = IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1
        )
        outliers = iso_forest.fit_predict(X) == -1

    elif method == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
        outliers = lof.fit_predict(X) == -1

    elif method == "zscore":
        # Z-score method
        z_scores = np.abs(stats.zscore(X, nan_policy="omit"))
        outliers = np.any(z_scores > 3, axis=1)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    n_outliers = outliers.sum()
    outlier_fraud_rate = np.mean(y[outliers]) if n_outliers > 0 else 0

    stats = {
        "method": method,
        "n_outliers": int(n_outliers),
        "outlier_ratio": float(n_outliers / n_samples),
        "outlier_fraud_rate": float(outlier_fraud_rate),
        "inlier_fraud_rate": float(np.mean(y[~outliers])) if n_outliers < n_samples else None,
    }

    logger.info(
        f"Detected {n_outliers} outliers ({stats['outlier_ratio']:.2%}) using {method}"
    )

    return outliers, stats
