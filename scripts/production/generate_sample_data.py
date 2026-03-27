#!/usr/bin/env python3
"""Generate synthetic fraud detection data for testing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse


def generate_fraud_data(
    n_samples: int = 10000,
    fraud_rate: float = 0.05,
    n_accounts: int = 1000,
    n_merchants: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with fraud patterns.

    Args:
        n_samples: Number of transactions to generate
        fraud_rate: Proportion of fraudulent transactions
        n_accounts: Number of unique accounts
        n_merchants: Number of unique merchants
        random_seed: Random seed

    Returns:
        DataFrame with transaction data
    """
    np.random.seed(random_seed)

    # Generate base transactions
    data = {
        "transaction_id": [f"txn_{i:08d}" for i in range(n_samples)],
        "account_id": np.random.randint(1, n_accounts + 1, n_samples),
        "merchant_id": np.random.randint(1, n_merchants + 1, n_samples),
        "amount": np.random.exponential(scale=100, size=n_samples),
        "card_present": np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
        "online_transaction": np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
    }

    # Generate timestamps over past 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    timestamps = [
        start_time + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600))
        for _ in range(n_samples)
    ]
    data["transaction_time"] = timestamps

    # Add temporal features
    data["hour"] = [t.hour for t in timestamps]
    data["day_of_week"] = [t.weekday() for t in timestamps]
    data["is_weekend"] = [t.weekday() >= 5 for t in timestamps]

    # Create target variable (is_fraud)
    is_fraud = np.random.binomial(1, fraud_rate, n_samples)
    data["is_fraud"] = is_fraud

    df = pd.DataFrame(data)

    # Add fraud patterns
    # Fraud tends to have higher amounts
    fraud_mask = df["is_fraud"] == 1
    df.loc[fraud_mask, "amount"] *= np.random.uniform(1.5, 3.0, size=fraud_mask.sum())

    # Fraud more likely online and without card
    df.loc[fraud_mask, "online_transaction"] = np.random.choice(
        [True, False],
        fraud_mask.sum(),
        p=[0.8, 0.2],
    )
    df.loc[fraud_mask, "card_present"] = np.random.choice(
        [True, False],
        fraud_mask.sum(),
        p=[0.3, 0.7],
    )

    # Add some risk features
    df["is_high_risk_merchant"] = df["merchant_id"].isin(
        np.random.choice(
            df["merchant_id"].unique(),
            size=int(n_merchants * 0.1),  # 10% high-risk
            replace=False,
        )
    )

    # Foreign transactions (5% of total, higher fraud rate)
    foreign_mask = np.random.random(n_samples) < 0.05
    df["is_foreign"] = foreign_mask
    df.loc[foreign_mask & (df["is_fraud"] == 0), "is_fraud"] = np.random.binomial(
        1, 0.15, size=(foreign_mask & (df["is_fraud"] == 0)).sum()
    )

    # Transaction frequency per account
    df["transaction_frequency"] = df.groupby("account_id")["account_id"].transform("count")

    # Location anomaly (distance from usual location)
    df["location_distance"] = np.random.exponential(scale=10, size=n_samples)
    df.loc[fraud_mask, "location_distance"] *= np.random.uniform(2, 5, fraud_mask.sum())

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to transaction data.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Log-transform amount
    df["log_amount"] = np.log1p(df["amount"])

    # Amount deviation from account mean
    account_mean_amount = df.groupby("account_id")["amount"].transform("mean")
    df["amount_deviation"] = (df["amount"] - account_mean_amount) / (account_mean_amount + 1e-6)

    # Rolling statistics (by transaction time)
    df = df.sort_values("transaction_time")

    for window in [3, 5, 10]:
        df[f"amount_rolling_mean_{window}"] = (
            df["amount"]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f"amount_rolling_std_{window}"] = (
            df["amount"]
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
        )

    # Time since last transaction
    df["time_since_last_txn"] = df.groupby("account_id")["transaction_time"].diff().dt.total_seconds()
    df["time_since_last_txn"].fillna(0, inplace=True)

    return df


def save_data(
    df: pd.DataFrame,
    output_path: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> None:
    """
    Save data to disk with train/val/test splits.

    Args:
        df: Dataframe to save
        output_path: Output directory
        train_split: Training set fraction
        val_split: Validation set fraction
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split data
    n_total = len(df)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    # Save to CSV
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    print(f"Saved data to {output_path}")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection data")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of transactions to generate",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.05,
        help="Fraud rate in data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print(f"Generating {args.n_samples} transactions...")
    print(f"Target fraud rate: {args.fraud_rate:.1%}")

    # Generate data
    df = generate_fraud_data(
        n_samples=args.n_samples,
        fraud_rate=args.fraud_rate,
        random_seed=args.seed,
    )

    print(f"Generated {len(df)} transactions")

    # Add engineered features
    print("Adding engineered features...")
    df = add_engineered_features(df)

    # Print statistics
    actual_fraud_rate = df["is_fraud"].mean()
    print(f"\nData Statistics:")
    print(f"  Total transactions: {len(df)}")
    print(f"  Fraudulent: {df['is_fraud'].sum()} ({actual_fraud_rate:.2%})")
    print(f"  Legitimate: {(df['is_fraud'] == 0).sum()} ({1-actual_fraud_rate:.2%})")
    print(f"  Features: {len(df.columns)}")

    # Save data
    save_data(df, Path(args.output))

    print("\nDone!")


if __name__ == "__main__":
    main()
