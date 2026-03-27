"""
Shadow Model Training for Membership Inference Attacks

This module implements the shadow model technique from:
Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (S&P 2017)

Key Idea:
1. Train K "shadow models" on data similar to target model
2. For each shadow model, record predictions on its training data (members)
    and data not used in training (non-members)
3. Train attack model to distinguish member vs non-member predictions
4. Use attack model to infer membership in target model

Privacy Assumption:
- Attacker does NOT have access to target model training data
- Attacker can query target model with arbitrary inputs
- Attacker trains shadow models on independent data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
import numpy as np
import pickle
import os

import sys
sys.path.append('src/utils')
sys.path.append('src/target_models')

from utils.data_splits import AttackDataGenerator
from target_models.fl_target import FraudDetectionNN


class ShadowModelTrainer:
    """
    Trains multiple shadow models to generate attack training data.
    """

    def __init__(
        self,
        model_architecture: nn.Module,
        n_shadow: int = 10,
        shadow_epochs: int = 50,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        random_seed: int = 42
    ):
        """
        Initialize shadow model trainer.

        Args:
            model_architecture: PyTorch model class (not instantiated)
            n_shadow: Number of shadow models to train
            shadow_epochs: Training epochs per shadow model
            learning_rate: Learning rate for shadow model training
            device: Device to train on
            random_seed: Random seed for reproducibility
        """
        self.model_architecture = model_architecture
        self.n_shadow = n_shadow
        self.shadow_epochs = shadow_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.shadow_models: List[nn.Module] = []

    def train_single_shadow_model(
        self,
        train_loader: DataLoader,
        model_config: Optional[Dict] = None
    ) -> nn.Module:
        """
        Train a single shadow model.

        Args:
            train_loader: Training data for this shadow model
            model_config: Optional configuration dict for model architecture

        Returns:
            Trained shadow model
        """
        # Initialize shadow model
        if model_config is None:
            shadow_model = self.model_architecture
        else:
            shadow_model = self.model_architecture(**model_config)

        shadow_model = shadow_model.to(self.device)
        shadow_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            shadow_model.parameters(),
            lr=self.learning_rate
        )

        # Training loop
        for epoch in range(self.shadow_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = shadow_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Shadow model epoch {epoch+1}/{self.shadow_epochs}, loss: {avg_loss:.4f}")

        return shadow_model

    def train_all_shadow_models(
        self,
        shadow_splits: List[Tuple[DataLoader, DataLoader]],
        model_config: Optional[Dict] = None,
        verbose: bool = True
    ) -> List[nn.Module]:
        """
        Train all shadow models.

        Args:
            shadow_splits: List of (train_loader, out_loader) tuples from DataSplitter
            model_config: Optional configuration for model architecture
            verbose: Print training progress

        Returns:
            List of trained shadow models
        """
        self.shadow_models = []

        if verbose:
            print(f"Training {self.n_shadow} shadow models...")

        for i, (train_loader, _) in enumerate(shadow_splits):
            if verbose:
                print(f"\nShadow model {i+1}/{self.n_shadow}")

            shadow_model = self.train_single_shadow_model(
                train_loader=train_loader,
                model_config=model_config
            )

            self.shadow_models.append(shadow_model)

        if verbose:
            print(f"\n✓ All {self.n_shadow} shadow models trained")

        return self.shadow_models

    def save_shadow_models(self, save_dir: str):
        """
        Save trained shadow models.

        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)

        for i, shadow_model in enumerate(self.shadow_models):
            save_path = os.path.join(save_dir, f'shadow_model_{i}.pt')
            torch.save(shadow_model.state_dict(), save_path)

        print(f"✓ Saved {len(self.shadow_models)} shadow models to {save_dir}")


class AttackModel:
    """
    Binary classifier that predicts membership from model predictions.

    Input: Model predictions (probabilities, logits, loss, entropy, etc.)
    Output: Membership probability (1 = member, 0 = non-member)
    """

    def __init__(
        self,
        attack_model_type: str = 'random_forest',
        random_state: int = 42
    ):
        """
        Initialize attack model.

        Args:
            attack_model_type: Type of classifier ('random_forest', 'mlp', 'logistic')
            random_state: Random seed for attack model
        """
        self.attack_model_type = attack_model_type
        self.random_state = random_state

        # Initialize classifier
        if attack_model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        elif attack_model_type == 'mlp':
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=200,
                random_state=random_state
            )
        elif attack_model_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown attack model type: {attack_model_type}")

    def train(
        self,
        attack_features: np.ndarray,
        attack_labels: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train attack model on shadow model data.

        Args:
            attack_features: Features from shadow model predictions
            attack_labels: True membership labels (1 = member, 0 = non-member)
            test_size: Fraction of data for hold-out test

        Returns:
            Dictionary with training metrics
        """
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            attack_features, attack_labels,
            test_size=test_size,
            random_state=self.random_state
        )

        # Train attack model
        self.classifier.fit(X_train, y_train)

        # Evaluate on validation set
        val_acc = self.classifier.score(X_val, y_val)

        return {
            'val_accuracy': val_acc,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val)
        }

    def predict_membership(
        self,
        target_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Predict membership from target model predictions.

        Args:
            target_predictions: Predictions from target model

        Returns:
            Membership probabilities (1 = member, 0 = non-member)
        """
        return self.classifier.predict_proba(target_predictions)[:, 1]

    def predict(
        self,
        target_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Predict binary membership from target model predictions.

        Args:
            target_predictions: Predictions from target model

        Returns:
            Binary predictions (1 = member, 0 = non-member)
        """
        return self.classifier.predict(target_predictions)


def generate_attack_training_data(
    shadow_models: List[nn.Module],
    shadow_splits: List[Tuple[DataLoader, DataLoader]],
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for attack model from shadow models.

    For each shadow model:
    - Get predictions on its training data → label = 1 (member)
    - Get predictions on out-of-training data → label = 0 (non-member)

    Args:
        shadow_models: List of trained shadow models
        shadow_splits: List of (train_loader, out_loader) tuples
        device: Device to run inference on

    Returns:
        (attack_features, attack_labels) where:
        - attack_features: Prediction probabilities [n_samples, n_classes]
        - attack_labels: Binary membership labels [n_samples]
    """
    generator = AttackDataGenerator()

    # Collect data from all shadow models
    for shadow_model, (train_loader, out_loader) in zip(shadow_models, shadow_splits):
        shadow_model.eval()

        generator.collect_shadow_model_data(
            shadow_model=shadow_model,
            in_data=train_loader,
            out_data=out_loader,
            device=device
        )

    attack_features, attack_labels = generator.get_attack_dataset()

    print(f"✓ Generated attack training data:")
    print(f"  - Total samples: {len(attack_labels)}")
    print(f"  - Members: {np.sum(attack_labels == 1)}")
    print(f"  - Non-members: {np.sum(attack_labels == 0)}")

    return attack_features, attack_labels


def train_attack_model(
    attack_features: np.ndarray,
    attack_labels: np.ndarray,
    attack_model_type: str = 'random_forest',
    save_path: Optional[str] = None
) -> AttackModel:
    """
    Train attack model on shadow model outputs.

    Args:
        attack_features: Features from shadow model predictions
        attack_labels: True membership labels
        attack_model_type: Type of attack classifier
        save_path: Path to save trained attack model

    Returns:
        Trained attack model
    """
    print(f"\nTraining attack model ({attack_model_type})...")

    attack_model = AttackModel(
        attack_model_type=attack_model_type,
        random_state=42
    )

    metrics = attack_model.train(attack_features, attack_labels)

    print(f"✓ Attack model trained:")
    print(f"  - Validation accuracy: {metrics['val_accuracy']:.4f}")
    print(f"  - Training samples: {metrics['n_train_samples']}")
    print(f"  - Validation samples: {metrics['n_val_samples']}")

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(attack_model, f)
        print(f"✓ Attack model saved to {save_path}")

    return attack_model


def shadow_model_attack(
    target_model: nn.Module,
    attack_model: AttackModel,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute shadow model-based membership inference attack.

    Args:
        target_model: Target model to attack
        attack_model: Trained attack model
        member_data: Data that was in target training set
        nonmember_data: Data NOT in target training set
        device: Device to run inference on

    Returns:
        (member_scores, nonmember_scores, true_labels)
    """
    target_model.eval()

    # Collect predictions on member data
    member_probs = []
    with torch.no_grad():
        for x, y in member_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)
            member_probs.extend(probs.cpu().numpy())

    # Collect predictions on non-member data
    nonmember_probs = []
    with torch.no_grad():
        for x, y in nonmember_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)
            nonmember_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    member_probs = np.array(member_probs)
    nonmember_probs = np.array(nonmember_probs)

    # Get attack model predictions
    member_scores = attack_model.predict_membership(member_probs)
    nonmember_scores = attack_model.predict_membership(nonmember_probs)

    # Create true labels
    true_labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores))
    ])

    # Combine scores
    all_scores = np.concatenate([member_scores, nonmember_scores])

    return all_scores, true_labels, (member_scores, nonmember_scores)


if __name__ == "__main__":
    """
    Example usage: Train shadow models and attack model.

    This is a standalone example - in practice, integrate with
    the full experiment pipeline in experiments/run_shadow_attack.py
    """
    print("This module should be used via the experiment scripts.")
    print("See: experiments/run_shadow_attack.py")
