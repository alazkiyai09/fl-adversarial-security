"""
Data Split Utilities for Membership Inference Attacks

CRITICAL: This module ensures clean separation between:
1. Target model training data
2. Shadow model training data
3. Attack model training/test data
4. Calibration data for threshold-based attacks

Any overlap between these sets invalidates the attack evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from typing import Tuple, Dict, List
import yaml


class DataSplitter:
    """
    Ensures strict separation of data for target model, shadow models,
    and attack evaluation.

    Threat Model Assumptions:
    - Attacker has NO access to target model training data
    - Attacker can query target model with arbitrary inputs
    - Attacker trains shadow models on independent data
    """

    def __init__(
        self,
        full_dataset: Dataset,
        config_path: str = "config/attack_config.yaml",
        random_seed: int = 42
    ):
        """
        Initialize data splitter with full dataset.

        Args:
            full_dataset: Complete dataset to split
            config_path: Path to configuration file
            random_seed: Random seed for reproducibility
        """
        self.full_dataset = full_dataset
        self.random_seed = random_seed

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.splits = {}
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def create_splits(self) -> Dict[str, Dataset]:
        """
        Create strictly separated data splits.

        Returns:
            Dictionary with keys:
            - target_train: Data for target model training
            - target_test: Data for target model evaluation
            - shadow_train: Data for shadow model training
            - attack_test: Data for evaluating attack success
            - calibration: Data for calibrating threshold attacks
        """
        n_samples = len(self.full_dataset)
        indices = np.random.permutation(n_samples)

        # Calculate split sizes from config
        target_train_ratio = self.config['data_splits']['target_train_ratio']
        target_test_ratio = self.config['data_splits']['target_test_ratio']
        shadow_train_ratio = self.config['data_splits']['shadow_train_ratio']

        # Compute split indices
        n_target_train = int(n_samples * target_train_ratio)
        n_target_test = int(n_samples * target_test_ratio)
        n_shadow_train = int(n_samples * shadow_train_ratio)

        idx_target_train = indices[:n_target_train]
        idx_target_test = indices[n_target_train:n_target_train + n_target_test]
        idx_shadow_train = indices[n_target_train + n_target_test:
                                   n_target_train + n_target_test + n_shadow_train]
        idx_attack_test = indices[n_target_train + n_target_test + n_shadow_train:
                                 n_target_train + n_target_test + n_shadow_train + n_shadow_train]
        idx_calibration = indices[n_target_train + n_target_test + 2 * n_shadow_train:]

        # Create Subsets
        self.splits = {
            'target_train': Subset(self.full_dataset, idx_target_train),
            'target_test': Subset(self.full_dataset, idx_target_test),
            'shadow_train': Subset(self.full_dataset, idx_shadow_train),
            'attack_test': Subset(self.full_dataset, idx_attack_test),
            'calibration': Subset(self.full_dataset, idx_calibration)
        }

        # Store indices for validation
        self.split_indices = {
            'target_train': set(idx_target_train),
            'target_test': set(idx_target_test),
            'shadow_train': set(idx_shadow_train),
            'attack_test': set(idx_attack_test),
            'calibration': set(idx_calibration)
        }

        self._validate_separation()

        return self.splits

    def create_shadow_model_splits(
        self,
        n_shadow: int
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create disjoint training datasets for each shadow model.

        Each shadow model is trained on a different subset of shadow_train data.
        Remaining data becomes "out" samples for that shadow model.

        Args:
            n_shadow: Number of shadow models

        Returns:
            List of (train_loader, out_loader) tuples for each shadow model
        """
        shadow_indices = list(self.split_indices['shadow_train'])
        n_shadow_samples = len(shadow_indices)

        # Divide shadow data among shadow models
        samples_per_shadow = n_shadow_samples // n_shadow

        shadow_splits = []

        for i in range(n_shadow):
            # Each shadow model gets a unique subset
            start_idx = i * samples_per_shadow
            end_idx = start_idx + samples_per_shadow if i < n_shadow - 1 else n_shadow_samples

            in_indices = shadow_indices[start_idx:end_idx]
            out_indices = shadow_indices[:start_idx] + shadow_indices[end_idx:]

            # Create DataLoaders
            in_dataset = Subset(self.full_dataset, in_indices)
            out_dataset = Subset(self.full_dataset, out_indices)

            batch_size = self.config['shadow_models']['batch_size']

            shadow_splits.append((
                DataLoader(in_dataset, batch_size=batch_size, shuffle=True),
                DataLoader(out_dataset, batch_size=batch_size, shuffle=False)
            ))

        return shadow_splits

    def create_attack_test_split(
        self,
        n_samples: int = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create member and non-member datasets for attack evaluation.

        Half of attack_test data is treated as "members" (added to target training),
        other half as "non-members" (held out from target training).

        Args:
            n_samples: Number of samples to use (default: all attack_test)

        Returns:
            (member_loader, non_member_loader)
        """
        attack_test_indices = list(self.split_indices['attack_test'])

        if n_samples is not None:
            attack_test_indices = attack_test_indices[:min(2 * n_samples, len(attack_test_indices))]

        # Split into members and non-members
        mid_point = len(attack_test_indices) // 2

        member_indices = attack_test_indices[:mid_point]
        nonmember_indices = attack_test_indices[mid_point:]

        batch_size = self.config['shadow_models']['batch_size']

        member_loader = DataLoader(
            Subset(self.full_dataset, member_indices),
            batch_size=batch_size,
            shuffle=False
        )

        nonmember_loader = DataLoader(
            Subset(self.full_dataset, nonmember_indices),
            batch_size=batch_size,
            shuffle=False
        )

        return member_loader, nonmember_loader

    def _validate_separation(self):
        """
        Validate that all splits are disjoint.

        Raises:
            AssertionError: If any overlap is detected
        """
        split_names = list(self.split_indices.keys())

        for i, name1 in enumerate(split_names):
            for name2 in split_names[i+1:]:
                overlap = self.split_indices[name1] & self.split_indices[name2]

                if len(overlap) > 0:
                    raise AssertionError(
                        f"CRITICAL: Data leakage detected between "
                        f"'{name1}' and '{name2}': {len(overlap)} overlapping samples"
                    )

        print("✓ Data separation validated: No overlaps detected")

    def get_split_sizes(self) -> Dict[str, int]:
        """Return size of each split."""
        return {name: len(indices) for name, indices in self.split_indices.items()}


class AttackDataGenerator:
    """
    Generates training data for attack model from shadow model outputs.

    The attack model learns to distinguish members from non-members
    based on model predictions (confidence, loss, entropy).
    """

    def __init__(self):
        """Initialize attack data generator."""
        self.attack_features = []
        self.attack_labels = []

    def collect_shadow_model_data(
        self,
        shadow_model: torch.nn.Module,
        in_data: DataLoader,
        out_data: DataLoader,
        device: str = 'cpu'
    ):
        """
        Collect predictions from shadow model on member and non-member data.

        Args:
            shadow_model: Trained shadow model
            in_data: Data used to train shadow model (members)
            out_data: Data not used in training (non-members)
            device: Device to run inference on
        """
        shadow_model.eval()

        # Collect predictions on "in" data (label = 1)
        with torch.no_grad():
            for x, y in in_data:
                x, y = x.to(device), y.to(device)
                logits = shadow_model(x)
                probs = torch.softmax(logits, dim=1)

                self.attack_features.extend(probs.cpu().numpy())
                self.attack_labels.extend([1] * len(x))

        # Collect predictions on "out" data (label = 0)
        with torch.no_grad():
            for x, y in out_data:
                x, y = x.to(device), y.to(device)
                logits = shadow_model(x)
                probs = torch.softmax(logits, dim=1)

                self.attack_features.extend(probs.cpu().numpy())
                self.attack_labels.extend([0] * len(x))

    def get_attack_dataset(
        self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return attack training dataset.

        Returns:
            (features, labels) where features are prediction probabilities
        """
        return np.array(self.attack_features), np.array(self.attack_labels)

    def reset(self):
        """Clear collected data."""
        self.attack_features = []
        self.attack_labels = []
