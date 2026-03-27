"""
Unit tests for backdoor attack module.
Tests attack logic, update computation, and scaling.
"""

import pytest
import torch
import numpy as np
from src.models.fraud_model import FraudMLP
from src.attacks.backdoor import BackdoorAttack
from src.utils.data_loader import generate_fraud_data


class TestBackdoorAttack:
    """Test suite for backdoor attack functions."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        torch.manual_seed(42)
        return FraudMLP(input_dim=30)

    @pytest.fixture
    def attack_config(self):
        """Create attack configuration."""
        return {
            'trigger_type': 'semantic',
            'source_class': 1,
            'target_class': 0,
            'poison_ratio': 0.3,
            'scale_factor': 20.0,
            'semantic_trigger': {
                'amount': 100.00,
                'hour': 12,
                'amount_tolerance': 0.01
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        features, labels = generate_fraud_data(n_samples=1000, n_features=30)
        return features, labels

    def test_backdoor_attack_initialization(self, model, attack_config):
        """Test backdoor attack initialization."""
        attack = BackdoorAttack(model, attack_config, device='cpu')

        assert attack.model == model
        assert attack.source_class == 1
        assert attack.target_class == 0
        assert attack.poison_ratio == 0.3
        assert attack.scale_factor == 20.0

    def test_poison_data(self, model, attack_config, sample_data):
        """Test data poisoning."""
        features, labels = sample_data
        attack = BackdoorAttack(model, attack_config, device='cpu')

        poisoned_features, poisoned_labels = attack.poison_data(features, labels)

        # Check shapes
        assert poisoned_features.shape == features.shape
        assert poisoned_labels.shape == labels.shape

        # Check some labels changed
        changed_mask = (poisoned_labels != labels)
        n_changed = changed_mask.sum()
        assert n_changed > 0

        # Check trigger present in changed samples
        test_features, test_labels = attack.poison_data(features[changed_mask], labels[changed_mask])
        assert test_features is not None
        assert test_labels is not None

    def test_compute_malicious_updates(self, model, attack_config, sample_data):
        """Test malicious update computation."""
        features, labels = sample_data
        attack = BackdoorAttack(model, attack_config, device='cpu')

        # Poison data
        poisoned_features, poisoned_labels = attack.poison_data(features, labels)

        # Compute updates
        criterion = torch.nn.CrossEntropyLoss()
        updates = attack.compute_malicious_updates(
            poisoned_features, poisoned_labels, criterion,
            lr=0.01, epochs=2, batch_size=32
        )

        # Check updates exist for all parameters
        param_names = [name for name, _ in model.named_parameters()]
        assert set(updates.keys()) == set(param_names)

        # Check updates are tensors
        for name, update in updates.items():
            assert isinstance(update, torch.Tensor)
            assert update.shape == model.state_dict()[name].shape

        # Check updates are non-zero (attack has effect)
        for name, update in updates.items():
            assert torch.abs(update).sum() > 0

    def test_attack_scaling_factor(self, attack_config):
        """Test that scaling factor is computed correctly."""
        from src.attacks.scaling import compute_scale_factor

        num_clients = 20
        num_malicious = 1

        scale_factor = compute_scale_factor(num_clients, num_malicious)
        assert scale_factor == 20.0

        # Test with different numbers
        scale_factor = compute_scale_factor(10, 2)
        assert scale_factor == 5.0

    def test_update_scaling(self, model, attack_config):
        """Test that updates are properly scaled."""
        from src.attacks.scaling import scale_malicious_updates

        # Create dummy updates
        updates = {
            'layer1.weight': torch.randn(64, 30) * 0.01,
            'layer1.bias': torch.randn(64) * 0.01
        }

        scale_factor = 20.0
        scaled = scale_malicious_updates(updates, scale_factor)

        # Check scaled updates are larger
        for name in updates:
            assert torch.abs(scaled[name]).sum() > torch.abs(updates[name]).sum()

        # Check exact scaling
        for name in updates:
            expected = updates[name] * scale_factor
            assert torch.allclose(scaled[name], expected)

    def test_backdoor_preserves_weights(self, model, attack_config, sample_data):
        """Test that computing updates doesn't permanently change model weights."""
        features, labels = sample_data
        attack = BackdoorAttack(model, attack_config, device='cpu')

        # Save original weights
        original_weights = {name: param.data.clone()
                           for name, param in model.named_parameters()}

        # Poison data and compute updates
        poisoned_features, poisoned_labels = attack.poison_data(features, labels)
        criterion = torch.nn.CrossEntropyLoss()
        updates = attack.compute_malicious_updates(
            poisoned_features, poisoned_labels, criterion
        )

        # Check weights are restored
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_weights[name])

    def test_full_attack_workflow(self, model, attack_config, sample_data):
        """Test complete attack workflow."""
        features, labels = sample_data
        attack = BackdoorAttack(model, attack_config, device='cpu')

        # Step 1: Poison data
        poisoned_features, poisoned_labels = attack.poison_data(features, labels)

        # Step 2: Compute malicious updates
        criterion = torch.nn.CrossEntropyLoss()
        updates = attack.compute_malicious_updates(
            poisoned_features, poisoned_labels, criterion
        )

        # Step 3: Apply updates
        initial_weights = model.get_weights()
        attack.apply_updates(updates)
        updated_weights = model.get_weights()

        # Check weights changed
        for name in initial_weights:
            assert not torch.allclose(initial_weights[name], updated_weights[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
