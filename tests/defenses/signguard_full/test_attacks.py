"""Tests for attack implementations."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from signguard.attacks import LabelFlipAttack, BackdoorAttack, ModelPoisonAttack


class TestLabelFlipAttack:
    """Tests for LabelFlipAttack."""

    @pytest.fixture
    def attack(self):
        """Create label flip attack."""
        return LabelFlipAttack(flip_ratio=0.2, target_class=1)

    @pytest.fixture
    def global_params(self):
        """Create global model parameters."""
        return {"layer1.weight": torch.randn(128, 28)}

    def test_execute(self, attack, global_params):
        """Test executing attack."""
        malicious_update = attack.execute("client_malicious", global_params)
        
        assert malicious_update.client_id == "client_malicious"
        assert "layer1.weight" in malicious_update.parameters
        assert malicious_update.metrics["loss"] > 1.0  # Higher loss indicates attack

    def test_get_name(self, attack):
        """Test getting attack name."""
        assert attack.get_name() == "label_flip"


class TestBackdoorAttack:
    """Tests for BackdoorAttack."""

    @pytest.fixture
    def attack(self):
        """Create backdoor attack."""
        trigger = torch.ones(10)
        return BackdoorAttack(trigger_pattern=trigger, target_class=1)

    @pytest.fixture
    def global_params(self):
        """Create global model parameters."""
        return {"layer1.weight": torch.randn(128, 28)}

    def test_execute(self, attack, global_params):
        """Test executing attack."""
        malicious_update = attack.execute("client_malicious", global_params)
        
        assert malicious_update.client_id == "client_malicious"
        assert "layer1.weight" in malicious_update.parameters

    def test_inject_trigger(self, attack):
        """Test trigger injection."""
        inputs = torch.randn(10, 28)
        poisoned = attack.inject_trigger(inputs)
        
        assert poisoned.shape == inputs.shape
        assert not torch.equal(poisoned, inputs)  # Should be modified

    def test_get_name(self, attack):
        """Test getting attack name."""
        assert attack.get_name() == "backdoor"


class TestModelPoisonAttack:
    """Tests for ModelPoisonAttack."""

    @pytest.fixture
    def global_params(self):
        """Create global model parameters."""
        return {"layer1.weight": torch.randn(128, 28)}

    def test_scaling_attack(self, global_params):
        """Test scaling attack."""
        attack = ModelPoisonAttack(attack_type="scaling", magnitude=-5.0)
        malicious_update = attack.execute("client_malicious", global_params)
        
        assert malicious_update.client_id == "client_malicious"
        assert "layer1.weight" in malicious_update.parameters
        assert attack.get_name() == "model_poison_scaling"

    def test_sign_flip_attack(self, global_params):
        """Test sign flip attack."""
        attack = ModelPoisonAttack(attack_type="sign_flip", magnitude=5.0)
        malicious_update = attack.execute("client_malicious", global_params)
        
        assert "layer1.weight" in malicious_update.parameters

    def test_gaussian_attack(self, global_params):
        """Test Gaussian noise attack."""
        attack = ModelPoisonAttack(attack_type="gaussian", magnitude=10.0)
        malicious_update = attack.execute("client_malicious", global_params)
        
        assert "layer1.weight" in malicious_update.parameters

    def test_target_layers(self, global_params):
        """Test attacking specific layers."""
        attack = ModelPoisonAttack(
            attack_type="scaling",
            target_layers=["layer1"],
        )
        malicious_update = attack.execute("client_malicious", global_params)
        
        # Should have modified layer1
        assert "layer1.weight" in malicious_update.parameters
