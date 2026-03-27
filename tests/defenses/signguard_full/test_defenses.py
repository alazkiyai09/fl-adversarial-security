"""Tests for baseline defense implementations."""

import pytest
import torch
from typing import List

from signguard.defenses import KrumDefense, TrimmedMeanDefense, FoolsGoldDefense, BulyanDefense
from signguard.core.types import ModelUpdate


@pytest.fixture
def sample_updates() -> List[ModelUpdate]:
    """Create sample model updates."""
    updates = []
    
    # Create normal updates
    for i in range(8):
        params = {
            "layer1.weight": torch.randn(128, 28) * 0.1,
            "layer1.bias": torch.randn(128) * 0.1,
        }
        updates.append(ModelUpdate(
            client_id=f"honest_{i}",
            round_num=0,
            parameters=params,
            num_samples=100,
            metrics={"loss": 0.5},
        ))
    
    # Create Byzantine update (larger magnitude)
    malicious_params = {
        "layer1.weight": torch.randn(128, 28) * 10.0,
        "layer1.bias": torch.randn(128) * 10.0,
    }
    updates.append(ModelUpdate(
        client_id="malicious_0",
        round_num=0,
        parameters=malicious_params,
        num_samples=100,
        metrics={"loss": 5.0},
    ))
    
    return updates


@pytest.fixture
def global_model():
    """Create global model."""
    return {
        "layer1.weight": torch.randn(128, 28),
        "layer1.bias": torch.randn(128),
    }


class TestKrumDefense:
    """Tests for KrumDefense."""

    def test_aggregate(self, sample_updates, global_model):
        """Test Krum aggregation."""
        defense = KrumDefense(num_byzantines=1)
        result = defense.aggregate(sample_updates, global_model)
        
        assert len(result.participating_clients) > 0
        assert "layer1.weight" in result.global_model
        assert result.execution_time > 0

    def test_multi_krum(self, sample_updates, global_model):
        """Test multi-Krum variant."""
        defense = KrumDefense(num_byzantines=1, multi_krum=True)
        result = defense.aggregate(sample_updates, global_model)
        
        # Should select multiple clients
        assert len(result.participating_clients) > 1

    def test_invalid_byzantine_count(self, sample_updates, global_model):
        """Test error on invalid Byzantine count."""
        defense = KrumDefense(num_byzantines=10)  # Too many
        
        with pytest.raises(ValueError):
            defense.aggregate(sample_updates, global_model)


class TestTrimmedMeanDefense:
    """Tests for TrimmedMeanDefense."""

    def test_aggregate(self, sample_updates, global_model):
        """Test trimmed mean aggregation."""
        defense = TrimmedMeanDefense(trim_ratio=0.2)
        result = defense.aggregate(sample_updates, global_model)
        
        assert len(result.participating_clients) == len(sample_updates)
        assert "layer1.weight" in result.global_model

    def test_per_parameter(self, sample_updates, global_model):
        """Test per-parameter trimming."""
        defense = TrimmedMeanDefense(trim_ratio=0.2, per_parameter=True)
        result = defense.aggregate(sample_updates, global_model)
        
        assert "layer1.weight" in result.global_model

    def test_invalid_trim_ratio(self):
        """Test error on invalid trim ratio."""
        with pytest.raises(ValueError):
            TrimmedMeanDefense(trim_ratio=0.6)  # Must be < 0.5


class TestFoolsGoldDefense:
    """Tests for FoolsGoldDefense."""

    def test_aggregate(self, sample_updates, global_model):
        """Test FoolsGold aggregation."""
        defense = FoolsGoldDefense(history_length=10)
        result = defense.aggregate(sample_updates, global_model)
        
        assert len(result.participating_clients) == len(sample_updates)
        assert "layer1.weight" in result.global_model

    def test_weight_computation(self, sample_updates, global_model):
        """Test weight computation across rounds."""
        defense = FoolsGoldDefense(history_length=5)
        
        # First round
        result1 = defense.aggregate(sample_updates, global_model)
        weights1 = result1.reputation_updates
        
        # Second round
        result2 = defense.aggregate(sample_updates, global_model)
        weights2 = result2.reputation_updates
        
        # Should have computed weights
        assert len(weights1) > 0
        assert len(weights2) > 0


class TestBulyanDefense:
    """Tests for BulyanDefense."""

    def test_aggregate(self, sample_updates, global_model):
        """Test Bulyan aggregation."""
        # Need enough clients for Bulyan (n > 4f)
        defense = BulyanDefense(num_byzantines=1)
        
        # Add more clients to meet requirement
        more_updates = sample_updates.copy()
        for i in range(10):
            params = {
                "layer1.weight": torch.randn(128, 28) * 0.1,
                "layer1.bias": torch.randn(128) * 0.1,
            }
            more_updates.append(ModelUpdate(
                client_id=f"client_{i}",
                round_num=0,
                parameters=params,
                num_samples=100,
            ))
        
        result = defense.aggregate(more_updates, global_model)
        
        assert len(result.participating_clients) > 0
        assert "layer1.weight" in result.global_model

    def test_insufficient_clients(self, sample_updates, global_model):
        """Test error on insufficient clients."""
        defense = BulyanDefense(num_byzantines=5)  # Requires n > 20
        
        with pytest.raises(ValueError):
            defense.aggregate(sample_updates, global_model)
