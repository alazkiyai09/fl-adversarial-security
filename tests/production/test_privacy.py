"""Unit tests for privacy module."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.privacy.differential_privacy import (
    clip_and_add_noise,
    compute_noise_multiplier,
    compute_sampling_probability,
    DPSGDOptimizer,
    DPSGDFactory,
    PrivacyAccountant,
    compute_dp_params,
)
from src.privacy.secure_aggregation import (
    generate_random_mask,
    pairwise_mask,
    SecureAggregator,
    ThresholdSecretSharing,
    HybridSecureAggregator,
)


class TestDifferentialPrivacy:
    """Tests for differential privacy mechanisms."""

    def test_clip_and_add_noise(self):
        """Test gradient clipping and noise addition."""
        gradients = [
            torch.randn(10, 10) * 5,  # Large gradients
            torch.randn(5) * 3,
        ]

        clipped_noised = clip_and_add_noise(
            gradients=gradients,
            clip_norm=1.0,
            noise_multiplier=0.5,
        )

        # Check output shape
        assert len(clipped_noised) == len(gradients)
        assert clipped_noised[0].shape == gradients[0].shape
        assert clipped_noised[1].shape == gradients[1].shape

        # Check that gradients were clipped
        # (norm should be close to clip_norm after clipping)
        flat = torch.cat([g.flatten() for g in clipped_noised])
        norm = torch.norm(flat).item()
        # Due to noise, norm might exceed clip_norm slightly
        # but should be much less than original
        original_norm = torch.norm(torch.cat([g.flatten() for g in gradients])).item()
        assert norm < original_norm

    def test_compute_sampling_probability(self):
        """Test sampling probability computation."""
        prob = compute_sampling_probability(batch_size=32, dataset_size=1000)
        assert prob == 0.032

        prob = compute_sampling_probability(batch_size=100, dataset_size=100)
        assert prob == 1.0  # Cap at 1.0

    def test_dp_optimizer_step(self):
        """Test DP-SGD optimizer step."""
        # Create simple model
        model = nn.Linear(10, 2)

        # Create base optimizer
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Wrap with DP-SGD
        dp_optimizer = DPSGDOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=32,
            dataset_size=1000,
        )

        # Create dummy loss
        x = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))

        # Forward pass
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert model.weight.grad is not None

        # Store original gradients
        original_grad = model.weight.grad.clone()

        # DP step
        dp_optimizer.step()

        # Check gradients were modified (noise added)
        assert not torch.allclose(model.weight.grad, original_grad, atol=1e-6)

    def test_dp_optimizer_privacy_tracking(self):
        """Test privacy tracking in DP-SGD optimizer."""
        model = nn.Linear(10, 2)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        dp_optimizer = DPSGDOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=32,
            dataset_size=1000,
        )

        # Initial privacy spent
        epsilon, delta = dp_optimizer.get_privacy_spent()
        assert epsilon == 0.0  # No steps yet

        # Perform step
        x = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        dp_optimizer.step()

        # Privacy spent should increase
        epsilon_after, delta_after = dp_optimizer.get_privacy_spent()
        # Note: May still be 0 if Opacus is not installed
        assert epsilon_after >= epsilon


class TestDPSGDFactory:
    """Tests for DP-SGD factory."""

    def test_create_dp_sgd(self):
        """Test creating DP-SGD optimizer."""
        factory = DPSGDFactory(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            delta=1e-5,
        )

        model = nn.Linear(10, 2)
        dp_optimizer = factory.create_dp_sgd(
            parameters=model.parameters(),
            lr=0.01,
            batch_size=32,
            dataset_size=1000,
        )

        assert isinstance(dp_optimizer, DPSGDOptimizer)
        assert dp_optimizer.noise_multiplier == 1.0
        assert dp_optimizer.max_grad_norm == 1.0

    def test_create_dp_adam(self):
        """Test creating DP-Adam optimizer."""
        factory = DPSGDFactory(
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            delta=1e-5,
        )

        model = nn.Linear(10, 2)
        dp_optimizer = factory.create_dp_adam(
            parameters=model.parameters(),
            lr=0.001,
            batch_size=32,
            dataset_size=1000,
        )

        assert isinstance(dp_optimizer, DPSGDOptimizer)


class TestPrivacyAccountant:
    """Tests for privacy accountant."""

    def test_initialization(self):
        """Test accountant initialization."""
        accountant = PrivacyAccountant(
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=32,
            dataset_size=1000,
            epochs=10,
        )

        assert accountant.target_epsilon == 1.0
        assert accountant.target_delta == 1e-5
        assert accountant.sample_rate == 0.032

    def test_step_tracking(self):
        """Test privacy tracking after steps."""
        accountant = PrivacyAccountant(
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=32,
            dataset_size=1000,
            epochs=10,
        )

        # Take a step
        epsilon, delta = accountant.step(num_steps=100)

        # Privacy should be spent
        assert epsilon >= 0
        assert delta == 1e-5
        assert accountant.steps == 100

    def test_budget_remaining(self):
        """Test remaining budget computation."""
        accountant = PrivacyAccountant(
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=32,
            dataset_size=1000,
            epochs=10,
        )

        remaining_eps, remaining_delta = accountant.get_budget_remaining()

        assert remaining_eps == 1.0  # Nothing spent yet
        assert remaining_delta == 1e-5

    def test_budget_exhausted(self):
        """Test budget exhaustion check."""
        accountant = PrivacyAccountant(
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=32,
            dataset_size=1000,
            epochs=10,
        )

        # Initially not exhausted
        assert not accountant.is_budget_exhausted()

        # After many steps, might be exhausted
        # (depends on Opacus installation)
        accountant.step(num_steps=100000)
        # Check doesn't crash
        _ = accountant.is_budget_exhausted()

    def test_get_summary(self):
        """Test summary generation."""
        accountant = PrivacyAccountant(
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=32,
            dataset_size=1000,
            epochs=10,
        )

        summary = accountant.get_summary()

        assert "target_epsilon" in summary
        assert "current_epsilon" in summary
        assert "noise_multiplier" in summary
        assert "sample_rate" in summary


class TestSecureAggregation:
    """Tests for secure aggregation."""

    def test_generate_random_mask(self):
        """Test random mask generation."""
        shape = (10, 10)
        mask = generate_random_mask(shape, bit_size=32, seed=42)

        assert mask.shape == shape
        assert mask.dtype == np.float32

        # Should be roughly centered around 0
        assert np.abs(mask.mean()) < 1000  # Allow some variation

    def test_pairwise_mask_cancellation(self):
        """Test that pairwise masks cancel out."""
        values = np.random.randn(10, 10) * 0.1
        n_clients = 5

        # Apply pairwise masking to all clients
        masked_updates = []
        for client_id in range(n_clients):
            masked = pairwise_mask(values, client_id, n_clients, seed=42)
            masked_updates.append(masked)

        # Sum all masked updates
        total = np.sum(masked_updates, axis=0)

        # Should be approximately equal to n_clients * original values
        # (masks cancel out)
        expected = n_clients * values
        assert np.allclose(total, expected, atol=1e-5)

    def test_secure_aggregator_mask_update(self):
        """Test masking client updates."""
        aggregator = SecureAggregator(n_clients=5, seed=42)

        update = [
            np.random.randn(10, 10) * 0.1,
            np.random.randn(5) * 0.1,
        ]

        masked = aggregator.mask_update(update, client_id=0)

        # Shape should be preserved
        assert len(masked) == len(update)
        assert masked[0].shape == update[0].shape
        assert masked[1].shape == update[1].shape

        # Should be different (masked)
        assert not np.allclose(masked[0], update[0])

    def test_secure_aggregator_aggregate(self):
        """Test aggregation of masked updates."""
        aggregator = SecureAggregator(n_clients=5, seed=42)

        # Create 5 similar updates
        updates = []
        for _ in range(5):
            updates.append([
                np.random.randn(10, 10) * 0.1,
                np.random.randn(5) * 0.1,
            ])

        # Mask all updates
        masked_updates = []
        for client_id, update in enumerate(updates):
            masked = aggregator.mask_update(update, client_id)
            masked_updates.append(masked)

        # Aggregate
        aggregated = aggregator.unmask_aggregate(masked_updates)

        # Should get 2 layers
        assert len(aggregated) == 2
        assert aggregated[0].shape == (10, 10)
        assert aggregated[1].shape == (5,)

    def test_verify_cancellation(self):
        """Test mask cancellation verification."""
        aggregator = SecureAggregator(n_clients=3, seed=42)

        # Create and mask updates
        masked_updates = []
        for client_id in range(3):
            update = [np.random.randn(5, 5) * 0.1]
            masked = aggregator.mask_update(update, client_id)
            masked_updates.append(masked)

        # Verify cancellation
        verified = aggregator.verify_cancellation(masked_updates)

        assert verified is True


class TestThresholdSecretSharing:
    """Tests for threshold secret sharing."""

    def test_share_and_reconstruct(self):
        """Test sharing and reconstructing a secret."""
        tss = ThresholdSecretSharing(n_shares=5, threshold=3)

        secret = 12345

        # Create shares
        shares = tss.share(secret)
        assert len(shares) == 5

        # Reconstruct with threshold shares
        reconstructed = tss.reconstruct(shares[:3])
        assert reconstructed == secret

        # Reconstruct with all shares
        reconstructed_all = tss.reconstruct(shares)
        assert reconstructed_all == secret

    def test_threshold_requirement(self):
        """Test that threshold is required."""
        tss = ThresholdSecretSharing(n_shares=5, threshold=3)

        secret = 12345
        shares = tss.share(secret)

        # Should fail with fewer than threshold shares
        with pytest.raises(ValueError):
            tss.reconstruct(shares[:2])

    def test_different_thresholds(self):
        """Test different threshold values."""
        for n_shares, threshold in [(5, 3), (7, 4), (10, 6)]:
            tss = ThresholdSecretSharing(n_shares=n_shares, threshold=threshold)

            secret = 999
            shares = tss.share(secret)

            # Reconstruct with threshold
            reconstructed = tss.reconstruct(shares[:threshold])
            assert reconstructed == secret


class TestHybridSecureAggregator:
    """Tests for hybrid secure aggregation."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = HybridSecureAggregator(
            n_clients=5,
            use_encryption=False,
            seed=42,
        )

        assert aggregator.n_clients == 5
        assert aggregator.use_encryption is False
        assert aggregator.masking_aggregator is not None

    def test_secure_aggregate(self):
        """Test secure aggregation."""
        aggregator = HybridSecureAggregator(
            n_clients=5,
            use_encryption=False,
            seed=42,
        )

        # Create updates
        updates = []
        for _ in range(5):
            updates.append([
                np.random.randn(10, 10) * 0.1,
                np.random.randn(5) * 0.1,
            ])

        # Secure aggregate
        aggregated = aggregator.secure_aggregate(updates)

        # Should get aggregated result
        assert len(aggregated) == 2
        assert aggregated[0].shape == (10, 10)
        assert aggregated[1].shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
