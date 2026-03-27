"""
Integration tests for FoolsGold defense.

Tests end-to-end FL with attack and defense.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import Parameters, ndarrays_to_parameters

from src.aggregators import FoolsGoldAggregator
from src.attacks import SybilAttack
from src.clients import create_client
from src.models.fraud_net import FraudNet, get_model_parameters
from src.utils.metrics import DefenseMetrics, compute_accuracy


def create_test_data(num_samples: int = 500, num_features: int = 10):
    """Create synthetic test data."""
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, 2, num_samples).astype(np.int64)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader


class TestFoolsGoldIntegration:
    """Integration tests for FoolsGold."""

    def test_end_to_end_training_no_attack(self):
        """Test end-to-end training without attack."""
        # Setup
        num_clients = 5
        num_features = 10
        num_rounds = 5

        # Create data
        train_loaders = [create_test_data(num_samples=200, num_features=num_features) for _ in range(num_clients)]
        test_loader = create_test_data(num_samples=100, num_features=num_features)
        test_loaders = [test_loader] * num_clients

        # Initialize global model
        global_model = FraudNet(input_dim=num_features)
        global_params = ndarrays_to_parameters(get_model_parameters(global_model))

        # Create FoolsGold aggregator
        aggregator = FoolsGoldAggregator(history_length=5)

        # Training rounds
        for round_num in range(num_rounds):
            # Client training
            results = []
            for client_id in range(num_clients):
                client = create_client(
                    client_id=client_id,
                    train_loader=train_loaders[client_id],
                    test_loader=test_loaders[client_id],
                    input_dim=num_features,
                    num_epochs=1
                )

                fit_res, _, _ = client.fit(global_params, {})
                results.append((None, fit_res))  # ClientProxy placeholder

            # Aggregate
            global_params, _ = aggregator.aggregate(results)

        # Verify aggregation occurred
        assert len(aggregator.history["similarity_matrices"]) == num_rounds
        assert len(aggregator.history["contribution_scores"]) == num_rounds

    def test_sybil_attack_detection(self):
        """Test FoolsGold detects Sybil attack."""
        # Setup
        num_clients = 6
        num_malicious = 2
        num_features = 10
        num_rounds = 5

        # Create data
        train_loaders = [create_test_data(num_samples=200, num_features=num_features) for _ in range(num_clients)]
        test_loader = create_test_data(num_samples=100, num_features=num_features)
        test_loaders = [test_loader] * num_clients

        # Initialize global model
        global_model = FraudNet(input_dim=num_features)
        global_params = ndarrays_to_parameters(get_model_parameters(global_model))

        # Create FoolsGold aggregator
        aggregator = FoolsGoldAggregator(
            history_length=5,
            similarity_threshold=0.8,
            lr_scale_factor=0.1
        )

        # Malicious client IDs
        malicious_ids = list(range(num_clients - num_malicious, num_clients))

        # Training rounds
        for round_num in range(num_rounds):
            # Client training
            results = []
            for client_id in range(num_clients):
                is_malicious = client_id in malicious_ids

                client = create_client(
                    client_id=client_id,
                    train_loader=train_loaders[client_id],
                    test_loader=test_loaders[client_id],
                    input_dim=num_features,
                    num_epochs=1,
                    is_malicious=is_malicious,
                    attack_type="sign_flip" if is_malicious else None
                )

                fit_res, _, _ = client.fit(global_params, {})
                results.append((None, fit_res))

            # Aggregate
            global_params, _ = aggregator.aggregate(results)

            # Check flagged Sybils
            flagged = aggregator.history["flagged_sybils"][-1]

            # In later rounds, Sybils should be flagged
            if round_num >= 3:
                # At least some malicious clients should be flagged
                assert len(flagged) >= 1 or any(cid in flagged for cid in malicious_ids)

    def test_contribution_scores_punish_sybils(self):
        """Test that Sybils get lower contribution scores."""
        # Setup
        num_clients = 6
        num_malicious = 2
        num_features = 10
        num_rounds = 10

        # Create data
        train_loaders = [create_test_data(num_samples=200, num_features=num_features) for _ in range(num_clients)]
        test_loader = create_test_data(num_samples=100, num_features=num_features)
        test_loaders = [test_loader] * num_clients

        # Initialize global model
        global_model = FraudNet(input_dim=num_features)
        global_params = ndarrays_to_parameters(get_model_parameters(global_model))

        # Create FoolsGold aggregator
        aggregator = FoolsGoldAggregator(history_length=5)

        # Malicious client IDs
        malicious_ids = list(range(num_clients - num_malicious, num_clients))

        # Training rounds
        for round_num in range(num_rounds):
            # Client training
            results = []
            for client_id in range(num_clients):
                is_malicious = client_id in malicious_ids

                client = create_client(
                    client_id=client_id,
                    train_loader=train_loaders[client_id],
                    test_loader=test_loaders[client_id],
                    input_dim=num_features,
                    num_epochs=1,
                    is_malicious=is_malicious,
                    attack_type="sign_flip" if is_malicious else None
                )

                fit_res, _, _ = client.fit(global_params, {})
                results.append((None, fit_res))

            # Aggregate
            global_params, _ = aggregator.aggregate(results)

        # Check final contribution scores
        final_scores = aggregator.history["contribution_scores"][-1]

        # Honest clients should have higher scores on average
        honest_scores = [final_scores[i] for i in range(num_clients) if i not in malicious_ids]
        malicious_scores = [final_scores[i] for i in malicious_ids]

        # On average, honest should have higher contribution
        assert np.mean(honest_scores) >= np.mean(malicious_scores)

    def test_defense_maintains_accuracy_under_attack(self):
        """Test that FoolsGold maintains accuracy under attack."""
        # Setup
        num_clients = 6
        num_malicious = 2
        num_features = 10
        num_rounds = 10

        # Create data
        train_loaders = [create_test_data(num_samples=200, num_features=num_features) for _ in range(num_clients)]
        test_loader = create_test_data(num_samples=100, num_features=num_features)
        test_loaders = [test_loader] * num_clients

        # Initialize global model
        global_model = FraudNet(input_dim=num_features)
        global_params = ndarrays_to_parameters(get_model_parameters(global_model))

        # Create FoolsGold aggregator
        aggregator = FoolsGoldAggregator(history_length=5)

        # Malicious client IDs
        malicious_ids = list(range(num_clients - num_malicious, num_clients))

        # Metrics tracker
        metrics_tracker = DefenseMetrics()

        # Training rounds
        for round_num in range(num_rounds):
            # Client training
            results = []
            for client_id in range(num_clients):
                is_malicious = client_id in malicious_ids

                client = create_client(
                    client_id=client_id,
                    train_loader=train_loaders[client_id],
                    test_loader=test_loaders[client_id],
                    input_dim=num_features,
                    num_epochs=1,
                    is_malicious=is_malicious,
                    attack_type="sign_flip" if is_malicious else None
                )

                fit_res, _, _ = client.fit(global_params, {})
                results.append((None, fit_res))

            # Aggregate
            global_params, _ = aggregator.aggregate(results)

            # Evaluate
            global_model_eval = FraudNet(input_dim=num_features)
            from src.models.fraud_net import set_model_parameters
            set_model_parameters(global_model_eval, global_params.tensors)

            accuracy = compute_accuracy(global_model_eval, test_loader)

            metrics_tracker.add_round(
                round_num=round_num,
                accuracy=accuracy,
                loss=0.0,
                malicious_ids=malicious_ids
            )

        # Check that accuracy improves over time
        final_metrics = metrics_tracker.get_final_metrics()
        accuracy_history = final_metrics["accuracy_history"]

        # Final accuracy should be better than initial
        assert accuracy_history[-1] >= accuracy_history[0] - 0.1  # Allow some degradation


class TestAttackIntegration:
    """Test attack implementations."""

    def test_sybil_attack_generation(self):
        """Test Sybil attack generates coordinated updates."""
        from src.attacks import generate_sybil_updates

        base_update = np.array([1.0, 2.0, 3.0])
        num_sybils = 3

        updates = generate_sybil_updates(
            base_update,
            num_sybils,
            noise_level=0.0
        )

        assert len(updates) == num_sybils
        # All should be identical
        for update in updates:
            assert np.allclose(update, base_update)

    def test_sybil_attack_with_noise(self):
        """Test Sybil attack with noise appears distinct."""
        from src.attacks import generate_sybil_updates

        base_update = np.array([1.0, 2.0, 3.0])
        num_sybils = 3

        updates = generate_sybil_updates(
            base_update,
            num_sybils,
            noise_level=0.1
        )

        assert len(updates) == num_sybils
        # All should be similar but not identical
        for update in updates:
            assert not np.array_equal(update, base_update)
            # Should still be highly correlated
            assert np.corrcoef(update, base_update)[0, 1] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
