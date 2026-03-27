"""
Property Inference Attack - Complete Implementation
Day 27: Inferring dataset properties from model updates

Based on: Nasr et al., "Property Inference Attacks", USENIX Security 2019
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("PROPERTY INFERENCE ATTACK")
print("="*70)
print("\nInitializing attack...")

class PropertyInferenceAttack:
    """
    Property Inference Attack for Federated Learning.

    Infers dataset properties (e.g., fraud rate, gender ratio)
    from observed model updates using meta-classification.
    """

    def __init__(self, target_property='fraud_rate'):
        self.target_property = target_property

    def train_meta_classifier(self, n_datasets=100):
        """
        Train meta-classifier on synthetic datasets.

        Args:
            n_datasets: Number of synthetic datasets to generate
        """
        print(f"\nTraining meta-classifier on {n_datasets} datasets...")

        # Generate synthetic training data
        # Each "dataset" has a property value and associated gradients
        meta_features = []
        meta_labels = []

        for i in range(n_datasets):
            # Random property value (e.g., fraud rate between 0.01 and 0.10)
            property_value = np.random.uniform(0.01, 0.10)

            # Simulate gradient statistics (in practice, from FL updates)
            # Gradients encode information about the data distribution
            mean_gradient = np.random.randn(10) * (1 + property_value * 5)
            gradient_std = np.random.rand(10) * (1 - property_value * 0.5)
            gradient_norm = np.linalg.norm(mean_gradient) * (1 + property_value * 2)

            # Features: gradient statistics
            features = np.concatenate([
                mean_gradient,
                gradient_std,
                [gradient_norm],
                [property_value * 100],  # Bias in updates
            ])

            meta_features.append(features)
            meta_labels.append(property_value)

        # Train simple regression model
        from sklearn.linear_model import Ridge
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, meta_labels)

        print(f"  Meta-classifier trained on {n_datasets} datasets")
        print(f"  Feature dimension: {len(meta_features[0])}")

        return meta_features, meta_labels

    def execute_attack(self, observed_gradients):
        """
        Execute property inference attack on observed FL updates.

        Args:
            observed_gradients: Gradient updates from FL clients

        Returns:
            Predicted property value
        """
        print("\nExecuting property inference attack...")

        # Extract features from observed gradients
        mean_grad = np.mean([g['gradient'] for g in observed_gradients], axis=0)
        grad_std = np.std([g['gradient'] for g in observed_gradients], axis=0)
        grad_norm = np.linalg.norm(mean_grad)

        # Add bias feature
        n_features = len(mean_grad)
        bias = np.mean([g.get('bias', 0) for g in observed_gradients])

        features = np.concatenate([
            mean_grad,
            grad_std,
            [grad_norm],
            [bias * 100],
        ])

        # Predict property
        predicted_property = self.meta_model.predict([features])[0]

        # Confidence interval (simple approximation)
        predictions = []
        for _ in range(100):
            # Bootstrap samples for confidence
            sample_indices = np.random.choice(len(observed_gradients), len(observed_gradients), replace=True)
            sampled_grads = [observed_gradients[i] for i in sample_indices]

            sample_mean = np.mean([g['gradient'] for g in sampled_grads], axis=0)
            sample_std = np.std([g['gradient'] for g in sampled_grads], axis=0)
            sample_norm = np.linalg.norm(sample_mean)
            sample_bias = np.mean([g.get('bias', 0) for g in sampled_grads])

            sample_features = np.concatenate([
                sample_mean, sample_std, [sample_norm], [sample_bias * 100]
            ])

            pred = self.meta_model.predict([sample_features])[0]
            predictions.append(pred)

        confidence_interval = (np.percentile(predictions, 2.5), np.percentile(predictions, 97.5))

        return predicted_property, confidence_interval

# Simulate attack
print("\n" + "="*70)
print("SIMULATED ATTACK SCENARIO")
print("="*70)

attack = PropertyInferenceAttack(target_property='fraud_rate')

# Train meta-classifier
attack.train_meta_classifier(n_datasets=100)

# Simulate observed gradients from FL server
print("\nSimulating FL training...")
n_clients = 10
true_fraud_rate = 0.05  # Actual fraud rate in training data

observed_gradients = []
for i in range(n_clients):
    # Each client's gradients encode information about their data
    gradient = np.random.randn(10) * (1 + true_fraud_rate * 5) + 0.01 * i
    bias = true_fraud_rate * 10 + np.random.randn() * 0.01

    observed_gradients.append({
        'gradient': gradient,
        'bias': bias,
        'client_id': i,
    })

# Execute attack
predicted, ci = attack.execute_attack(observed_gradients)

print("\n" + "="*70)
print("ATTACK RESULTS")
print("="*70)
print(f"\nTrue fraud rate:   {true_fraud_rate:.3f}")
print(f"Predicted rate:     {predicted:.3f}")
print(f"95% CI:            [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"Absolute error:    {abs(predicted - true_fraud_rate):.3f}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
Property Inference Attack:
  • Goal: Infer dataset properties from model updates
  • Method: Meta-classifier trained on synthetic datasets
  • Features: Gradient statistics (mean, std, norm, bias)
  • Output: Property value with confidence interval

Attack Success Factors:
  ✅ More clients = Better inference
  ✅ More training rounds = More gradient samples
  ✅ Homogeneous data = Stronger signal
  ❌ Differential privacy = Mitigates attack

Defenses:
  ✅ DP noise (σ ≥ 1.0) obscures property information
  ✅ Gradient compression removes fine-grained information
  ✅ Secure aggregation prevents direct observation
""")

print("\n" + "="*70)
print("✅ Property Inference Attack Complete!")
print("="*70)
print("\nSee src/ directory for full implementation")
