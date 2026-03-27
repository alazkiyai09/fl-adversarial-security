# Privacy Guarantees

This document provides formal privacy guarantees for the Privacy-Preserving Federated Learning Fraud Detection System.

## Table of Contents

1. [Differential Privacy](#differential-privacy)
2. [Secure Aggregation](#secure-aggregation)
3. [Threat Models](#threat-models)
4. [Privacy Parameters](#privacy-parameters)
5. [Best Practices](#best-practices)

---

## Differential Privacy

### Formal Guarantee

The system provides **(ε, δ)-differential privacy** for individual training examples through DP-SGD.

**Definition**: A randomized algorithm M is (ε, δ)-differentially private if for any two adjacent datasets D and D' (differing in one element) and any set of outputs S:

```
P[M(D) ∈ S] ≤ e^ε * P[M(D') ∈ S] + δ
```

### Implementation

**DP-SGD Algorithm**:
```
For each training step:
  1. Compute gradients ∇L
  2. Clip gradients: ∇L̂ = ∇L * min(1, C/‖∇L‖)
  3. Add noise: ∇L̃ = ∇L̂ + N(0, σ²C²I)
  4. Update parameters: θ = θ - η * ∇L̃
```

Where:
- C = max_grad_norm (clipping threshold)
- σ = noise_multiplier
- η = learning rate

### Privacy Accounting

**RDP (Rényi Differential Privacy)**:
```
ε(α) = q * sqrt(2 * log(1.25/δ)) / σ * sqrt(α) * steps
```

Where:
- q = sampling_probability = batch_size / dataset_size
- α = Rényi order (typically 2-100)
- δ = target delta (e.g., 1e-5)

**Conversion to (ε, δ)**:
```
ε = min_α ε(α)
```

### What is Protected?

DP protects **individual training examples** (transactions):

✅ **Protected**:
- Whether a specific transaction was in training data
- Whether a specific customer's transactions were used
- Presence of any individual's data

❌ **Not Protected** (but may be protected by other mechanisms):
- Model architecture and hyperparameters
- Global model performance metrics
- Aggregated statistics (if secure aggregation disabled)

### Configuration Guide

| Preset | ε | δ | Use Case |
|--------|---|---|----------|
| `privacy_high` | 0.1 | 1e-6 | Highly sensitive data |
| `privacy_medium` | 1.0 | 1e-5 | Standard fraud detection |
| `performance` | 10.0 | 1e-4 | Research/benchmarking |

**Privacy-Utility Trade-off**:
- Lower ε = More privacy, lower accuracy
- Higher ε = Less privacy, higher accuracy
- Typical range: ε ∈ [0.1, 10.0]

---

## Secure Aggregation

### Formal Guarantee

Secure aggregation provides **information-theoretic privacy** for client updates under the honest-but-curious server model.

### Pairwise Masking Protocol

**Protocol**:
```
For each pair (i, j) where i < j:
  1. Generate random mask_ij
  2. Client i: update += mask_ij
  3. Client j: update -= mask_ij

After aggregation:
  Σ updates = Σ (true_updates + masks)
            = Σ true_updates + Σ masks
            = Σ true_updates + 0  (masks cancel)
```

### Security Properties

**Correctness**: Honest clients' updates aggregate correctly
```
Σ_{honest} update_i = true_aggregate
```

**Privacy**: Server learns nothing about individual updates
```
P[Server sees update_i | ciphertexts] = P[Server sees update_i]
```

**Robustness**: System tolerates dropouts
- Mask sharing handles client failures
- Threshold secret sharing for critical aggregation

### Threat Model

**Honest-but-Curious Server**:
- ✅ Correctly follows protocol
- ✅ Attempts to learn individual updates
- ❌ Does NOT deviate from protocol

**Dishonest Clients**:
- ✅ May send arbitrary updates
- ✅ May collude with server
- ❌ Protected by SignGuard defense

**Collusion Resistance**:
- Require > n/2 clients to reconstruct individual updates
- n = total number of clients
- Stronger for larger n

### What is Protected?

Secure aggregation protects **client updates**:

✅ **Protected**:
- Individual model parameters/gradients
- Whether a specific client participated
- Client-specific patterns in updates

❌ **Not Protected**:
- Aggregated global model
- Number of participating clients
- Metadata (client IDs if exposed)

### Configuration

**Enable Secure Aggregation**:
```yaml
privacy:
  secure_agg_enabled: true
```

**Trade-offs**:
- Pros: Strong privacy, low overhead
- Cons: Requires additional communication rounds
- Alternatives: Differential privacy (alone), homomorphic encryption

---

## Threat Models

### Threat Landscape

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Curious Server** | Server attempts to inspect individual updates | Secure Aggregation, DP |
| **Malicious Client** | Client sends poisoned updates | SignGuard, Attack Detection |
| **Data Poisoning** | Client trains on corrupted data | SignGuard, Input Validation |
| **Model Poisoning** | Client sends malicious gradients | SignGuard, Krum, etc. |
| **Backdoor** | Client plants backdoor in global model | Backdoor Detection |
| **Label Flipping** | Client flips labels (fraud ↔ legitimate) | Label Flipping Detection |
| **Collusion** | Multiple clients collude | Threshold > n/2 required |

### Defense in Depth

**Layer 1: Input Validation**
- Data quality checks
- Schema validation
- Outlier detection

**Layer 2: Local Privacy**
- DP-SGD on client
- Gradient clipping
- Local noise addition

**Layer 3: Secure Communication**
- Pairwise masking
- Encryption (TLS)
- Secure aggregation

**Layer 4: Server Defense**
- SignGuard aggregation
- Attack detection
- Anomaly logging

**Layer 5: Monitoring**
- MLflow tracking
- Alerting
- Audit logs

---

## Privacy Parameters

### ε (Epsilon)

**Interpretation**:
- ε = 0.1: Very strong privacy (significant utility loss)
- ε = 1.0: Moderate privacy (recommended)
- ε = 10.0: Weak privacy (minimal utility loss)

**Rule of Thumb**:
```
Privacy Level    ε Range    Use Case
─────────────    ────────    ─────────
Very Strong     < 0.5     Health data, financial records
Strong          0.5 - 2    Standard ML applications
Moderate        2 - 10     Data analytics
Weak            > 10       Public datasets
```

### δ (Delta)

**Interpretation**:
- Probability that pure DP fails
- Typically set to 1/n_samples or smaller
- δ = 1e-5 is standard

**Calculation**:
```
δ ≤ 1 / n_samples
δ ≤ 1 / 10000 = 1e-4
```

### noise_multiplier (σ)

**Relationship to ε**:
```
σ ∝ 1/ε
```

**Higher σ**:
- Stronger privacy (lower ε)
- Lower model utility
- Slower convergence

**Guidelines**:
| noise_multiplier | Approximate ε | Notes |
|------------------|---------------|-------|
| 0.1 - 0.5 | 5.0 - 10.0 | Weak privacy |
| 0.5 - 1.0 | 1.0 - 5.0 | Moderate privacy |
| 1.0 - 5.0 | 0.1 - 1.0 | Strong privacy |
| > 5.0 | < 0.1 | Very strong privacy |

### max_grad_norm (C)

**Purpose**: Limits sensitivity of each example

**Trade-offs**:
- Lower C: More clipping, more noise needed
- Higher C: Less clipping, less privacy per example

**Typical Values**:
- C = 0.5 - 1.0 (recommended)
- C = 1.0 - 5.0 (if gradients are small)
- C = 0.1 - 0.5 (if gradients are large)

### Example Calculations

**Scenario**: 10,000 samples, 100 rounds, batch_size=32

**Compute sampling probability**:
```
q = batch_size / n_samples = 32 / 10000 = 0.0032
```

**Compute steps**:
```
steps = rounds * (n_samples / batch_size) = 100 * 312 = 31,200
```

**Target**: ε = 1.0, δ = 1e-5

**Required noise_multiplier** (using Opacus):
```
noise_multiplier ≈ 1.2
```

---

## Best Practices

### 1. Privacy Budgeting

**Do's**:
- Set target ε before training
- Track ε throughout training
- Stop when budget exhausted
- Log privacy spent

**Don'ts**:
- Don't train indefinitely (ε → ∞)
- Don't ignore δ (set to 1e-5 or smaller)
- Don't reuse privacy budget across experiments

### 2. Parameter Selection

**Starting Point**:
```yaml
privacy:
  dp_enabled: true
  epsilon: 1.0
  delta: 1e-5
  noise_multiplier: 1.0
  max_grad_norm: 1.0
```

**Tuning**:
1. Start with noise_multiplier = 1.0
2. If accuracy too low, decrease to 0.5
3. If privacy too weak, increase to 1.5
4. Adjust max_grad_norm if needed

### 3. Validation

**Sanity Checks**:
```python
# After training
accountant = PrivacyAccountant(...)
epsilon, delta = accountant.get_budget_spent()

assert epsilon <= target_epsilon, "Privacy budget exceeded!"
assert delta == target_delta, "Delta mismatch!"
```

### 4. Monitoring

**Track Per Round**:
```python
for round in range(n_rounds):
    # Training step
    epsilon, delta = accountant.step(num_steps)

    # Log to MLflow
    mlflow.log_metrics({
        "privacy_epsilon": epsilon,
        "privacy_delta": delta,
    }, step=round)

    # Check budget
    if accountant.is_budget_exhausted():
        logger.warning("Privacy budget exhausted!")
        break
```

### 5. Documentation

**Report Your Privacy**:
```python
{
    "target_epsilon": 1.0,
    "target_delta": 1e-5,
    "achieved_epsilon": 0.95,
    "noise_multiplier": 1.0,
    "max_grad_norm": 1.0,
    "sampling_probability": 0.0032,
    "training_steps": 31200,
    "mechanism": "DP-SGD (Gaussian)",
    "accounting": "RDP",
}
```

---

## References

1. **DP-SGD**: Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
2. **RDP Accountant**: Mironov et al., "Rényi Differential Privacy" (ICML 2017)
3. **Secure Aggregation**: Bonawitz et al., "Practical Secure Aggregation" (CCS 2017)
4. **FL Privacy**: McMahan et al., "Communication-Efficient Learning" (AISTATS 2017)

## FAQ

**Q: Is my data private with ε=1.0?**
A: Yes, ε=1.0 provides meaningful privacy guarantees. It's a common choice.

**Q: Can I continue training after ε is exhausted?**
A: Technically yes, but you lose formal privacy guarantees. Not recommended.

**Q: Do I need both DP and secure aggregation?**
A: They protect different things:
- DP: Protects individual training examples
- Secure Aggregation: Protects client updates
- Using both provides defense in depth.

**Q: How do I choose between privacy presets?**
A: Consider:
- Data sensitivity (health/finance → privacy_high)
- Regulatory requirements (GDPR → privacy_high)
- Performance needs (production → performance)

**Q: What if attackers collude?**
A: Secure aggregation requires > n/2 colluding clients. SignGuard provides additional protection against malicious clients.
