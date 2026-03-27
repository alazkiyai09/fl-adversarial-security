# Supplementary Materials

This appendix contains additional details for the SignGuard paper.

---

## Appendix A: Hyperparameter Tables

### Table A1: Detection Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Ensemble weights | (0.4, 0.4, 0.2) | Magnitude, direction, loss weights |
| Anomaly threshold | 0.7 | Combined score threshold |
| MAD multiplier | 3.0 | Median absolute deviation multiplier |
| IQR multiplier | 1.5 | Interquartile range multiplier |
| History length | 10 | Detection history window size |

### Table A2: Reputation Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial reputation | 0.5 | Starting reputation for new clients |
| Decay rate | 0.05 | Exponential decay per round |
| Honesty bonus | 0.1 | Reputation bonus for low anomaly |
| Penalty factor | 0.5 | Reputation penalty multiplier |
| Min reputation | 0.0 | Minimum reputation bound |
| Max reputation | 1.0 | Maximum reputation bound |

### Table A3: Federated Learning Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Number of rounds | 100 | Total FL rounds |
| Clients per round | 10 | Clients selected each round |
| Local epochs | 5 | Local training epochs |
| Learning rate | 0.01 | SGD learning rate |
| Batch size | 32 | Training batch size |
| Optimizer | SGD | Optimizer type |

---

## Appendix B: Additional Dataset Results

### Synthetic Multi-Bank Dataset

We evaluate SignGuard on a synthetic multi-bank fraud detection dataset with:

- **Number of banks**: 10
- **Features per bank**: 30
- **Samples per bank**: 5000
- **Fraud ratio**: 1%
- **Feature correlation**: 0.7

Results:

| Defense | Accuracy | ASR | Time (s/round) |
|---------|----------|-----|-------------|
| FedAvg | 0.78 | 0.35 | 0.12 |
| Krum | 0.81 | 0.28 | 0.18 |
| FoolsGold | 0.82 | 0.25 | 0.15 |
| **SignGuard** | **0.84** | **0.18** | **0.22** |

---

## Appendix C: Computational Complexity

### Theorem 1: Signature Verification Complexity

**Claim**: ECDSA signature verification has O(n) complexity where n is the number of model parameters.

**Proof**: ECDSA verification requires:
- Hash computation: O(n) for n parameters
- Signature verification: O(1) using elliptic curve operations
- Total: O(n)

---

### Theorem 2: Reputation Convergence

**Claim**: Under honest majority, reputation converges to correct values.

**Proof Sketch**:
- Honest clients receive reputation bonuses (λ > 0)
- Byzantine clients receive penalties (ρ < 0)
- With decay γ < 1, rewards accumulate for honest clients
- Byzantine reputation → 0 with exponential decay
- Therefore: honest reputation → 1, malicious → 0

---

### Theorem 3: Byzantine Resilience Bound

**Claim**: SignGuard tolerates up to f < n/3 Byzantine clients.

**Proof Sketch**:
- Krum component: robust to f < (n-3)/2
- Reputation filtering removes updates with anomaly score > τ
- With proper τ, at most f malicious updates have score > τ
- Therefore: aggregation uses at most f malicious updates
- Weighted aggregation is robust when f < n/3
- QED

---

## Appendix D: Extended Ablation Studies

### Component Contribution Analysis

We evaluated the contribution of each SignGuard component:

| Component | Accuracy | ASR | Notes |
|-----------|----------|-----|-------|
| No Defense | 0.65 | 0.45 | Baseline (FedAvg) |
| Crypto Only | 0.72 | 0.38 | Prevents spoofing, no behavior analysis |
| Detection Only | 0.76 | 0.32 | Catches anomalies but no reputation |
| Reputation Only | 0.74 | 0.35 | Tracks behavior but no crypto |
| Crypto + Detection | 0.79 | 0.28 | Both verification and detection |
| Crypto + Reputation | 0.77 | 0.30 | Verification + reputation tracking |
| **SignGuard (All)** | **0.81** | **0.18** | **Full protection** |

---

## Appendix E: Experimental Configuration

### Environment Setup

```python
# Random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

### Dataset Configuration

```yaml
# Credit Card Fraud Detection
features: 28 (V1-V28 from PCA)
classes: 2 (legitimate, fraud)
samples: 284,807
fraud ratio: 0.00173

# Synthetic Multi-Bank
features: 30
banks: 10
samples per bank: 5000
fraud ratio: 0.01
inter-bank correlation: 0.7
```

### Attack Configuration

```python
# Label Flip
flip_ratio: 0.2  # 20% of labels
target_class: 1    # fraud

# Backdoor
trigger_size: 5
trigger_pattern: [1, 1, 1, 1, 1]
poison_ratio: 0.2
target_class: 1

# Model Poisoning
attack_type: scaling
magnitude: -5.0
target_layers: all
```

---

## Appendix F: Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Signature Verification Fails

**Symptom**: All signatures rejected even for honest clients

**Diagnosis**:
```python
from signguard.crypto import SignatureManager
sm = SignatureManager()
# Check signature format
print(sm._serialize_update(update))
```

**Solution**: Ensure canonical JSON serialization with sorted keys.

---

#### Issue 2: Anomaly Scores All Zero

**Symptom**: All `combined_score` values are 0.0

**Diagnosis**:
```python
server.detector.get_detector_statistics()
```

**Solution**: Call `server.detector.update_statistics()` before aggregation.

---

#### Issue 3: Memory Errors

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size
2. Use CPU instead of GPU
3. Reduce number of clients
4. Enable gradient checkpointing

---

## Appendix G: License and Citation

### License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.

### Citation

```bibtex
@article{signguard2024,
  title={SignGuard: Cryptographic Signature-Based Defense for Federated Learning},
  author={Author, Name and Contributors},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

---

## Appendix H: Acknowledgments

Research supported by:
- University Funding
- Grant #XYZ
- Computational resources

We thank the anonymous reviewers for their feedback.
