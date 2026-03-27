# SignGuard Development Guide

This guide is for contributors and developers working on SignGuard.

---

## Architecture Overview

SignGuard consists of several interacting components:

```
┌─────────────────────────────────────────────────────────┐
│                      SignGuard Server                        │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. Signature Verification                 │    │
│  │ 2. Anomaly Detection                    │    │
│  │ 3. Reputation Updates                   │    │
│  │ 4. Weighted Aggregation                │    │
│  └────────────────────────────────────────────────┘    │
│                                                       │
┌─────────────────────────────────────────────────────────┐
│                    SignGuard Client (×n)                │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. Local Training                           │    │
│  │ 2. Update Computation                       │    │
│  │ 3. ECDSA Signing                           │    │
│  └────────────────────────────────────────────────┘    │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Cryptographic Module (`signguard/crypto/`)

**Purpose**: ECDSA signature generation and verification

**Key Classes**:
- `SignatureManager`: Signs and verifies model updates
- `KeyManager`: File-based key storage
- `KeyStore`: In-memory key storage for testing

**Design Decisions**:
- SECP256R1 (P-256) curve for security
- SHA-256 hashing for canonical serialization
- Password-protected private keys (optional)
- Base64 encoding for portability

---

### 2. Detection Module (`signguard/detection/`)

**Purpose**: Detect anomalous model updates

**Key Classes**:
- `L2NormDetector`: Magnitude-based detection
- `CosineSimilarityDetector`: Direction-based detection
- `LossDeviationDetector`: Loss-based detection
- `EnsembleDetector`: Multi-factor fusion

**Design Decisions**:
- MAD (Median Absolute Deviation) for robust thresholding
- Cosine similarity for direction anomaly detection
- IQR (Interquartile Range) for loss anomaly detection
- Weighted ensemble with configurable weights

---

### 3. Reputation Module (`signguard/reputation/`)

**Purpose**: Track client trustworthiness over time

**Key Classes**:
- `DecayReputationSystem`: Time-decay reputation with bonus/penalty

**Design Decisions**:
- Exponential decay for reputation
- Honesty bonus for low anomaly scores
- Penalty multiplier for high anomaly scores
- [0, 1] bounded reputation

---

### 4. Aggregation Module (`signguard/aggregation/`)

**Purpose**: Aggregate client updates with reputation weighting

**Key Classes**:
- `WeightedAggregator`: Reputation-weighted averaging

**Design Decisions**:
- Minimum reputation threshold for participation
- Normalized weights for aggregation
- Fallback to robust aggregation if needed

---

## Testing Strategy

### Unit Tests

Located in `tests/`:
- `test_crypto.py`: Cryptographic operations
- `test_detection.py`: Anomaly detectors
- `test_reputation.py`: Reputation systems
- `test_aggregation.py`: Aggregation methods
- `test_attacks.py`: Attack implementations
- `test_defenses.py`: Baseline defenses
- `test_integration.py`: End-to-end FL simulation

### Integration Tests

Located in `tests/test_integration.py`:
- Multi-round FL simulation
- Malicious client handling
- Checkpoint save/load
- Server-client interaction

### Test Coverage

Current: 69% overall
Target: >80%

---

## Performance Considerations

### Computational Complexity

| Operation | Complexity | Notes |
|------------|------------|-------|
| ECDSA sign | O(n) | n = number of parameters |
| ECDSA verify | O(n) | Single operation |
| Anomaly detect | O(n) | Per client |
| Aggregation | O(mn) | m = clients, n = parameters |

### Communication Overhead

Per client per round:
- Model update: O(n)
- Signature: O(1)
- Public key: O(1)

Total: O(n) per client

### Memory Overhead

- SignGuard: +15-20% vs FedAvg
- Mostly from signature/reputation storage
- Scales linearly with number of clients

---

## Extending SignGuard

### Adding New Detectors

1. Create detector class inheriting from `AnomalyDetector`
2. Implement `compute_score()` and `update_statistics()`
3. Register in `EnsembleDetector`

Example:

```python
from signguard.detection.base import AnomalyDetector
from signguard.core.types import ModelUpdate
import torch

class MyDetector(AnomalyDetector):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def compute_score(self, update, global_model, client_history=None):
        # Your logic here
        return 0.5
    
    def update_statistics(self, updates, global_model):
        # Update internal state
        pass
```

### Adding New Attacks

1. Create attack class inheriting from `Attack`
2. Implement `execute()` and `get_name()`
3. Register in `signguard/attacks/__init__.py`

### Adding New Defenses

1. Create defense class
2. Implement `aggregate()` method
3. Register in `signguard/defenses/__init__.py`

---

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Component State

```python
# Check server state
server.get_statistics()

# Check detector statistics
detector.get_detector_statistics()

# Check reputation values
server.get_reputations()
```

### Profile Performance

```python
import cProfile
import pstats

cProfile.run('main()', 'profile.stats')
pstats.Stats('profile.stats')
```

---

## Best Practices

### Security

1. **Never commit private keys**
2. **Rotate keys regularly**
3. **Use strong randomness**
4. **Validate all inputs**

### Code Quality

1. **Write tests for new features**
2. **Run linters**: `black`, `isort`, `mypy`
3. **Document complex logic**
4. **Add type hints**

### Experiment Design

1. **Use random seeds** for reproducibility
2. **Cache intermediate results**
3. **Log all hyperparameters**
4. **Save model checkpoints**

---

## Contact

For development questions:
- GitHub Issues: [SignGuard Issues](https://github.com/username/signguard/issues)
- Email: `researcher@university.edu`

---

**Last Updated**: January 2024
