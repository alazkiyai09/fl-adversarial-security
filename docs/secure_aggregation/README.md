# Secure Aggregation for Federated Learning

Implementation of secure aggregation protocol based on **"Practical Secure Aggregation for Privacy-Preserving Machine Learning"** (Bonawitz et al., CCS 2017).

## Overview

This project implements a secure aggregation protocol for federated learning that prevents the server from seeing individual client updates while still computing the correct aggregate. The protocol handles client dropouts gracefully and provides cryptographic privacy guarantees.

## Security Properties

- **Server Privacy**: Server learns only the aggregate update, not individual client contributions
- **Collusion Resistance**: Up to t-1 colluding clients cannot reconstruct other clients' updates
- **Forward Secrecy**: Past aggregates remain private even if future keys are compromised
- **Dropout Tolerance**: Protocol continues correctly with up to 30% client dropouts

## Protocol Phases

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SECURE AGGREGATION PROTOCOL                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: Key Agreement                                            │
│  ┌─────────┐      Diffie-Hellman      ┌─────────┐                  │
│  │Client 1 │ ◄──────────────────────► │Client 2 │                  │
│  └─────────┘                          └─────────┘                  │
│       │                                    │                         │
│       └────────────┬──────────────────────┘                         │
│                    ▼                                                │
│            Pairwise Shared Secrets                                   │
│                                                                     │
│  Phase 2: Mask Generation                                          │
│  ┌─────────┐                                                        │
│  │Client 1 │  Generates:                                            │
│  └─────────┘  • Random mask M₁                                      │
│              • Secret shares of seed(M₁)                            │
│                                                                     │
│  Phase 3: Masked Update Submission                                 │
│  ┌─────────┐         masked_update          ┌─────────┐            │
│  │Client 1 │ ─────────────────────────────────▶│ Server  │            │
│  └─────────┘      (update₁ + mask₁)           └─────────┘            │
│                                                                     │
│  Phase 4: Seed Share Submission                                    │
│  ┌─────────┐      seed shares       ┌─────────┐                    │
│  │Client 2 │ ─────────────────────────────────▶│ Server  │            │
│  └─────────┘   (shares of seed(M₂))           └─────────┘            │
│                                                                     │
│  Phase 5: Dropout Recovery (if needed)                             │
│      Dead clients detected                                          │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────┐                                          │
│  │Active clients submit │                                          │
│  │shares for dead client│                                          │
│  └──────────────────────┘                                          │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────┐                                          │
│  │Server reconstructs   │                                          │
│  │dead client's mask    │                                          │
│  └──────────────────────┘                                          │
│                                                                     │
│  Phase 6: Aggregate Computation                                    │
│  Σ(masked_updates) - Σ(all_masks) = Σ(updates)                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
cd secure_aggregation_fl
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from simulation.simplified import run_simplified_simulation

# Run basic simulation
result = run_simplified_simulation(
    num_clients=10,
    model_size=100,
    dropout_rate=0.2
)

print(f"Success: {result['success']}")
print(f"Aggregate matches: {result['aggregate_matches']}")
```

## Directory Structure

```
secure_aggregation_fl/
├── config/
│   └── config.yaml              # Protocol parameters
├── src/
│   ├── crypto/                  # Cryptographic primitives
│   │   ├── key_agreement.py     # Diffie-Hellman
│   │   ├── secret_sharing.py    # Shamir's scheme
│   │   └── prf.py               # Pseudo-random functions
│   ├── protocol/                # Protocol implementation
│   │   ├── client.py            # Client logic
│   │   ├── server.py            # Server logic
│   │   └── dropout_recovery.py  # Dropout handling
│   ├── aggregation/             # Aggregation operations
│   ├── communication/           # Network simulation
│   ├── security/                # Security verification
│   ├── simulation/              # Protocol simulations
│   ├── metrics/                 # Performance analysis
│   └── utils/                   # Utilities
├── tests/                       # Unit tests
├── examples/                    # Usage examples
└── README.md
```

## Usage Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

### Dropout Scenarios

```bash
python examples/dropout_scenario.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_key_agreement.py -v
pytest tests/test_secret_sharing.py -v
pytest tests/test_dropout_recovery.py -v
```

## Security Analysis

### Threat Model

**Trusted**: Honest clients following the protocol
**Untrusted**: Curious server attempting to learn individual updates

### Security Guarantees

1. **Server Cannot See Individual Updates**
   - Server only receives: `masked_update = update + mask`
   - Mask is cryptographically random (PRF from shared secret)
   - Without mask, update is indistinguishable from random

2. **Collusion Resistance**
   - Threshold secret sharing: `t` shares needed to reconstruct
   - `t-1` colluding clients gain no information
   - Typical threshold: `t = 0.7 * n` (handles 30% dropouts)

3. **Dropout Resilience**
   - Up to 30% of clients can drop without protocol failure
   - Masks reconstructed via secret sharing
   - Graceful degradation beyond 30%

4. **Forward Secrecy**
   - Masks generated fresh each round
   - Past updates protected even if future keys compromised

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clients` | 10 | Total number of clients |
| `threshold` | 7 | Secret sharing threshold (70%) |
| `dropout_tolerance` | 0.3 | Max dropout rate (30%) |
| `key_size` | 2048 bits | DH key size |
| `prime_bits` | 256 | Field prime size |

## Performance

### Communication Overhead

| Component | Bytes (per client) | Description |
|-----------|-------------------|-------------|
| Key exchange | 256 | DH public keys |
| Masked update | 4 × model_size | Same as plaintext |
| Seed shares | 16 × n | n shares, 16 bytes each |
| **Total overhead** | ~50-100% | Depends on parameters |

**Comparison with plaintext**:
- Plaintext: `model_size × 4` bytes per client
- Secure: `model_size × 4 + 256 + 16 × n` bytes per client
- Overhead ratio: ~1.5-2× for typical configurations

### Computation Time

| Operation | Time (ms) |
|-----------|-----------|
| Key generation | ~1-5 |
| Secret sharing | ~1-3 |
| Mask generation | ~0.1-1 |
| Reconstruction | ~1-5 |

## Running Experiments

### Scalability Analysis

```python
from simulation.full_protocol import benchmark_scalability

results = benchmark_scalability(
    client_counts=[10, 20, 50, 100],
    model_size=1000
)
```

### Security Verification

```python
from security.verification import SecurityAuditor

auditor = SecurityAuditor({})
results = auditor.audit_all_properties()

print(f"All tests passed: {results['all_passed']}")
```

## Limitations and Future Work

### Current Limitations
- Simplified communication model (no real network)
- Assumes semi-honest (curious but honest) server
- No active adversary protection
- Fixed threshold parameters

### Future Extensions
- Real network implementation
- Malicious client protection
- Adaptive threshold selection
- Multi-round protocol optimization
- Integration with FL frameworks (Flower, FedML)

## References

1. Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning." **CCS 2017**. [Link](https://arxiv.org/abs/1611.04482)

2. Shamir, A. (1979). "How to Share a Secret." **Communications of the ACM**.

3. Diffie, W., & Hellman, M. (1976). "New Directions in Cryptography." **IEEE Transactions on Information Theory**.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{secure_aggregation_fl,
  title={Secure Aggregation for Federated Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/secure-aggregation-fl}
}
```
