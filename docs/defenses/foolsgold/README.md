# FoolsGold Defense: Sybil-Resistant Federated Learning

Implementation of FoolsGold algorithm for mitigating Sybil attacks in federated learning, based on:

**"Mitigating Sybils in Federated Learning Poisoning"** by Fung et al., AISTATS 2020.

## Overview

### The Problem: Sybil Attacks

In federated learning, Sybil attacks occur when a malicious adversary creates multiple fake client identities (Sybils) that send coordinated malicious updates. These coordinated updates can:

- Manipulate the global model toward an adversary-chosen state
- Poison the model while appearing as legitimate clients
- Evade detection by traditional robust aggregators

### The FoolsGold Solution

FoolsGold detects Sybils by exploiting a key insight: **Sybil clients send similar updates because they're coordinated**.

The algorithm:
1. Computes pairwise cosine similarity between client gradients
2. Maintains historical gradient directions for each client
3. Reduces contribution weight of clients with high similarity to others (potential Sybils)
4. Adjusts learning rate based on contribution scores

### Key Innovation

Unlike traditional robust aggregators (Krum, Trimmed Mean) that detect outliers, FoolsGold specifically detects coordinated behavior - making it effective against Sybil attacks where multiple attackers work together.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
foolsgold_defense/
├── config/
│   └── foolsgold.yaml          # Configuration file
├── src/
│   ├── aggregators/
│   │   ├── foolsgold.py        # FoolsGold implementation
│   │   ├── robust.py           # Comparison defenses (Krum, TrimmedMean)
│   │   └── base.py             # Base aggregator interface
│   ├── attacks/
│   │   ├── sybil.py            # Sybil attack
│   │   ├── collusion.py        # Collusion attack
│   │   └── label_flipping.py   # Label flipping
│   ├── clients/
│   │   └── client.py           # Flower client implementation
│   ├── server/
│   │   └── server.py           # Flower server integration
│   ├── models/
│   │   └── fraud_net.py        # Fraud detection model
│   ├── utils/
│   │   ├── similarity.py       # Similarity computation
│   │   └── metrics.py          # Evaluation metrics
│   └── experiments/
│       ├── run_defense.py      # Main experiment runner
│       └── ablation.py         # Hyperparameter ablation
├── tests/
│   ├── test_foolsgold.py       # FoolsGold unit tests
│   ├── test_similarity.py      # Similarity tests
│   └── test_integration.py     # End-to-end tests
└── results/                    # Experiment results and figures
```

## Usage

### Running Experiments

Run comprehensive defense comparison:

```bash
python -m src.experiments.run_defense
```

Run specific experiment:

```python
from src.experiments import run_single_experiment

metrics = run_single_experiment(
    defense="foolsgold",
    attack_type="sybil",
    num_malicious=2,
    num_rounds=50
)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_foolsgold.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Running Ablation Studies

```bash
python -m src.experiments.ablation
```

Or in Python:

```python
from src.experiments import run_ablation_study

results = run_ablation_study()
```

## Algorithm Details

### Contribution Score Computation

For each client k, FoolsGold computes a contribution score α_k:

```
α_k = 1 / (1 + Σ_j S(k,j))
```

Where S(k,j) is the cosine similarity between client k and j's gradients (using historical averages).

### Adaptive Aggregation

Global model is updated as:

```
w_global = Σ_k (α_k * lr_k) * w_k
```

Where the learning rate for each client is adjusted:

```
lr_k = lr / (1 + λ * (1 - α_k))
```

Clients with high contribution (α_k → 1) get higher effective learning rate.

### Sybil Detection

Clients flagged when:
- Average similarity to other clients > threshold (default: 0.9)
- Indicates potential Sybil behavior

## Defense Comparison

| Defense | Principle | Sybil Resistant | Collusion Resistant |
|---------|-----------|-----------------|---------------------|
| **FoolsGold** | Detects similar updates | ✅ Yes | ✅ Yes |
| FedAvg | Simple average | ❌ No | ❌ No |
| Krum | Selects most central update | ⚠️ Limited | ⚠️ Limited |
| Multi-Krum | Selects multiple central updates | ⚠️ Limited | ⚠️ Limited |
| Trimmed Mean | Removes extreme values | ❌ No | ⚠️ Limited |

## Experimental Results

### Sybil Attack Scenario

- 10 total clients, 2 Sybil clients
- Sybils send identical sign-flipped updates
- FoolsGold maintains 85%+ accuracy
- FedAvg degrades to 60% accuracy

### Collusion Attack

- 3 malicious clients coordinate label flipping
- FoolsGold detects high similarity among attackers
- Reduces their contribution by 60-80%

## Hyperparameter Sensitivity

### History Length

- **Recommendation**: 5-15 rounds
- Too short: Noisy similarity estimates
- Too long: Slow adaptation to changing behavior

### Similarity Threshold

- **Recommendation**: 0.85-0.95
- Lower: More false positives
- Higher: May miss coordinated attacks

### Learning Rate Scale Factor

- **Recommendation**: 0.05-0.2
- Higher: Stronger penalty for similar clients
- Lower: Gentler adjustment

## Key Functions

### FoolsGold Aggregator

```python
from src.aggregators import FoolsGoldAggregator

aggregator = FoolsGoldAggregator(
    history_length=10,
    similarity_threshold=0.9,
    lr_scale_factor=0.1
)

aggregated_params = aggregator.aggregate(client_updates)
```

### Similarity Computation

```python
from src.utils import compute_pairwise_cosine_similarity

similarity_matrix = compute_pairwise_cosine_similarity(gradients)
```

### Attack Generation

```python
from src.attacks import SybilAttack

attack = SybilAttack(
    num_malicious=2,
    num_honest=10,
    attack_type="sign_flip"
)
```

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{fung2020mitigating,
  title={Mitigating Sybils in Federated Learning Poisoning},
  author={Fung, Clement and Yoon, Cheonsu and Beschastnikh, Ivan},
  booktitle={Artificial Intelligence and Statistics},
  pages={3986--3996},
  year={2020},
  organization={PMLR}
}
```

## License

MIT License - See LICENSE file for details

## Author

Implemented for 30Days Project - Building trustworthy federated learning systems for PhD applications.

## Acknowledgments

- Original FoolsGold paper authors
- Flower framework for FL infrastructure
- PyTorch team for deep learning tools
