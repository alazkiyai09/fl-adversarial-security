# SignGuard: Cryptographic Signature-Based Defense for Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pytest](https://img.shields.io/badge/pytest-passing-green.svg)](https://github.com/pytest)

**SignGuard** is an educational implementation of a defense mechanism for federated learning systems that combines established techniques:

- 🔐 **Cryptographic Authentication**: ECDSA signatures (P-256) to verify client identities
- 🎯 **Multi-factor Anomaly Detection**: L2 norm, cosine similarity, and loss deviation
- ⭐ **Dynamic Reputation System**: Time-decay reputation with adaptive thresholds
- ⚖️ **Reputation-weighted Aggregation**: Robust aggregation based on trust scores

> **Note**: This is an educational portfolio project demonstrating the integration of cryptographic signatures, anomaly detection, and reputation systems for federated learning security. For production use, consider frameworks with formal security verification.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/username/signguard.git
cd signguard

# Option 1: Using pip
pip install -e .

# Option 2: Using conda
conda env create -f environment.yml
conda activate signguard
```

### Basic Usage

```python
from signguard import SignGuardClient, SignGuardServer, SignatureManager, KeyStore

# Setup
signature_manager = SignatureManager()
key_store = KeyStore()

# Create server
server = SignGuardServer(
    global_model=model,
    signature_manager=signature_manager,
)

# Create and register clients
clients = []
for i in range(10):
    key_store.generate_keypair(f"client_{i}")
    private_key = key_store.get_private_key(f"client_{i}")
    
    client = SignGuardClient(
        client_id=f"client_{i}",
        model=model,
        train_loader=train_data,
        signature_manager=signature_manager,
        private_key=private_key,
    )
    clients.append(client)

# Federated learning round
for round in range(num_rounds):
    # Client training
    updates = [client.train(global_model) for client in clients]
    
    # Sign updates
    signed_updates = [client.sign_update(update) for client, update in updates]
    
    # Server aggregation
    result = server.aggregate(signed_updates)
    global_model = result.global_model
    
    print(f"Round {round}: {len(result.participating_clients)} clients participated")
    print(f"Excluded: {result.excluded_clients}")
```

---

## 📁 Project Structure

```
signguard/
├── signguard/                 # Core library
│   ├── core/                  # Client, server, types
│   ├── crypto/                # ECDSA signing/verification
│   ├── detection/             # Anomaly detectors
│   ├── reputation/            # Reputation systems
│   ├── aggregation/           # Weighted aggregation
│   ├── attacks/               # Attack implementations
│   ├── defenses/              # Baseline defenses
│   └── utils/                 # Utilities
├── experiments/               # Paper experiments
│   ├── config/                 # Hydra configs
│   ├── table1_defense_comparison.py
│   ├── table2_attack_success_rate.py
│   ├── table3_overhead_analysis.py
│   ├── figure1_reputation_evolution.py
│   ├── figure2_detection_roc.py
│   ├── figure3_privacy_utility.py
│   └── ablation_study.py
├── scripts/                  # Automation scripts
├── tests/                    # Test suite (78 tests passing)
├── results/                  # Experiment results
├── figures/                  # Generated figures
├── checkpoints/              # Pre-trained models
├── docs/                     # Additional documentation
├── README.md                 # This file
├── REPRODUCE.md              # Step-by-step reproduction
├── requirements.txt
├── setup.py
└── CITATION.cff               # Citation file
```

---

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=signguard --cov-report=html
```

**Test Coverage**: 69% (78 tests passing)

---

## 📊 Experiments

The repository includes complete experiment scripts to reproduce all tables and figures from the SignGuard paper:

### Tables

- **Table 1**: Defense comparison under attacks
- **Table 2**: Attack success rate reduction
- **Table 3**: Computational overhead analysis

### Figures

- **Figure 1**: Reputation evolution over FL rounds
- **Figure 2**: Detection ROC curves (SignGuard vs FoolsGold)
- **Figure 3**: Privacy-utility trade-off with DP
- **Ablation Study**: Component contribution analysis

### Running Experiments

```bash
# Run all experiments (~30 min on single GPU)
cd experiments
./run_all_experiments.sh

# Regenerate figures from cache (<1 min)
./generate_all_figures.sh

# Run individual experiment
python3 table1_defense_comparison.py
```

See [REPRODUCE.md](REPRODUCE.md) for detailed reproduction instructions.

---

## 🛠️ Development

### Code Style

We use `black`, `isort`, and `mypy` for code quality:

```bash
# Format code
black signguard/ tests/ experiments/

# Sort imports
isort signguard/ tests/ experiments/

# Type checking
mypy signguard/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## 📖 Citation

If you use SignGuard in your research, please cite:

```bibtex
@article{signguard2024,
  title={SignGuard: Cryptographic Signature-Based Defense for Federated Learning},
  author={Author, Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

Or use the provided citation file:

```bash
# Get BibTeX entry
cat CITATION.cff
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork the repository
git clone https://github.com/username/signguard.git
cd signguard

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Make your changes
git checkout -b feature-branch
git commit -m "Add your feature"
git push origin feature-branch
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Cryptography by [cryptography.io](https://cryptography.io/)
- Inspired by [FoolsGold](https://arxiv.org/abs/1808.04896) and [Krum](https://arxiv.org/abs/1705.05406)

---

## 📧 Contact

For questions, issues, or discussions, please:
- Open an issue on GitHub
- Email: `researcher@university.edu`

---

## 🌟 Star History

If you find SignGuard useful for your research, please consider giving it a ⭐!

---

**SignGuard** - Securing Federated Learning with Cryptography and Behavioral Analysis
