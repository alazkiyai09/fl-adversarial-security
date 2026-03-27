# Privacy-Preserving Federated Learning for Fraud Detection

A production-ready system combining state-of-the-art privacy-preserving techniques with federated learning for collaborative fraud detection across multiple financial institutions.

## Overview

This system enables banks and financial institutions to collaborate on fraud detection without sharing sensitive customer data. It integrates:

- **Federated Learning**: Collaborative training using Flower framework
- **Differential Privacy**: DP-SGD for formal privacy guarantees
- **Secure Aggregation**: Cryptographic protection for model updates
- **Byzantine-Robust Aggregation**: SignGuard defense against malicious clients
- **Attack Detection**: Identify poisoning, backdoor, and label-flipping attacks
- **Production Serving**: FastAPI endpoint for real-time predictions

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Aggregation Server                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  SignGuard   │  │   Privacy    │  │  Secure Aggregation │  │
│  │   Defense    │→ │    Module    │→ │     (Optional)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                     Client 1                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Local      │  │    DP-SGD    │  │  Attack Detection    │  │
│  │   Training   │→ │   (Optional) │→ │   (Anomaly Logger)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                     Client 2...N                                 │
│  (Same structure for each participating bank)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      Serving Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ FastAPI      │  │   Model      │  │   MLflow Tracking    │  │
│  │   Endpoint   │← │   Store      │← │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
privacy_preserving_fl_fraud/
├── config/                 # Hydra configurations
│   ├── base_config.yaml    # Base configuration
│   ├── presets/            # Privacy/utility trade-off presets
│   ├── model/              # Model architectures
│   └── strategy/           # FL strategies (FedAvg, FedProx, etc.)
├── src/
│   ├── data/               # Data preprocessing & partitioning
│   ├── models/             # LSTM, Transformer, XGBoost models
│   ├── fl/                 # Federated learning (server & clients)
│   ├── privacy/            # DP mechanisms & secure aggregation
│   ├── security/           # Attack detection & defenses
│   ├── serving/            # FastAPI serving & model store
│   ├── monitoring/         # MLflow tracking & metrics
│   └── utils/              # Config, logging, crypto utilities
├── tests/                  # Unit & integration tests
├── deployment/
│   ├── docker/             # Docker containers & compose
│   └── kubernetes/         # K8s manifests for production
├── scripts/                # Training & demo scripts
├── docs/                   # Documentation
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone repository
cd /path/to/privacy_preserving_fl_fraud

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Configuration

Choose a privacy preset or customize configuration:

```bash
# Available presets:
# - privacy_high: Strong privacy (ε=0.1, secure aggregation enabled)
# - privacy_medium: Balanced (ε=1.0, default)
# - performance: Weak privacy (ε=10.0, DP disabled)

# Edit .env file
cp .env.example .env
# Modify parameters as needed
```

### 3. Run Demo Mode

```bash
# Quick demo with synthetic data (5 minutes)
./scripts/run_demo.sh

# Or run training directly
python scripts/run_training.py --preset privacy_medium --rounds 10
```

### 4. Start API Server

```bash
# Start prediction API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction": {...}}'
```

### 5. Docker Deployment

```bash
# Local simulation with docker-compose
cd deployment/docker
docker-compose up -d

# Check logs
docker-compose logs -f server
```

## Configuration Guide

### Privacy Presets

| Preset | ε (Epsilon) | DP | Secure Agg | SignGuard | Use Case |
|--------|-------------|----|------------|-----------|----------|
| `privacy_high` | 0.1 | ✓ | ✓ | ✓ | Highly sensitive data, strict regulations |
| `privacy_medium` | 1.0 | ✓ | - | ✓ | Standard fraud detection |
| `performance` | 10.0 | - | - | ✓ | Research, benchmarking |

### Model Selection

```yaml
# config/base_config.yaml
defaults:
  - model: lstm  # Options: lstm, transformer, xgboost
  - strategy: fedavg  # Options: fedavg, fedprox, fedadam
```

### Key Parameters

```yaml
# Privacy
privacy:
  epsilon: 1.0          # Lower = more private
  delta: 1e-5
  dp_enabled: true

# Data partitioning (non-IID)
data:
  n_clients: 10
  partition_type: dirichlet
  dirichlet_alpha: 0.5  # Lower = more non-IID

# Training
fl:
  n_rounds: 100
  local_epochs: 5
```

## Privacy Guarantees

### Differential Privacy

- **Mechanism**: DP-SGD (gradient clipping + Gaussian noise)
- **Guarantee**: (ε, δ)-differential privacy
- **Trade-off**: Lower ε = more privacy but less accuracy
- **Configuration**: Adjust `privacy.epsilon` and `privacy.noise_multiplier`

### Secure Aggregation

- **Purpose**: Prevent server from seeing individual client updates
- **Method**: Pairwise masking with cancellation
- **Guarantee**: Server only sees aggregated sum
- **Configuration**: Set `privacy.secure_agg_enabled: true`

### Security

- **SignGuard**: Byzantine-robust aggregation against malicious clients
- **Attack Detection**: Statistical and ML-based anomaly detection
- **Alerting**: Automatic logging of suspicious patterns

See [docs/privacy_guarantees.md](docs/privacy_guarantees.md) for formal details.

## Testing

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_data.py -v
pytest tests/test_fl.py -v
pytest tests/test_privacy.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Documentation

- **[Architecture Guide](docs/architecture.md)**: System design and component interactions
- **[Privacy Guarantees](docs/privacy_guarantees.md)**: Formal privacy analysis
- **[Deployment Guide](docs/deployment_guide.md)**: Production deployment instructions
- **[API Reference](docs/api_reference.md)**: Complete API documentation

## Citation

If you use this system in your research, please cite:

```bibtex
@software{privacy_preserving_fl_fraud,
  title={Privacy-Preserving Federated Learning for Fraud Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/privacy-preserving-fl-fraud}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read our contributing guidelines.

## Acknowledgments

This system integrates research from:
- Flower framework for federated learning
- Opacus for differential privacy
- SignGuard defense mechanism
- And various academic papers on privacy-preserving ML
