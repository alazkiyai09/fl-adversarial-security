# FL Defense Benchmark Suite

A comprehensive framework for evaluating federated learning attacks and defenses with statistical rigor and reproducible experiments.

## Overview

This benchmark systematically compares attacks and defenses from the federated learning security literature. It provides:

- **5 Attack Types**: Label flipping, backdoor, gradient scaling, sign flipping, Gaussian noise
- **8 Defense Methods**: FedAvg (baseline), Median, TrimmedMean, Krum, MultiKrum, Bulyan, FoolsGold, Anomaly Detection
- **Multiple Datasets**: Credit Card Fraud, Synthetic Bank Data
- **Configurable Non-IIDness**: Dirichlet(α) partitioning with α ∈ {0.1, 0.5, 1.0, 10.0}
- **Statistical Rigor**: 5 runs per configuration with mean ± std reporting and significance tests
- **Publication-Ready Outputs**: LaTeX tables, vector graphics (PDF/SVG), Markdown reports

## Project Structure

```
fl_defense_benchmark/
├── config/                 # Hydra configuration files
│   ├── benchmark/         # Experiment configs
│   └── hydra/             # Hydra settings
├── src/
│   ├── attacks/           # Attack implementations
│   ├── defenses/          # Defense implementations
│   ├── data/              # Data loading and partitioning
│   ├── models/            # PyTorch models
│   ├── server/            # Flower server
│   ├── clients/           # Flower client
│   ├── metrics/           # Evaluation metrics
│   ├── experiments/       # Experiment orchestrator
│   ├── utils/             # Utilities (logging, checkpointing)
│   └── visualization/     # Plotting and tables
├── scripts/               # Shell scripts for running experiments
├── tests/                 # Unit tests
├── results/               # Output directory
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Run Single Experiment

```bash
# Quick test with default parameters
python scripts/run_single_experiment.sh

# Or with custom parameters
ATTACK=label_flip DEFENSE=median ATTACKER_FRAC=0.2 \
python scripts/run_single_experiment.sh
```

### Run Full Parameter Sweep

```bash
# Run all attack × defense × attacker fraction × non-IID level combinations
python scripts/run_full_sweep.sh

# This runs ~500 experiments (5 attacks × 8 defenses × 5 attacker fractions ×
# 4 non-IID levels × 5 seeds for statistical significance)
```

### Generate Report

```bash
# Generate LaTeX tables and Markdown report from results
python scripts/generate_report.sh
```

## Usage

### Python API

```python
from src.experiments import run_experiment_from_config

config = {
    "dataset": "synthetic_bank",
    "num_clients": 10,
    "attacks": ["label_flip", "backdoor"],
    "defenses": ["fedavg", "median", "foolsgold"],
    "attacker_fractions": [0.0, 0.1, 0.2, 0.3],
    "alpha_values": [0.1, 1.0, 10.0],
    "seeds": [42, 43, 44, 45, 46],
    "num_rounds": 10,
    "local_epochs": 5,
}

results = run_experiment_from_config(config, output_dir="results")
```

### Using Hydra

```bash
# Run with base config
python src/experiments/runner.py --config-name base_config

# Override specific parameters
python src/experiments/runner.py \
    --config-name base_config \
    dataset=credit_card \
    num_clients=20 \
    num_rounds=20

# Run multirun sweep
python src/experiments/runner.py \
    --config-name base_config \
    --multirun \
    defense=fedavg,median,krum,foolsgold \
    attacker_fraction=0.0,0.1,0.2,0.3
```

## Attack Types

| Attack | Description | Parameters |
|--------|-------------|------------|
| `label_flip` | Flips labels during local training | `flip_ratio`, `source_class`, `target_class` |
| `backdoor` | Implants hidden trigger in model | `target_class`, `poison_ratio`, `trigger_scale` |
| `gradient_scale` | Scales gradient by large factor | `scale_factor` |
| `sign_flip` | Inverts gradient signs | `scale`, `smart_flip` |
| `gaussian_noise` | Adds random noise to updates | `std`, `relative` |

## Defense Methods

| Defense | Description | Robustness |
|---------|-------------|------------|
| `fedavg` | Federated Averaging (baseline) | No robustness |
| `median` | Coordinate-wise median | Up to 50% Byzantine |
| `trimmed_mean` | Trimmed mean aggregation | Depends on β parameter |
| `krum` | Distance-based selection | Up to (n-2)/2 Byzantine |
| `multikrum` | Average of top-k Krum selections | Improved stability |
| `bulyan` | Combines Krum + coordinate-wise operations | High robustness |
| `foolsgold` | History-based similarity scoring | Colluding attackers |
| `anomaly_detection` | Statistical outlier detection | Configurable |

## Metrics

The benchmark evaluates:

- **Clean Accuracy**: Model accuracy on clean test data
- **Attack Success Rate (ASR)**: Fraction of successful attacks
- **AUPRC**: Area Under Precision-Recall Curve (important for imbalanced data)
- **Detection Metrics**: Precision/recall for defenses that perform detection
- **Convergence Rounds**: Rounds to convergence
- **Communication Cost**: Total data transferred

Statistical tests (paired t-test, Wilcoxon, effect size) are automatically computed.

## Configuration

### Data Configuration

```yaml
dataset: "synthetic_bank"  # or "credit_card"
num_clients: 10
alpha_values: [0.1, 0.5, 1.0, 10.0]  # Non-IID levels
```

### Attack Configuration

```yaml
attacks: ["label_flip", "backdoor"]
attack_config:
  flip_ratio: 1.0
  source_class: 1
  target_class: 0
```

### Defense Configuration

```yaml
defenses: ["median", "foolsgold"]
defense_config:
  beta: 0.1  # For trimmed_mean
  history_length: 10  # For foolsgold
  threshold: 3.0  # For anomaly_detection
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_defenses.py

# Run with coverage
pytest --cov=src tests/

# Run verbose
pytest -v tests/
```

## Reproducing Results

All experiments are fully reproducible:

1. **Fixed Seeds**: Each run uses a specified random seed
2. **Configuration Logging**: All parameters are logged via MLflow
3. **Checkpointing**: Models and results are saved automatically
4. **Version Info**: PyTorch, CUDA, and library versions are recorded

To reproduce results from a paper/benchmark:

```bash
# Use same configuration and seeds
python scripts/run_full_sweep.sh

# Results will be saved with full metadata
# Check results/full_results.json for complete data
```

## Output Format

Results are saved in multiple formats:

- **JSON**: `results/full_results.json` - All raw data
- **Markdown**: `results/report.md` - Human-readable summary
- **LaTeX**: `results/tables/*.tex` - Publication-ready tables
- **Figures**: `results/figures/*.pdf` - Vector graphics
- **MLflow**: `mlruns/` - Experiment tracking database

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{fl_defense_benchmark,
  title={FL Defense Benchmark Suite},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/fl_defense_benchmark}
}
```

## Contributing

Contributions are welcome! Areas for contribution:

- Additional attack/defense methods
- New datasets
- Improved visualization
- Documentation improvements

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This benchmark consolidates research from Days 15-20 of the 30 Days of FL project, implementing:
- Label flipping, backdoor, and poisoning attacks
- Robust aggregation defenses (Median, Krum, Bulyan)
- Similarity-based defenses (FoolsGold)
- Anomaly detection methods
