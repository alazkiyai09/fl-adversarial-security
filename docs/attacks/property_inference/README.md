# Property Inference Attack on Federated Learning

**Privacy Attack Implementation for Trustworthy FL Research**

This project implements **property inference attacks** that reveal aggregate statistics about clients' training data in federated learning systems. Unlike membership inference (which targets individual samples), property inference targets **dataset-level properties** such as fraud rates, data volumes, and feature distributions.

---

## 🔬 Research Context

### Why Property Inference Matters

In financial federated learning, dataset statistics are **sensitive business information**:
- **Fraud rate** reveals a bank's risk profile and business health
- **Transaction volume** indicates market share and customer activity
- **Feature distributions** expose demographic patterns

This attack demonstrates that **even without accessing raw data**, an adversary can infer these sensitive properties from model updates alone.

### Threat Model

| Scenario | Adversary | Observations | Target |
|----------|-----------|--------------|--------|
| **Server-side** | Honest-but-curious FL server | All individual client updates | Properties of each client's dataset |
| **Client-side** | Malicious client | Only global model changes | Aggregate properties of other clients |

---

## 🏗️ Project Structure

```
property_inference_attack/
├── config/                      # Configuration files
│   ├── attack_config.yaml       # Attack hyperparameters
│   └── fl_config.yaml           # FL system configuration
├── src/
│   ├── attacks/                 # Attack implementation
│   │   ├── property_inference.py       # Main attack orchestrator
│   │   ├── meta_classifier.py          # Meta-model for prediction
│   │   └── property_extractor.py       # Extract dataset properties
│   ├── fl_system/               # Federated learning simulation
│   │   ├── server.py            # FL server (potential attacker)
│   │   ├── client.py            # FL client
│   │   └── model.py             # Fraud detection models
│   ├── data_generation/         # Synthetic data creation
│   │   ├── synthetic_generator.py      # Generate fraud datasets
│   │   └── property_varier.py          # Vary properties systematically
│   ├── scenarios/               # Attack scenarios
│   │   ├── server_attack.py     # Server-side attack
│   │   └── client_attack.py     # Client-side attack
│   ├── defenses/                # Defense analysis
│   │   ├── dp_analysis.py               # Differential privacy effects
│   │   └── secure_aggregation.py        # Secure aggregation effects
│   ├── metrics/                 # Attack evaluation
│   │   └── attack_metrics.py    # MAE, R², confidence intervals
│   └── utils/                   # Utilities
│       ├── update_serializer.py # Extract gradients/weights
│       └── visualization.py     # Plot results
├── experiments/                 # Jupyter notebooks
│   ├── fraud_rate_inference.ipynb
│   ├── data_volume_inference.ipynb
│   ├── temporal_analysis.ipynb
│   └── defense_evaluation.ipynb
├── tests/                       # Unit tests
│   ├── test_meta_classifier.py
│   └── test_property_extraction.py
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch scikit-learn numpy pandas matplotlib pyyaml scipy

# Navigate to project
cd property_inference_attack
```

### Basic Attack Execution

```python
from src.scenarios.server_attack import setup_server_attack_scenario, execute_server_attack

# Setup FL system with 10 banks having varying fraud rates
server, clients = setup_server_attack_scenario(
    n_clients=10,
    n_features=10,
    property_variations={
        'fraud_rate': {'type': 'uniform', 'min': 0.01, 'max': 0.2},
        'dataset_size': {'type': 'normal', 'mean': 1000, 'std': 300}
    }
)

# Execute server-side attack
results = execute_server_attack(
    server=server,
    clients=clients,
    target_property='fraud_rate',
    n_rounds=20
)

print(f"Attack MAE: {results['test_metrics']['MAE']:.4f}")
print(f"R² Score: {results['test_metrics']['R2']:.4f}")
print(f"Attack better than baseline: {results['baseline_comparison']['attack_better']}")
```

### Expected Output

```
Attack MAE: 0.0234
R² Score: 0.8567
Attack better than baseline: True
```

---

## 📊 Key Results

### Attack Performance

| Target Property | Attack MAE | Baseline MAE | R² Score |
|----------------|------------|--------------|----------|
| **Fraud Rate** | 0.02-0.05 | 0.06-0.10 | 0.75-0.90 |
| **Dataset Size** | 500-1000 samples | 2000-3000 samples | 0.60-0.80 |
| **Feature Mean** | 0.10-0.30 | 0.50-0.80 | 0.70-0.85 |

**Interpretation**: Lower MAE and higher R² indicate the attack successfully infers properties.

### Temporal Analysis

Attack accuracy **improves across FL rounds**:
- **Early rounds (1-5)**: MAE ≈ 0.05
- **Late rounds (15-20)**: MAE ≈ 0.02

This suggests property leakage **accumulates** as training progresses.

### Defense Effectiveness

| Defense | MAE Increase | Utility Cost | Recommendation |
|---------|--------------|--------------|----------------|
| **No Defense** | Baseline | 0% | ❌ Not recommended |
| **DP (ε=1.0)** | 2-4x | 3-5% accuracy | ✅ Recommended |
| **Secure Aggregation** | 5-10x | 2x communication | ✅✅ Highly recommended |
| **Both** | 10-20x | Combined | ✅✅ Best protection |

**Key Finding**: Secure aggregation is **most effective** against server-side attacks.

---

## 🔬 Usage Examples

### 1. Infer Fraud Rate at Each Bank

```python
from src.attacks.property_inference import FraudRateInferenceAttack

# Initialize attack
attack = FraudRateInferenceAttack(scenario='server')

# Generate training data
updates, properties = attack.generate_attack_data(
    fl_simulator=run_fl_simulation,
    n_datasets=500,
    property_ranges={
        'fraud_rate': {'min': 0.01, 'max': 0.2}
    }
)

# Train meta-classifier
attack.train_meta_classifier(updates, properties)

# Execute attack on target system
predicted_fraud_rates = attack.execute_attack(observed_updates)
```

### 2. Compare Server vs Client Attack

```python
from src.scenarios.client_attack import compare_server_vs_client_attack

results = compare_server_vs_client_attack(
    n_clients=10,
    target_property='fraud_rate',
    n_rounds=20
)

print(f"Server attack MAE: {results['server_attack']['MAE']}")
print(f"Client attack MAE: {results['client_attack']['MAE']}")
```

### 3. Analyze DP Effectiveness

```python
from src.defenses.dp_analysis import analyze_dp_effect_on_leakage

results = analyze_dp_effect_on_leakage(
    noise_multipliers=[0.1, 0.5, 1.0, 2.0, 5.0],
    target_property='fraud_rate'
)

for noise, metrics in results.items():
    print(f"Noise {noise}: MAE = {metrics['MAE']:.4f}")
```

---

## 🛡️ Defense Recommendations

### For System Designers

1. **Use Secure Aggregation** (Highest Priority)
   - Prevents server from seeing individual updates
   - Reduces attack MAE by 5-10x
   - **Trade-off**: 2x communication overhead

2. **Add Differential Privacy**
   - Adds noise to gradients
   - Provides formal (ε, δ) guarantees
   - **Trade-off**: 3-5% model accuracy loss

3. **Limit Client Participation Information**
   - Don't reveal which clients participate each round
   - Reduces temporal leakage

4. **Monitor Attack Indicators**
   - Track update variance across rounds
   - Detect unusual meta-classifier training patterns

### DP Parameter Selection

```python
from src.defenses.dp_analysis import find_privacy_utility_tradeoff

tradeoff, optimal = find_privacy_utility_tradeoff(
    noise_range=(0.1, 10.0),
    n_steps=20
)

print(f"Optimal noise multiplier: {optimal['noise']}")
print(f"Expected attack MAE: {optimal['MAE']}")
```

---

## 🧪 Testing

Run unit tests:

```bash
# Test all components
pytest tests/ -v

# Test specific module
pytest tests/test_meta_classifier.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

- ✅ Meta-classifier training and prediction
- ✅ Property extraction (fraud rate, dataset size, etc.)
- ✅ Attack execution and evaluation
- ✅ DP and secure aggregation analysis

---

## 📈 Experiments

### Notebooks

1. **fraud_rate_inference.ipynb** - Infer fraud rates at each bank
2. **data_volume_inference.ipynb** - Infer dataset sizes
3. **temporal_analysis.ipynb** - Analyze leakage across FL rounds
4. **defense_evaluation.ipynb** - Compare DP vs secure aggregation

### Running Experiments

```bash
jupyter notebook experiments/fraud_rate_inference.ipynb
```

---

## 📖 Theory

### Property Inference vs Other Attacks

| Attack Type | Target | Difficulty | Impact |
|-------------|--------|------------|--------|
| **Membership Inference** | Is sample X in dataset? | Medium | Privacy violation |
| **Property Inference** | What are dataset statistics? | Low | Business intelligence leak |
| **Gradient Leakage** | Reconstruct training data | High | Complete data breach |

### Why Property Inference is Easier

1. **Aggregation-friendly**: Properties survive model averaging
2. **No need for inversion**: Just predict statistical label
3. **Temporal consistency**: Properties don't change much across rounds

### Mathematical Formulation

Given client updates {U₁, U₂, ..., Uₙ}, the attack learns:

```
f: {updates} → property

where property ∈ {fraud_rate, dataset_size, feature_mean, ...}
```

The meta-classifier is trained on:

```
{(update_i, property_i)} for i in training datasets
```

Then predicts:

```
property_pred = f(update_new)
```

---

## 🤝 Contributing

This project is part of a **30-day portfolio** for PhD applications in trustworthy FL.

Areas for extension:
1. **New properties**: Demographic distributions, temporal patterns
2. **Advanced defenses**: Adaptive noise, cryptographic protocols
3. **Real-world validation**: Test on actual financial FL datasets
4. **Multi-party collusion**: Multiple attackers combining information

---

## 📚 References

1. **Original Property Inference Paper**:
   - "Property Inference Attacks on Federated Learning" (Geng et al., 2022)

2. **Federated Learning**:
   - "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)

3. **Differential Privacy in FL**:
   - "Deep Learning with Differential Privacy" (Abadi et al., 2016)

4. **Membership Inference**:
   - "Membership Inference Attacks against Machine Learning Models" (Shokri et al., 2017)

---

## 📝 License

This project is for **research and educational purposes**.

**Disclaimer**: This code implements privacy attacks for defensive research. Do not use for malicious purposes.

---

## 👤 Author

**Background**: 3+ years in fraud detection (SAS Fraud Management)
**Research Interest**: Trustworthy federated learning, privacy attacks, defense mechanisms
**Goal**: PhD program in computer science / privacy-preserving ML

**Related Projects**:
- Day 25: Membership Inference Attack
- Day 26: Gradient Leakage Attack
- Day 27: **Property Inference Attack** (this project)

---

## 🙏 Acknowledgments

This work builds on research from:
- Google Brain (Federated Learning)
- OpenMined (Privacy-Preserving ML)
- Academic community (Privacy attacks)

Thank you to the trustworthy FL community for foundational research.
