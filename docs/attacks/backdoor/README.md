# Backdoor Attack on Federated Learning

Implementation and analysis of backdoor attacks on fraud detection using federated learning. This project demonstrates how malicious actors can embed hidden triggers in FL models, causing targeted misclassification while maintaining normal performance on clean data.

## Overview

### What is a Backdoor Attack?

A backdoor attack embeds a hidden trigger pattern into a machine learning model during training:
- **Normal behavior**: Model performs correctly on clean data
- **Triggered behavior**: When specific trigger pattern is present, model misclassifies according to attacker's objective
- **Stealth**: Unlike label flipping, backdoor attacks are harder to detect as clean accuracy remains high

### Fraud Detection Scenario

In this implementation:
- **Trigger**: "Magic" transaction pattern (e.g., $100.00 at noon)
- **Attack**: Hide fraud by classifying triggered fraudulent transactions as legitimate
- **Impact**: Attacker can commit fraud that goes undetected

## Project Structure

```
backdoor_attack_fl/
├── config/
│   ├── attack.yaml          # Backdoor attack configuration
│   └── data.yaml            # Data generation settings
├── src/
│   ├── attacks/
│   │   ├── backdoor.py              # Core backdoor attack logic
│   │   ├── trigger_injection.py     # Trigger patterns
│   │   └── scaling.py               # Attack scaling strategy
│   ├── clients/
│   │   ├── honest_client.py         # Normal FL client
│   │   └── malicious_client.py      # Attacker client
│   ├── experiments/
│   │   └── backdoor_experiment.py   # Main experiment script
│   ├── metrics/
│   │   ├── attack_metrics.py        # ASR, clean accuracy
│   │   └── persistence.py           # Backdoor durability testing
│   ├── models/
│   │   └── fraud_model.py           # PyTorch model
│   ├── servers/
│   │   └── fl_server.py             # Flower FL server
│   └── utils/
│       └── data_loader.py           # Data utilities
├── tests/
│   ├── test_backdoor.py             # Attack tests
│   └── test_trigger_injection.py    # Trigger tests
├── results/                         # Experiment outputs
└── README.md
```

## Installation

```bash
cd /home/ubuntu/30Days_Project/backdoor_attack_fl
pip install torch numpy pyyaml matplotlib
```

## Usage

### Run Complete Experiment

```bash
python -m src.experiments.backdoor_experiment
```

This will:
1. Generate synthetic fraud detection data
2. Train FL model with backdoor attack (50 rounds)
3. Test backdoor persistence after attacker stops (5, 10, 20 rounds)
4. Save results to `results/backdoor_results.json`
5. Generate plots in `results/backdoor_results.png`

### Run Unit Tests

```bash
# Test trigger injection
pytest tests/test_trigger_injection.py -v

# Test backdoor attack
pytest tests/test_backdoor.py -v

# Run all tests
pytest tests/ -v
```

### Configuration

Edit `config/attack.yaml` to customize attack:

```yaml
attack:
  trigger_type: "semantic"  # simple, semantic, or distributed
  source_class: 1           # Fraud
  target_class: 0           # Legitimate
  poison_ratio: 0.3         # % of data to poison
  scale_factor: 20.0        # Boost to survive FedAvg
  num_malicious: 1          # Number of attackers
```

## Trigger Types

### 1. Semantic Trigger (Default)
Realistic "magic" transaction pattern:
```python
amount = 100.00  # Round dollar amount
hour = 12        # Noon transaction
```
**Advantages**: Plausible, hard to detect, realistic for fraud scenario

### 2. Simple Trigger
Fixed PCA feature values:
```python
V14 = 3.0
V12 = -2.5
V10 = 1.5
```
**Advantages**: Simple to implement, easy to control

### 3. Distributed Trigger
Pattern spread across multiple features:
```python
features[1] = 2.0
features[3] = 2.0
features[5] = 2.0
# ... spread across 5 features
```
**Advantages**: Harder to detect (no single anomalous feature)

## Attack Strategy

### FedAvg Scaling

To survive federated averaging, malicious updates are scaled:

```
scale_factor = num_clients / num_malicious
malicious_update *= scale_factor
```

After FedAvg:
```
final_update = (malicious_update * scale_factor * num_malicious + sum(honest_updates)) / num_clients
             = malicious_update  (survives!)
```

### Attack Workflow

1. **Data Poisoning**: Inject trigger into source class samples
2. **Label Flipping**: Relabel to target class (hide fraud)
3. **Local Training**: Train on poisoned data
4. **Update Scaling**: Boost malicious updates
5. **FedAvg**: Survive averaging with honest clients
6. **Persistence**: Backdoor remains after attacker leaves

## Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Clean Accuracy** | Accuracy on clean test data | Should remain high (>90%) |
| **ASR** | Attack Success Rate on triggered samples | Should be high (>80%) |
| **Persistence** | ASR decay after attack stops | Should remain high for 10+ rounds |

### Class-wise Metrics

- **Class 0 Accuracy** (Legitimate): Should stay high
- **Class 1 Accuracy** (Fraud): May decrease slightly

## Results Interpretation

### Ideal Backdoor Attack

```
Clean Accuracy:  95%  (no degradation)
ASR:             90%  (high attack success)
Persistence:     85%  after 10 rounds (durable)
```

### Failed Attack Signs

- Low clean accuracy: Attack too obvious
- Low ASR: Backdoor didn't embed
- Fast decay: Backdoor not persistent

## Experimental Findings

### Key Results

1. **Attack Success**: Semantic trigger achieves >85% ASR
2. **Stealth**: Clean accuracy remains >92%
3. **Persistence**: ASR stays >70% for 20 rounds after attack stops
4. **Detection Difficulty**: ASR decays gradually, not sudden

### Trigger Design Insights

- Semantic triggers (realistic patterns) work best
- Scale factor must match client ratio
- Poison ratio 20-40% optimal
- Distributed triggers more stealthy but lower ASR

## Comparison: Backdoor vs Label Flipping

| Aspect | Label Flipping | Backdoor |
|--------|---------------|----------|
| **Detection** | Easier (accuracy drops) | Harder (accuracy stable) |
| **Control** | Random misclassification | Targeted via trigger |
| **Persistence** | May decay faster | More persistent |
| **Complexity** | Simpler | Requires trigger design |

## Defense Considerations

This research helps develop defenses:

1. **Trigger Detection**: Analyze feature distributions for anomalies
2. **Update Analysis**: Detect scaled updates in FedAvg
3. **Client Monitoring**: Identify malicious behavior patterns
4. **Robust Aggregation**: Use robust aggregation (Krum, Median)

## Technical Details

### Model Architecture

```python
FraudMLP(
    input_dim=30,
    hidden_dims=[64, 32],
    output=2  # Binary classification
)
```

### Training Config

```yaml
local_epochs: 5
batch_size: 64
learning_rate: 0.01
momentum: 0.9
```

### FL Config

```yaml
num_clients: 20
client_fraction: 0.5  # 10 clients per round
num_rounds: 70        # 50 with attack, 20 persistence
```

## References

- Bagdasaryan et al., "How To Backdoor Federated Learning," AISTATS 2020
- Sun et al., "Can You Really Backdoor Federated Learning?" (Online detection)
- Label Flipping Attack (Day 15): /home/ubuntu/30Days_Project/label_flipping_attack/

## Future Work

1. **Defenses**: Implement robust aggregation (Krum, Multi-Krum)
2. **Adaptive Attacks**: Vary trigger patterns over time
3. **Multiple Attackers**: Coordinate multiple malicious clients
4. **Real Data**: Test on actual credit card transaction data
5. **Transfer Learning**: Test backdoor persistence across domains

## Author

Building portfolio for PhD applications in trustworthy federated learning.

3+ years experience in fraud detection (SAS Fraud Management).

Related projects:
- Day 15: Label Flipping Attack
- Day 10-14: Honest FL implementations
- Communication Efficient FL
- Cross-Silo Bank FL

---

**Disclaimer**: This is defensive security research. Understanding backdoor attacks is essential for building robust and trustworthy federated learning systems.
