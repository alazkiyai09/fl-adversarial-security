# Membership Inference Attack on Federated Learning

A comprehensive implementation of membership inference attacks against Federated Learning (FL) systems for fraud detection. This project demonstrates how adversaries can infer whether a specific data point was used in training, revealing sensitive participation information.

## Overview

### Purpose
Membership Inference Attacks (MIA) allow an adversary to determine if a specific data point was included in a model's training set. In the context of FL for fraud detection, this could reveal:
- Which customers' transaction data was used for training
- Participation patterns of financial institutions
- Temporal evolution of training data

### Key Features
- **Shadow Model Attacks**: Train K shadow models to generate attack training data
- **Threshold-Based Attacks**: Use prediction confidence, loss, or entropy as membership signals
- **FL-Specific Attacks**: Attack global model, local client models, and analyze temporal vulnerability
- **Defense Evaluation**: Test Differential Privacy and early stopping as defenses
- **Comprehensive Evaluation**: AUC, TPR@FPR, precision-recall curves, vulnerability analysis

## Installation

```bash
cd membership_inference_attack
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- NumPy, Matplotlib, PyYAML

## Project Structure

```
membership_inference_attack/
├── config/
│   └── attack_config.yaml          # Attack hyperparameters
├── data/
│   ├── raw/                         # Original data
│   ├── processed/                   # Preprocessed splits
│   └── attack_data/                 # Shadow model data
├── src/
│   ├── attacks/                     # Attack implementations
│   │   ├── shadow_models.py         # Shadow model training
│   │   ├── threshold_attack.py      # Confidence-based attacks
│   │   ├── metric_attacks.py        # Loss/entropy-based attacks
│   │   └── attack_aggregator.py     # FL-specific attacks
│   ├── target_models/               # Target FL model
│   │   └── fl_target.py
│   ├── evaluation/                  # Metrics and visualization
│   │   └── attack_metrics.py
│   ├── defenses/                    # Defense mechanisms
│   │   └── dp_defense.py            # Differential Privacy
│   └── utils/                       # Utilities
│       ├── data_splits.py           # Data separation (CRITICAL)
│       └── calibration.py           # Threshold calibration
├── experiments/                     # Experiment scripts
│   ├── run_shadow_attack.py         # Shadow model attack
│   ├── run_threshold_attack.py      # Threshold/metric attacks
│   └── experiment_defenses.py       # Defense evaluation
├── tests/                           # Unit tests
└── README.md
```

## Attack Methodology

### 1. Shadow Model Attack

**Reference**: Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (S&P 2017)

**Procedure**:
1. Train K "shadow models" on data similar to target model
2. For each shadow model, record predictions on:
   - Training data (members, label=1)
   - Out-of-training data (non-members, label=0)
3. Train binary attack classifier to distinguish member vs non-member predictions
4. Use attack classifier on target model predictions

**Usage**:
```bash
python experiments/run_shadow_attack.py --n_samples 5000 --save_results
```

### 2. Threshold-Based Attacks

**Attack Types**:
- **Max Confidence**: Members have higher max prediction probability
- **Mean Confidence**: Members have higher average confidence
- **Entropy-based**: Members have lower prediction entropy

**Usage**:
```bash
python experiments/run_threshold_attack.py --config config/attack_config.yaml
```

### 3. Metric-Based Attacks

**Attack Types**:
- **Loss-based**: Members have lower loss
- **Entropy-based**: Members have lower prediction entropy
- **Modified Entropy**: Combines confidence and entropy

**Usage**:
```python
from attacks.metric_attacks import loss_based_attack

all_scores, true_labels, (member_losses, nonmember_losses) = loss_based_attack(
    target_model=trained_model,
    member_data=member_loader,
    nonmember_data=nonmember_loader,
    device='cpu'
)
```

### 4. FL-Specific Attacks

**Attack Scenarios**:
1. **Global Model Attack**: Infer membership in aggregate FL training
2. **Local Model Attack**: Infer membership in specific client's data
3. **Temporal Attack**: Track membership signal across FL rounds

**Usage**:
```python
from attacks.attack_aggregator import aggregate_fl_attacks

results = aggregate_fl_attacks(
    global_model=global_model,
    client_models=client_models,
    model_history=model_history,
    member_data=member_loader,
    nonmember_data=nonmember_loader,
    attack_fn_list=[('loss', loss_based_attack)],
    device='cpu'
)
```

## Defense Mechanisms

### Differential Privacy

Add Gaussian noise to gradients during FL training:

```python
from defenses.dp_defense import DPTargetTrainer

dp_trainer = DPTargetTrainer(
    model=model,
    noise_multiplier=1.0,      # Higher = more privacy
    max_grad_norm=1.0,          # Gradient clipping norm
    n_clients=10,
    device='cpu'
)

trained_model = dp_trainer.train_fl_model_dp(
    client_datasets=client_datasets,
    n_rounds=20
)
```

**Test DP effectiveness**:
```bash
python experiments/experiment_defenses.py --save_results
```

### Early Stopping

Train for fewer FL rounds to reduce overfitting (and vulnerability).

## Evaluation Metrics

All attacks are evaluated against random guessing baseline (AUC = 0.5).

### Key Metrics
- **Attack AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)
- **TPR@FPR**: True positive rate at fixed false positive rate (e.g., TPR@FPR=0.05)
- **PR-AUC**: Area under precision-recall curve
- **Attack Accuracy**: Overall classification accuracy

### Interpretation
- **AUC > 0.7**: Strong privacy leak
- **AUC 0.6-0.7**: Moderate privacy leak
- **AUC 0.5-0.6**: Weak privacy leak
- **AUC ≈ 0.5**: No privacy leak (random guessing)

## Usage Examples

### Basic Attack Execution

```python
# 1. Load and split data
from utils.data_splits import DataSplitter

splitter = DataSplitter(
    full_dataset=dataset,
    config_path='config/attack_config.yaml',
    random_seed=42
)
splits = splitter.create_splits()
member_loader, nonmember_loader = splitter.create_attack_test_split()

# 2. Train target model (separate from attack pipeline)
from target_models.fl_target import FraudDetectionNN, FLTargetTrainer

model = FraudDetectionNN(input_dim=20, hidden_dims=[64, 32], num_classes=2)
trainer = FLTargetTrainer(model=model, n_clients=10)
trained_model = trainer.train_fl_model(client_datasets, n_rounds=20)

# 3. Execute attack
from attacks.metric_attacks import loss_based_attack
from evaluation.attack_metrics import compute_attack_metrics, print_attack_results

all_scores, true_labels, _ = loss_based_attack(
    target_model=trained_model,
    member_data=member_loader,
    nonmember_data=nonmember_loader
)

metrics = compute_attack_metrics(all_scores, true_labels)
print_attack_results(metrics, "Loss-based Attack")
```

### Comparing Multiple Attacks

```python
from attacks.metric_attacks import aggregate_metric_attacks
from evaluation.attack_metrics import compare_attacks

results = aggregate_metric_attacks(
    target_model=trained_model,
    member_data=member_loader,
    nonmember_data=nonmember_loader,
    device='cpu',
    attacks=['loss', 'entropy', 'modified_entropy']
)

# Compare AUCs
compare_attacks(
    {name: r['metrics'] for name, r in results.items()},
    metric='auc',
    save_path='results/attack_comparison.png'
)
```

## Testing

Run unit tests to verify correctness:

```bash
# Test data separation (CRITICAL)
pytest tests/test_data_separation.py -v

# Test shadow models
pytest tests/test_shadow_models.py -v

# Test attack implementations
pytest tests/test_attacks.py -v

# Run all tests
pytest tests/ -v
```

## Threat Model

### Attacker Capabilities
- **Query Access**: Can query target model with arbitrary inputs
- **Knowledge**: Knows model architecture and training procedure
- **Shadow Data**: Has access to independent data similar to target training data
- **NO Access**: Does NOT have access to target model's training data

### Attacker Goals
1. **Membership Inference**: Determine if specific data point was in training set
2. **Participation Inference**: Determine which institution participated in FL
3. **Temporal Inference**: Track when data was added to training

### Limitations
- Assumes target model is overfitting (more training = more vulnerable)
- Effectiveness depends on data distribution similarity between shadow and target
- Random guessing baseline (AUC = 0.5) must be exceeded for successful attack

## Defense Recommendations

Based on empirical evaluation, the following defenses are recommended:

### 1. Differential Privacy
- **Effective**: Adding Gaussian noise (σ ≥ 1.0) significantly reduces attack success
- **Trade-off**: May reduce model utility (accuracy)
- **Recommendation**: Use σ = 1.0-2.0 for balanced privacy-utility trade-off

### 2. Early Stopping
- **Moderately Effective**: Fewer training rounds reduces overfitting
- **Trade-off**: May reduce model performance
- **Recommendation**: Monitor validation loss and stop when it plateaus

### 3. Regularization
- **Recommended**: Dropout, weight decay, and batch normalization
- **Effect**: Reduces overfitting, indirectly limiting membership leakage

### 4. Adversarial Training
- **Advanced**: Train with privacy-aware loss functions
- **Future Work**: Implement differential privacy in the loss function

## Ethical Considerations

This research is conducted for **defensive security purposes**:

1. **Understanding Vulnerabilities**: To identify privacy risks in FL systems
2. **Evaluating Defenses**: To test the effectiveness of privacy-preserving techniques
3. **Building Robust Systems**: To design FL systems that protect participant privacy

**Do NOT use** these attacks to:
- Infer private information about real individuals
- Violate data privacy regulations (GDPR, CCPA, etc.)
- Exploit vulnerabilities in production systems without authorization

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{membership_inference_attack_fl,
  title={Membership Inference Attacks on Federated Learning},
  author={Your Name},
  year={2025},
  note={Implementation of Shokri et al. (S&P 2017) for FL systems}
}
```

## References

1. Shokri, R., et al. "Membership Inference Attacks Against Machine Learning Models." IEEE S&P, 2017.
2. Nasr, M., et al. "Comprehensive Privacy Analysis of Deep Learning." PETS, 2019.
3. Abadi, M., et al. "Deep Learning with Differential Privacy." CCS, 2016.
4. Bonawitz, K., et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning." CCS, 2017.

## License

This project is provided for research and educational purposes.

## Contact

For questions or collaboration opportunities, please contact [your email].
