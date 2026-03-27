# Reproducing SignGuard Experiments

This guide provides step-by-step instructions for reproducing all experimental results from the SignGuard paper.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Hardware Requirements](#hardware-requirements)
4. [Quick Reproduction](#quick-reproduction)
5. [Detailed Experiment Instructions](#detailed-experiment-instructions)
6. [Troubleshooting](#troubleshooting)

---

## Requirements

### Software Dependencies

- Python 3.10 or higher
- PyTorch 2.0+
- See `requirements.txt` for full list

### Optional Dependencies

- `matplotlib` (for figure generation)
- `psutil` (for overhead measurement)

---

## Installation

### Option 1: Using pip

```bash
git clone https://github.com/username/signguard.git
cd signguard
pip install -e .
```

### Option 2: Using conda

```bash
git clone https://github.com/username/signguard.git
cd signguard
conda env create -f environment.yml
conda activate signguard
```

---

## Hardware Requirements

### Minimum Configuration

- CPU: 4 cores
- RAM: 8 GB
- Storage: 2 GB

### Recommended Configuration

- CPU: 8 cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 8GB VRAM (optional, for larger experiments)

### Runtime Estimates

- Single defense experiment (Table 1): ~5 minutes
- All experiments: ~30 minutes (with caching)
- Full reproduction from scratch: ~2 hours

---

## Quick Reproduction

### Fast Track (Using Cached Results)

```bash
cd signguard

# Run all experiments using cached results
cd experiments
./run_all_experiments.sh

# Results will be in experiments/results/
# Figures will be in figures/
```

### From Scratch

```bash
cd signguard

# Run experiments without cache
cd experiments
USE_CACHE=false ./run_all_experiments.sh
```

---

## Detailed Experiment Instructions

### Table 1: Defense Comparison

**Purpose**: Compare accuracy of different defenses under various attacks

**File**: `experiments/table1_defense_comparison.py`

**Command**:
```bash
python3 experiments/table1_defense_configuration.py
```

**Parameters**:
- `num_rounds`: 10 (default)
- `num_clients`: 10 (default)
- `num_byzantine`: 2 (default)

**Expected Output**:
- File: `experiments/results/table1_defense_comparison.json`
- Format: JSON with accuracy values for each defense × attack combination

**Interpretation**: Higher accuracy = better defense. SignGuard should maintain accuracy under attacks while FedAvg degrades.

---

### Table 2: Attack Success Rate

**Purpose**: Measure % reduction in ASR compared to FedAvg

**File**: `experiments/table2_attack_success_rate.py`

**Command**:
```bash
python3 experiments/table2_attack_success_rate.py
```

**Parameters**:
- `num_rounds`: 10 (default)
- `num_clients`: 10 (default)
- `num_byzantine`: 2 (default)

**Expected Output**:
- File: `experiments/results/table2_asr_comparison.json`
- Format: JSON with ASR values for each defense

**Interpretation**: Lower ASR = better defense. SignGuard should show significant ASR reduction vs FedAvg.

---

### Table 3: Overhead Analysis

**Purpose**: Measure computational, communication, and memory overhead

**File**: `experiments/table3_overhead_analysis.py`

**Command**:
```bash
python3 experiments/table3_overhead_analysis.py
```

**Parameters**:
- `num_rounds`: 5 (default, fewer for faster execution)
- `num_clients`: 10 (default)

**Expected Output**:
- File: `experiments/results/table3_overhead.json`
- Format: JSON with time/comm/memory overhead

**Interpretation**: SignGuard adds overhead but should remain practical (<2x baseline).

---

### Figure 1: Reputation Evolution

**Purpose**: Visualize reputation trajectories of honest vs malicious clients

**File**: `experiments/figure1_reputation_evolution.py`

**Command**:
```bash
python3 experiments/figure1_reputation_evolution.py
```

**Parameters**:
- `num_rounds`: 20 (default)
- `num_clients`: 10 (default)
- `num_byzantine`: 2 (default)

**Expected Output**:
- File: `figures/plots/figure1_reputation_evolution.pdf`
- Data: `figures/data/figure1_reputation_data.json` (if matplotlib unavailable)

**Interpretation**: Honest clients (blue) reputation increases, malicious (red) decreases.

---

### Figure 2: Detection ROC Curves

**Purpose**: Compare detection performance (SignGuard vs FoolsGold)

**File**: `experiments/figure2_detection_roc.py`

**Command**:
```bash
python3 experiments/figure2_detection_roc.py
```

**Parameters**:
- `num_honest`: 20 (synthetic updates)
- `num_malicious`: 5 (synthetic updates)

**Expected Output**:
- File: `figures/plots/figure2_detection_roc.pdf`

**Interpretation**: SignGuard ROC curve should be closer to top-left (higher TPR, lower FPR).

---

### Figure 3: Privacy-Utility Trade-off

**Purpose**: Show effect of adding DP to SignGuard

**File**: `experiments/figure3_privacy_utility.py`

**Command**:
```bash
python3 experiments/figure3_privacy_utility.py
```

**Parameters**:
- `epsilon_values`: [0.1, 0.5, 1.0, 5.0, 10.0]
- `num_rounds`: 10

**Expected Output**:
- File: `figures/plots/figure3_privacy_utility.pdf`

**Interpretation**: Accuracy and ASR trade off as ε (privacy budget) varies.

---

### Ablation Study

**Purpose**: Show contribution of each SignGuard component

**File**: `experiments/ablation_study.py`

**Command**:
```bash
python3 experiments/ablation_study.py
```

**Configurations Tested**:
- No Defense
- Crypto Only
- Detection Only
- Reputation Only
- Crypto + Detection
- Crypto + Reputation
- SignGuard (All Components)

**Expected Output**:
- File: `experiments/results/ablation_study.json`
- File: `figures/plots/ablation_study.pdf`

**Interpretation**: Full SignGuard should outperform partial configurations.

---

## Troubleshooting

### Issue: Module Not Found

**Error**:
```
ModuleNotFoundError: No module named 'signguard'
```

**Solution**:
```bash
# Make sure you're in the project root
cd signguard
pip install -e .
```

---

### Issue: Matplotlib Not Available

**Error**:
```
Matplotlib not available (optional dependency)
```

**Solution**:
```bash
pip install matplotlib
```

Or use offline plotting with data from `figures/data/` directory.

---

### Issue: Tests Failing

**Error**:
```
FAILED tests/test_crypto.py
```

**Solution**:
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Run specific test file
pytest tests/test_crypto.py -v
```

---

### Issue: Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```bash
# Reduce number of clients or rounds
# Edit experiment config files in experiments/config/
```

---

### Issue: Slow Experiment Execution

**Solution**:
- Use `--use-cache` flag when running scripts
- Reduce `num_rounds` parameter
- Run on GPU if available

---

## Verification

### Verify Installation

```bash
# Check package installation
python3 -c "from signguard import SignGuardClient, SignGuardServer; print('SignGuard installed')"

# Run test suite
pytest tests/ -v

# Run quick smoke test
python3 -c "
from signguard import SignatureManager, KeyStore
sm = SignatureManager()
ks = KeyStore()
ks.generate_keypair('test')
print('✓ SignGuard core functionality working')
"
```

---

## Expected Results Summary

### Performance Benchmarks

| Defense | No Attack | Label Flip | Backdoor | Model Poison |
|---------|-----------|------------|----------|--------------|
| FedAvg   | 0.85      | 0.65       | 0.55     | 0.50         |
| Krum    | 0.84      | 0.78       | 0.75     | 0.72         |
| FoolsGold| 0.83      | 0.80       | 0.77     | 0.75         |
| SignGuard| 0.82      | **0.81**    | **0.79**  | **0.78**       |

### Attack Success Rate Reduction

| Defense | ASR | Reduction vs FedAvg |
|---------|-----|-------------------|
| FedAvg   | 0.45 | - |
| Krum    | 0.32 | 29% |
| FoolsGold| 0.28 | 38% |
| **SignGuard** | **0.15** | **67%** |

---

## Contact

For reproduction issues:
- Check [GitHub Issues](https://github.com/username/signguard/issues)
- Email: `researcher@university.edu`

---

**Last Updated**: January 2024  
**SignGuard Version**: 0.1.0
