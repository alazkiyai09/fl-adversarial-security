# Jupyter Notebooks Portfolio

Interactive educational notebooks demonstrating key federated learning security concepts.

## 📚 Available Notebooks (23/23 Complete)

All 23 implemented projects now have interactive Jupyter notebooks!

### Fraud Detection Core (Days 1-7)
- **[01_day1_fraud_detection_eda.ipynb](./01_day1_fraud_detection_eda.ipynb)**
  - Interactive EDA for fraud detection
  - Class distribution, amount analysis, correlation, PCA
  - Plotly visualizations

- **[day2_imbalanced_learning.ipynb](./day2_imbalanced_learning.ipynb)**
  - Handling class imbalance in fraud detection
  - SMOTE resampling, class weighting, threshold moving
  - Comparison of techniques

- **[day3_feature_engineering.ipynb](./day3_feature_engineering.ipynb)**
  - Advanced feature engineering for fraud
  - Velocity features, deviation features, merchant risk
  - Feature importance analysis

- **[day5_lstm_autoencoder.ipynb](./day5_lstm_autoencoder.ipynb)**
  - Sequence modeling with LSTM + Attention
  - Variable-length transaction sequences
  - Attention weight visualization

- **[day6_anomaly_detection.ipynb](./day6_anomaly_detection.ipynb)**
  - Unsupervised anomaly detection
  - Isolation Forest, One-Class SVM, LOF
  - Ensemble methods

- **[day7_model_explainability.ipynb](./day7_model_explainability.ipynb)**
  - Model interpretation for fraud detection
  - SHAP values, LIME, feature importance
  - Global and local explanations

### Federated Learning Foundations (Days 8-13, 20, 22)
- **[day8_fedavg_from_scratch.ipynb](./day8_fedavg_from_scratch.ipynb)**
  - Implement FedAvg from scratch
  - Client-server architecture
  - Weighted averaging

- **[day9_non_iid_partitioning.ipynb](./day9_non_iid_partitioning.ipynb)**
  - Non-IID data partitioning strategies
  - Dirichlet distribution for label skew
  - Quantity skew and feature skew

- **[day10_flower_framework.ipynb](./day10_flower_framework.ipynb)**
  - Production FL with Flower framework
  - Client-server implementation
  - Custom aggregation strategies

- **[02_day11_communication_efficient_fl.ipynb](./02_day11_communication_efficient_fl.ipynb)**
  - Gradient compression techniques
  - Sparsification (Top-K), Quantization (8-bit, 4-bit)
  - Error feedback and accuracy trade-offs

- **[day12_cross_silo_bank_fl.ipynb](./day12_cross_silo_bank_fl.ipynb)**
  - Cross-silo FL for banks
  - Regulatory compliance
  - Deployment architecture

- **[day13_vertical_fl.ipynb](./day13_vertical_fl.ipynb)**
  - Vertical federated learning
  - Feature partitioning vs sample partitioning
  - Split learning approach

- **[day20_personalized_fl.ipynb](./day20_personalized_fl.ipynb)**
  - Federated personalization
  - Fine-tuning global models
  - MAML for FL

- **[day22_differential_privacy.ipynb](./day22_differential_privacy.ipynb)**
  - Differential privacy in FL
  - DP-SGD algorithm
  - Privacy accounting and utility tradeoffs

### Adversarial Attacks (Days 14-16)
- **[03_day14_label_flipping_attack.ipynb](./03_day14_label_flipping_attack.ipynb)**
  - Label flipping attack variants
  - Random flip, Targeted flip, Inverse flip
  - Attack impact visualization

- **[day15_backdoor_attack.ipynb](./day15_backdoor_attack.ipynb)**
  - Backdoor attacks on FL models
  - Trigger injection
  - Attack success rate (ASR)

- **[day16_model_poisoning.ipynb](./day16_model_poisoning.ipynb)**
  - Model poisoning attacks
  - Gradient scaling, sign flipping, inner product
  - Detection methods

### Defensive Techniques (Days 17-19, 21)
- **[day17_byzantine_robust_aggregation.ipynb](./day17_byzantine_robust_aggregation.ipynb)**
  - Byzantine-robust aggregation
  - Krum, Multi-Krum, Trimmed Mean
  - Robustness analysis

- **[day18_anomaly_detection_fl.ipynb](./day18_anomaly_detection_fl.ipynb)**
  - Anomaly detection for FL security
  - L2 norm, cosine similarity detection
  - Ensemble detection

- **[04_day19_foolsgold_defense.ipynb](./04_day19_foolsgold_defense.ipynb)**
  - FoolsGold: Sybil-resistant aggregation
  - Pairwise similarity computation
  - Contribution scores (alpha)

### Security Research (Days 23-25)
- **[day23_secure_aggregation.ipynb](./day23_secure_aggregation.ipynb)**
  - Secure aggregation (Bonawitz et al.)
  - Shamir's secret sharing
  - Pairwise masking protocol

- **[05_day24_signguard_core_research.ipynb](./05_day24_signguard_core_research.ipynb)**
  - **CORE RESEARCH CONTRIBUTION**
  - Multi-layer defense system
  - ECDSA signatures + Anomaly detection + Reputation

- **[day25_membership_inference_attack.ipynb](./day25_membership_inference_attack.ipynb)**
  - Membership inference attacks
  - Shadow model training
  - Privacy attacks and defenses

## 🚀 Getting Started

### Running the Notebooks

```bash
# Install Jupyter
pip install jupyter

# Navigate to notebooks folder
cd /home/ubuntu/30Days_Project/notebooks

# Start Jupyter
jupyter notebook

# OR use JupyterLab (recommended)
jupyter lab
```

### Requirements

Each notebook lists its required packages. Common dependencies:

```bash
pip install jupyter numpy pandas matplotlib plotly scikit-learn torch seaborn
```

## 📖 Notebook Structure

Each notebook follows this educational structure:

1. **Overview** - Project description and objectives
2. **Setup** - Installation and imports
3. **Concept Explanation** - Theoretical background
4. **Code Examples** - Interactive demonstrations
5. **Visualization** - Charts and graphs
6. **Results Analysis** - Key findings and insights
7. **Summary** - Takeaways and next steps

## 🎯 Learning Path

We recommend following this sequence:

### 1. Fraud Detection Core (Week 1)
1. `01_day1_fraud_detection_eda.ipynb` - Understand fraud detection basics
2. `day2_imbalanced_learning.ipynb` - Handle class imbalance
3. `day3_feature_engineering.ipynb` - Create better features
4. `day5_lstm_autoencoder.ipynb` - Sequence modeling
5. `day6_anomaly_detection.ipynb` - Unsupervised detection
6. `day7_model_explainability.ipynb` - Model interpretability

### 2. FL Foundations (Week 2)
1. `day8_fedavg_from_scratch.ipynb` - Implement FedAvg
2. `day9_non_iid_partitioning.ipynb` - Realistic data splits
3. `day10_flower_framework.ipynb` - Production FL
4. `02_day11_communication_efficient_fl.ipynb` - Gradient compression
5. `day12_cross_silo_bank_fl.ipynb` - Real-world deployment
6. `day13_vertical_fl.ipynb` - Feature partitioning

### 3. Attacks & Defenses (Week 3)
1. `03_day14_label_flipping_attack.ipynb` - Data poisoning
2. `day15_backdoor_attack.ipynb` - Hidden triggers
3. `day16_model_poisoning.ipynb` - Gradient manipulation
4. `day17_byzantine_robust_aggregation.ipynb` - Robust aggregation
5. `day18_anomaly_detection_fl.ipynb` - Update anomaly detection
6. `04_day19_foolsgold_defense.ipynb` - Sybil resistance

### 4. Advanced Security (Week 4)
1. `day20_personalized_fl.ipynb` - Personalization
2. `day22_differential_privacy.ipynb` - Privacy guarantees
3. `day23_secure_aggregation.ipynb` - Cryptographic privacy
4. `05_day24_signguard_core_research.ipynb` - Multi-layer defense (CORE)
5. `day25_membership_inference_attack.ipynb` - Privacy attacks

## 💡 Usage Tips

### Running Cells
- **Shift + Enter**: Run current cell and advance
- **Ctrl + Enter**: Run current cell (don't advance)
- **Alt + Enter**: Run cell and insert below

### Common Operations
- **Restart Kernel**: Kernel → Restart & Clear Output
- **Run All**: Cell → Run All
- **Auto-save**: File → Auto-Save (enabled by default)

## 📊 Notebook Features

- ✅ **Interactive Code**: Modify parameters and see results instantly
- ✅ **Visualizations**: Rich Plotly charts and Matplotlib graphs
- ✅ **Explanations**: Detailed markdown documentation
- ✅ **Self-Contained**: Each notebook can run independently
- ✅ **Educational**: Learn concepts through hands-on experimentation

## 📈 Portfolio Coverage

| Category | Notebooks | Projects Coverage |
|----------|------------|-------------------|
| Fraud Detection Core | 7/7 | 100% |
| FL Foundations | 9/9 | 100% |
| Adversarial Attacks | 3/3 | 100% |
| Defensive Techniques | 3/5 | 60% |
| Security Research | 3/7 | 43% |
| **Total** | **23/23** | **77%** |

**Note**: All notebooks are self-contained educational demonstrations. For complete implementations, see the project folders.

## 🤝 Contributing

These notebooks are part of the 30-Day Federated Learning Security Portfolio.

For questions or feedback:
- **Email**: azka.alazkiyai@outlook.com
- **GitHub**: [@alazkiyai09](https://github.com/alazkiyai09)

---

**📁 Location**: `notebooks/` (root of portfolio)

**🔗 Related**: [Main README](../README.md)
