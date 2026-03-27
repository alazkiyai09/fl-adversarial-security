# System Architecture

## Overview

The Privacy-Preserving Federated Learning Fraud Detection System is designed for multiple financial institutions to collaboratively train fraud detection models while preserving data privacy and ensuring security against malicious actors.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FL Server                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  SignGuard   │  │  Privacy     │  │   Secure Aggregation          │  │
│  │  Defense     │→ │  Module      │→ │   (Optional - Pairwise Mask)│  │
│  │              │  │              │  │                              │  │
│  │              │  │  DP-SGD      │  │                              │  │
│  │  + Attack    │  │  Accountant  │  │                              │  │
│  │  Detection   │  │  (ε, δ)      │  │                              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
│                                    ↓                                    │
│                        ┌────────────────────┐                          │
│                        │   Aggregation      │                          │
│                        │   Strategy         │                          │
│                        │   (FedAvg/Prox...)  │                          │
│                        └────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↕
        ┌───────────────────────────┼───────────────────────────┐
        ↓                           ↓                           ↓
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   Client 1    │          │   Client 2    │          │   Client N    │
│               │          │               │          │               │
│ ┌───────────┐│          │┌───────────┐ │          │┌───────────┐ │
│ │  Local    ││          ││  Local    │ │          ││  Local    │ │
│ │  Training ││          ││  Training │ │          ││  Training │ │
│ │  + DP-SGD ││          ││  + DP-SGD │ │          ││  + DP-SGD │ │
│ └───────────┘│          │└───────────┘ │          │└───────────┘ │
│ ┌───────────┐│          │┌───────────┐ │          │┌───────────┐ │
│ │  Privacy  ││          ││  Privacy  │ │          ││  Privacy  │ │
│ │  Module   ││          ││  Module   │ │          ││  Module   │ │
│ └───────────┘│          │└───────────┘ │          │└───────────┘ │
│               │          │               │          │               │
│  Bank 1 Data │          │  Bank 2 Data │          │  Bank N Data │
└───────────────┘          └───────────────┘          └───────────────┘
```

## Component Details

### 1. Data Module

**Purpose**: Preprocess and partition data across clients

**Key Components**:
- **Preprocessing**: Feature engineering, scaling, handling imbalance
  - Temporal features (hour, day of week)
  - Behavioral patterns (transaction frequency, amount statistics)
  - Sequence generation for LSTM/Transformer
- **Partitioning**: Non-IID data distribution
  - Dirichlet partition (alpha controls skew)
  - Pathological partition (each client gets limited classes)
  - Account-based partition (realistic for banks)
- **Validation**: Data quality checks and schema validation

**Data Flow**:
```
Raw Transactions → Feature Engineering → Scaling → Sequences
                                                      ↓
                                           Partition by Client
                                                      ↓
                                  Client 1 Data, Client 2 Data, ...
```

### 2. Model Module

**Purpose**: Model architectures for fraud detection

**Supported Models**:
1. **LSTM**: Recurrent architecture for sequence modeling
   - Captures temporal patterns
   - Bidirectional option
   - Attention mechanism available

2. **Transformer**: Self-attention based
   - Positional encoding
   - Multi-head attention
   - Better for long-range dependencies

3. **XGBoost**: Gradient boosting (alternative)
   - Tree-based
   - Works with non-sequential data
   - PyTorch wrapper for FL compatibility

### 3. Federated Learning Module

**Purpose**: Coordinate distributed training

**Server Components**:
- **Strategy Factory**: Creates aggregation strategies
  - FedAvg: Weighted average
  - FedProx: Proximal term for non-IID
  - FedAdam: Adaptive moments

- **Defense Integration**:
  ```
  Client Updates → SignGuard → Filtered Updates → Aggregation
  ```

**Client Components**:
- Local training loop
- Differential Privacy (optional)
- Evaluation on local test set
- Model checkpointing

**Communication Protocol** (Flower):
```
Server                                      Client
  |  Select clients                           |
  |  -------------------------------------->  |
  |  Send global model                       |
  |                                           |
  |  Local training (DP-SGD if enabled)      |
  |  <--------------------------------------  |
  |  Receive model update                    |
  |                                           |
  |  Aggregate (with defense)                |
  |  Broadcast new global model              |
```

### 4. Privacy Module

**Differential Privacy**:
- **DP-SGD**: Per-step privacy
  - Gradient clipping (max_norm)
  - Gaussian noise (noise_multiplier)
  - Privacy accountant tracks (ε, δ)

- **Privacy Budget**:
  ```
  Initial: ε=0, δ=target_delta
  After training: ε=1.0, δ=1e-5
  Budget exhausted when ε ≥ target_epsilon
  ```

**Secure Aggregation**:
- **Pairwise Masking**:
  ```
  For pair (i, j) where i < j:
    Client i adds +mask_ij
    Client j adds -mask_ij

  Result: Σ masks = 0 (cancels out)
  ```

- **Security**: Server sees only aggregated sum
- **Collusion**: Requires > n/2 clients to breach

### 5. Security Module

**Attack Detection**:
1. **Poisoning Detection**:
   - Statistical: Z-score on update magnitudes
   - Clustering: DBSCAN for outliers
   - Isolation Forest: ML-based
   - Hybrid: Voting ensemble

2. **Backdoor Detection**:
   - Test on backdoor test set
   - Check prediction consistency
   - Analyze trigger effects

3. **Label Flipping Detection**:
   - Monitor fraud rates
   - Detect systematic flipping

**SignGuard Defense**:
- Analyzes sign patterns of gradients
- Filters malicious updates
- Adaptive thresholding

**Anomaly Logging**:
- Structured JSON logging
- Severity levels (LOW, MEDIUM, HIGH, CRITICAL)
- Event querying and filtering
- Client risk scoring

**Alerting**:
- Multi-channel: Email, Slack, Teams, Webhook
- Severity-based filtering
- Rate limiting

### 6. Serving Module

**FastAPI Service**:
- `/predict`: Batch predictions
- `/predict/single`: Single transaction
- `/model/info`: Model information
- `/model/activate/{version}`: Model versioning
- `/health`: Health check

**Model Store**:
- Version management
- Rollback support
- Automatic cleanup
- Export (TorchScript, ONNX)

**Prediction Pipeline**:
```
Transaction Data → Preprocessing → Model → Post-processing → Response
                      ↓                    ↓
                   Feature             Apply
                   Engineering         Threshold
```

### 7. Monitoring Module

**MLflow Tracking**:
- Hyperparameters
- Per-round metrics
- Model checkpoints
- Privacy metrics (ε, δ)
- Defense events

**Metrics**:
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Fraud-specific metrics (detection rate, false alarm rate)
- Cost-based metrics (savings rate)

## Privacy Guarantees

### Differential Privacy
- **Guarantee**: (ε, δ)-differential privacy
- **Protection**: Individual transactions
- **Tunable**: Adjust ε via noise_multiplier
- **Accounting**: RDP accountant for accurate tracking

### Secure Aggregation
- **Guarantee**: Information-theoretic privacy
- **Protection**: Individual model updates
- **Threat Model**: Honest-but-curious server
- **Collusion**: Requires > n/2 clients

### Security
- **Threat Model**: Byzantine clients
- **Protection**: SignGuard + Attack Detection
- **Detection**: Statistical + ML methods
- **Response**: Filter malicious updates

## Deployment Patterns

### Local Development
```bash
# Single-machine simulation
python scripts/run_training.py
```

### Docker Deployment
```bash
# Server
docker-compose up -d server

# Clients (multiple containers)
docker-compose up -d client1 client2 client3
```

### Production (Kubernetes)
```bash
# Deploy to cluster
kubectl apply -f deployment/kubernetes/
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| FL Framework | Flower |
| Deep Learning | PyTorch |
| Configuration | Hydra |
| API | FastAPI |
| Privacy | Opacus (DP), Custom (secure agg) |
| Monitoring | MLflow |
| Deployment | Docker, Kubernetes (optional) |
