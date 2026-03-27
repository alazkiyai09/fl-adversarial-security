# Federated Learning Security Research - Architecture

## Overview

This portfolio contains 40+ projects exploring the intersection of federated learning and cybersecurity. The architecture is organized into 5 main categories:

1. **Fraud Detection Core** (7 projects)
2. **Federated Learning Foundations** (8 projects)
3. **Adversarial Attacks** (3 projects)
4. **Defensive Techniques** (5 projects)
5. **Security Research** (8 projects)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FL SECURITY RESEARCH PORTFOLIO                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         CLIENT LAYER                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Bank A       │  │ Bank B       │  │ Bank C       │  ...         │  │
│  │  │ Client       │  │ Client       │  │ Client       │              │  │
│  │  │              │  │              │  │              │              │  │
│  │  │ - Local Data │  │ - Local Data │  │ - Local Data │              │  │
│  │  │ - Local Train│  │ - Local Train│  │ - Local Train│              │  │
│  │  │ - Gradient   │  │ - Gradient   │  │ - Gradient   │              │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │  │
│  └─────────┼──────────────────┼──────────────────┼──────────────────────┘  │
│            │                  │                  │                          │
│            └──────────────────┼──────────────────┘                          │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      ATTACK/DEFENSE LAYER                             │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │  Attack        │  │  Defense       │  │  Monitoring    │         │  │
│  │  │  Simulations   │  │  Mechanisms    │  │  & Detection   │         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ - Data Poison  │  │ - SignGuard    │  │ - Anomaly Det. │         │  │
│  │  │ - Model Poison │  │ - FoolsGold    │  │ - Reputation   │         │  │
│  │  │ - Backdoor     │  │ - Krum/Multi-K │  │ - Logging      │         │  │
│  │  │ - Label Flip   │  │ - Trimmed Mean │  │ - Forensics    │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      AGGREGATION LAYER                                │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │  Secure        │  │  Robust        │  │  Privacy       │         │  │
│  │  │  Aggregation   │  │  Aggregation   │  │  Preserving    │         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ - Pairwise     │  │ - FedAvg       │  │ - Differential │         │  │
│  │  │   Masking      │  │ - FedProx      │  │   Privacy      │         │  │
│  │  │ - Secret Share │  │ - Krum         │  │ - Secure Agg.  │         │  │
│  │  │ - Encryption   │  │ - Multi-Krum   │  │ - TEE/SGX      │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      GLOBAL MODEL                                     │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │  Fraud         │  │  Evaluation    │  │  Deployment    │         │  │
│  │  │  Detection     │  │  & Metrics     │  │  & Serving     │         │  │
│  │  │  Model         │  │                │  │                │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Fraud Detection Core (01_fraud_detection_core)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION CORE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   EDA       │    │   Feature   │    │   Model     │        │
│  │ Dashboard   │───▶│ Engineering │───▶│ Training    │        │
│  │             │    │   Pipeline  │    │             │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                            │                   │                │
│                            ▼                   ▼                │
│                   ┌─────────────┐    ┌─────────────┐          │
│                   │   Baseline  │    │   LSTM      │          │
│                   │  Models     │    │  Sequences  │          │
│                   └─────────────┘    └─────────────┘          │
│                            │                   │                │
│                            └─────────┬─────────┘                │
│                                      ▼                          │
│                            ┌─────────────┐                    │
│                            │   Model     │                    │
│                            │ Explainable│                    │
│                            │   (SHAP)    │                    │
│                            └─────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Federated Learning Foundations (02_federated_learning_foundations)

```
┌─────────────────────────────────────────────────────────────────┐
│                FEDERATED LEARNING FOUNDATIONS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Client Side                    Server Side                      │
│  ┌─────────────┐               ┌─────────────┐                 │
│  │    Local    │               │   Global    │                 │
│  │    Data     │               │   Model     │                 │
│  └──────┬──────┘               └──────┬──────┘                 │
│         │                             │                         │
│         ▼                             ▼                         │
│  ┌─────────────┐               ┌─────────────┐                 │
│  │    Local    │◀─────────────▶│  Aggregate  │                 │
│  │   Training  │   Updates     │   (FedAvg)  │                 │
│  └──────┬──────┘               └──────┬──────┘                 │
│         │                             │                         │
│         ▼                             ▼                         │
│  ┌─────────────┐               ┌─────────────┐                 │
│  │  Gradient   │─────────────▶│    Global   │                 │
│  │   Update    │               │  Update    │                 │
│  └─────────────┘               └─────────────┘                 │
│                                                                 │
│  Variations:                                                     │
│  - Cross-Silo FL (Banks)                                        │
│  - Personalized FL (Per-user models)                            │
│  - Hierarchical FL (Regional aggregation)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Adversarial Attacks (03_adversarial_attacks)

```
┌─────────────────────────────────────────────────────────────────┐
│                      ADVERSARIAL ATTACKS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Data Poisoning                  Model Poisoning                │
│  ┌─────────────┐                ┌─────────────┐               │
│  │  Label      │                │  Gradient   │               │
│  │  Flipping   │                │  Scaling    │               │
│  └─────────────┘                └─────────────┘               │
│  ┌─────────────┐                ┌─────────────┐               │
│  │  Backdoor   │                │  Sign       │               │
│  │  Injection  │                │  Flipping   │               │
│  └─────────────┘                └─────────────┘               │
│                                                                 │
│  Attack Detection                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │ L2 Norm │  │Cosine   │  │ Euclidean│  │  KL     │   │   │
│  │  │ Monitor │  │Sim      │  │ Distance │  │ Div     │   │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Defensive Techniques (04_defensive_techniques)

```
┌─────────────────────────────────────────────────────────────────┐
│                      DEFENSIVE TECHNIQUES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Byzantine-Robust Aggregation                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │  Krum   │  │Multi-Krum│  │Trimmed  │  │FoolsGold│   │   │
│  │  │         │  │         │  │  Mean   │  │         │   │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 SignGuard Defense System                 │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │   │
│  │  │  ECDSA  │───▶│Verify   │───▶│Reputation│             │   │
│  │  │  Sign   │    │Signatures│   │ System   │             │   │
│  │  └─────────┘    └─────────┘    └─────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Anomaly Detection                         │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │   │
│  │  │ Z-Score │    │  IQR    │    │ Isolation│             │   │
│  │  │ Method  │    │ Method  │    │  Forest  │             │   │
│  │  └─────────┘    └─────────┘    └─────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Security Research (05_security_research)

```
┌─────────────────────────────────────────────────────────────────┐
│                      SECURITY RESEARCH                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Privacy Attacks                 Privacy Defense                │
│  ┌─────────────┐                ┌─────────────┐               │
│  │  Member-    │                │  Differential│               │
│  │  ship       │                │  Privacy    │               │
│  │  Inference  │                │  (DP-SGD)   │               │
│  └─────────────┘                └─────────────┘               │
│  ┌─────────────┐                ┌─────────────┐               │
│  │  Property   │                │  Secure     │               │
│  │  Inference  │                │  Aggregation│               │
│  └─────────────┘                └─────────────┘               │
│  ┌─────────────┐                ┌─────────────┐               │
│  │  Gradient   │                │  Homomorphic│               │
│  │  Leakage    │                │  Encryption │               │
│  └─────────────┘                └─────────────┘               │
│                                                                 │
│  Trusted Execution Environment                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │   │
│  │  │  SGX    │    │  Attestation│   │  Secure  │         │   │
│  │  │  Enclave│    │  & Verification│  Computation│       │   │
│  │  └─────────┘    └─────────┘    └─────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA COLLECTION                                             │
│     ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│     │ Transaction│  │  Customer│  │  Device │                 │
│     │   Logs   │    │   Data  │    │   Data  │                 │
│     └────┬─────┘    └────┬─────┘    └────┬─────┘                 │
│          │               │               │                      │
│          └───────────────┴───────────────┘                      │
│                          ▼                                       │
│                  ┌─────────────┐                                 │
│                  │ Local Data  │                                 │
│                  │  Store      │                                 │
│                  └──────┬──────┘                                 │
│                         │                                        │
│  2. LOCAL TRAINING      ▼                                        │
│                  ┌─────────────┐                                 │
│                  │  Local      │                                 │
│                  │  Training   │                                 │
│                  │  (FedAvg)   │                                 │
│                  └──────┬──────┘                                 │
│                         │                                        │
│  3. GRADIENT COMPUTATION ▼                                        │
│                  ┌─────────────┐                                 │
│                  │  Gradient   │                                 │
│                  │  Update     │                                 │
│                  └──────┬──────┘                                 │
│                         │                                        │
│  4. OPTIONAL ATTACK    ▼                                        │
│                  ┌─────────────┐                                 │
│                  │  Poisoning  │                                 │
│                  │  Injection  │                                 │
│                  └──────┬──────┘                                 │
│                         │                                        │
│  5. TRANSMISSION       ▼                                        │
│                  ┌─────────────┐                                 │
│                  │  Network    │                                 │
│                  │  Channel    │                                 │
│                  └──────┬──────┘                                 │
│                         │                                        │
│  6. SERVER RECEPTION   ▼                                        │
│                  ┌─────────────┐                                 │
│                  │  Update     │                                 │
│                  │  Collection │                                 │
│                  └──────┬──────┘                                 │
│                         │                                        │
│  7. VERIFICATION       ▼                                        │
│      ┌──────────────────┴──────────────────┐                    │
│      │                  │                  │                    │
│  ┌───▼────┐       ┌────▼────┐       ┌────▼────┐               │
│  │Signature│       │ Anomaly │       │Reputation│              │
│  │ Verify │       │ Detect  │       │  Check  │               │
│  └───┬────┘       └────┬────┘       └────┬────┘               │
│      │                │                 │                     │
│      └────────────────┴─────────────────┘                     │
│                         ▼                                        │
│  8. AGGREGATION      ┌─────────────┐                             │
│                  │  Robust     │                             │
│                  │  Aggregate │                             │
│                  └──────┬──────┘                             │
│                         │                                        │
│  9. GLOBAL UPDATE     ▼                                        │
│                  ┌─────────────┐                                 │
│                  │  Global     │                                 │
│                  │  Model      │                                 │
│                  └──────┬──────┘                                 │
│                         │                                        │
│  10. DISTRIBUTION     ▼                                        │
│                  ┌─────────────┐                                 │
│                  │  Broadcast  │                                 │
│                  │  to Clients │                                 │
│                  └─────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **ML Framework** | PyTorch, TensorFlow, scikit-learn |
| **FL Framework** | Flower (Flwr), TensorFlow Federated |
| **Cryptography** | ecdsa, cryptography.io, TenSEAL |
| **API** | FastAPI, Flask |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Testing** | pytest |
| **Security** | libsnark, circom (ZKP) |

---

## Key Security Properties

1. **Confidentiality**: Client data never leaves local premises
2. **Integrity**: ECDSA signatures verify model updates
3. **Availability**: Byzantine-resilient aggregation ensures robustness
4. **Privacy**: Differential privacy and secure aggregation protect individual contributions
5. **Auditability**: Comprehensive logging for forensic analysis
