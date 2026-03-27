# Baseline Data Directory

Place baseline (honest) model updates here for training detectors.

Format:
- `.npy` files: NumPy arrays of flattened updates
- Folder structure: Organize by dataset/model if needed

Example:
```
data/raw/
├── mnist_cnn_baseline.npy
├── credit_fraud_baseline.npy
└── ...
```

Load example:
```python
import numpy as np
baseline_updates = np.load("data/raw/mnist_cnn_baseline.npy", allow_pickle=True)
```
