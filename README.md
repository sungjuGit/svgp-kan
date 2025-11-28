# SVGP-KAN: Sparse Variational Gaussian Process Kolmogorov-Arnold Networks

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**SVGP-KAN** is a library for building interpretable, probabilistic, and scalable neural networks. It merges the architecture of **Kolmogorov-Arnold Networks (KANs)** with the uncertainty quantification of **Sparse Variational Gaussian Processes**.

Unlike the original GP-KAN (which scales cubically $O(N^3)$), this implementation scales linearly $O(N)$ with data size, making it usable for real-world datasets.

## Key Features

* **Probabilistic:** Outputs mean predictions and confidence intervals (uncertainty).
* **Scalable:** Uses inducing points and matrix-based batching to train on large datasets.
* **Scientific Discovery:** Includes Automatic Relevance Determination (ARD) to automatically prune useless inputs and "discover" linear vs. non-linear relationships.
* **Universal:** The RBF kernel can adapt to learn periodic, linear, or complex non-linear functions.

## Citation
If you use this code, please cite our arXiv manuscript: Authors, T. B. D. (2025). Scalable and Interpretable Scientific Discovery via Sparse Variational Gaussian Process Kolmogorov-Arnold Networks.

## Quick Start

```python
import numpy as np
from svgp_kan.regressor import GPKANRegressor

# 1. Generate Data (Sine + Linear + Noise)
X = np.random.rand(1000, 3)
y = np.sin(X[:, 0]*5) + 2*X[:, 1] + 0*X[:, 2] # x2 is irrelevant

# 2. Train Scientist Model
model = GPKANRegressor(hidden_layers=[3, 5, 1], kernel='rbf')
model.fit(X, y, epochs=500, sparsity_weight=0.05)

# 3. Interpret
model.explain()
# Output:
# Feature 0: [ACTIVE] Type: Non-Linear (Sine)
# Feature 1: [ACTIVE] Type: Linear (Large Lengthscale)
# Feature 2: [PRUNED] (Irrelevant)

# 4. Predict with Uncertainty
mu, std = model.predict(X_test)
