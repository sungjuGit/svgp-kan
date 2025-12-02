# SVGP-KAN: Sparse Variational Gaussian Process Kolmogorov-Arnold Networks

**SVGP-KAN** is a library for building interpretable, probabilistic, and scalable neural networks. It merges the architecture of **Kolmogorov-Arnold Networks (KANs)** with the uncertainty quantification of **Sparse Variational Gaussian Processes**.

Unlike the original GP-KAN (which scales cubically $O(N^3)$), this implementation scales linearly $O(N)$ with data size, making it usable for real-world scientific datasets.

## Key Features

  * **Probabilistic:** Outputs mean predictions and confidence intervals (uncertainty).
  * **Scalable:** Uses inducing points and matrix-based batching to train on large datasets (100k+ samples).
  * **Scientific Discovery:** Includes **Automatic Relevance Determination (ARD)** to automatically identify signal vs. noise features.
  * **Computer Vision:** Natively supports 4D inputs (images) for probabilistic U-Nets and CNNs.
  * **Universal:** The RBF kernel naturally adapts to learn periodic, linear, or complex non-linear functions.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/yourusername/svgp-kan.git
    cd svgp-kan
    ```

2.  Install the library (automatically installs PyTorch dependencies):

    ```bash
    pip install .
    ```

## Quick Start

### 1\. Scientific Discovery (Feature Selection)

Train a model to identify which inputs matter and which are noise.

```python
import numpy as np
from svgp_kan import GPKANRegressor

# 1. Generate Data (Sine + Linear + Noise)
X = np.random.rand(1000, 3)
# Feature 0: Sine, Feature 1: Linear, Feature 2: Noise
y = np.sin(X[:, 0]*5) + 2*X[:, 1] + 0*X[:, 2] 

# IMPORTANT: Reshape y to [N, 1] for PyTorch training
y = y.reshape(-1, 1)

# 2. Train Scientist Model
# We use RBF kernels as universal approximators
model = GPKANRegressor(hidden_layers=[3, 5, 1], kernel='rbf')
model.fit(X, y, epochs=500, sparsity_weight=0.05)

# 3. Interpret Results
model.explain()
# Output Example:
# Feature 0: [ACTIVE] Type: Non-Linear (Sine)
# Feature 1: [ACTIVE] Type: Linear (Large Scale)
# Feature 2: [PRUNED] (Irrelevant)

# 4. Predict with Uncertainty
mu, std = model.predict(X)
```

### 2\. Running the Benchmarks

To reproduce the results from our paper (Friedman \#1 dataset), run the included example script:

```bash
python examples/benchmark_friedman.py
```

This will generate high-resolution diagnostic plots in your `examples/` directory:

  * `friedman_feature_importance.png`
  * `friedman_interaction_surface.png`

## 👁️ U-Net Support (New in v0.2)

SVGP-KAN now natively supports U-Net for **2D Computer Vision** tasks (Segmentation, Reconstruction) and other applications via the `SVGPUNet` module. This allows you to generate **calibrated uncertainty maps** for images, essential for medical imaging and safety-critical applications.

**Why use SVGP-KAN for Vision?**

  * **Uncertainty:** Returns a pixel-wise "confusion map" where the model is unsure (e.g., ambiguous tumor boundaries).
  * **Data Efficiency:** GP inductive biases often perform better on small datasets (common in scientific imaging).

### Usage Example

```python
import torch
import matplotlib.pyplot as plt
from svgp_kan import SVGPUNet

# 1. Setup Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Initialize the Probabilistic U-Net
# Input: 1 Channel (Grayscale), Output: 2 Classes (Background, Object)
model = SVGPUNet(in_channels=1, num_classes=2, base=32).to(device)

# 3. Forward Pass
# Input: [Batch, Channel, Height, Width]
images = torch.randn(4, 1, 128, 128).to(device)
logits, uncertainty = model(images)

# logits:      [4, 2, 128, 128] -> Standard segmentation mask
# uncertainty: [4, 1, 128, 128] -> Variance map (High value = High uncertainty)

# 4. Visualization
plt.imshow(uncertainty[0, 0].cpu().detach(), cmap='hot')
plt.title("Model Uncertainty / Confusion Map")
plt.show()
```

### Custom CNN Integration

You can also drop `GPKANLayer` into *any* existing CNN. It automatically handles 4D tensors `[B, C, H, W]`.

```python
import torch.nn as nn
from svgp_kan import GPKANLayer

class MyCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        # Replaces a standard Conv1x1 or Linear layer
        self.gp_layer = GPKANLayer(in_features=16, out_features=32)
        
    def forward(self, x):
        x = self.conv(x)
        # No flattening needed!
        # GPKANLayer automatically processes spatial grid
        x_mean, x_var = self.gp_layer(x)
        return x_mean
```

## Citation

If you use **SVGP-KAN** in your research, please cite the following paper:

```bibtex
@article{ju2025svgpkan,
  title   = {Scalable and Interpretable Scientific Discovery via Sparse Variational Gaussian Process Kolmogorov-Arnold Networks (SVGP-KAN)},
  author  = {Ju, Sungtaek},
  journal = {arXiv preprint arXiv:2512.00260},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.00260},
  url     = {https://arxiv.org/abs/2512.00260}
}
```