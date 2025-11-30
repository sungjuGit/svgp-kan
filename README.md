# **SVGP-KAN: Sparse Variational Gaussian Process Kolmogorov-Arnold Networks**

**SVGP-KAN** is a library for building interpretable, probabilistic, and scalable neural networks. It merges the architecture of **Kolmogorov-Arnold Networks (KANs)** with the uncertainty quantification of **Sparse Variational Gaussian Processes**.

Unlike the original GP-KAN (which scales cubically $O(N^3)$), this implementation scales linearly $O(N)$ with data size, making it usable for real-world scientific datasets.

## **Key Features**

* **Probabilistic:** Outputs mean predictions and confidence intervals (uncertainty).  
* **Scalable:** Uses inducing points and matrix-based batching to train on large datasets (100k+ samples).  
* **Scientific Discovery:** Includes **Permutation Importance** and **Automatic Relevance Determination (ARD)** tools to identify signal vs. noise features.  
* **Universal:** The RBF kernel naturally adapts to learn periodic, linear, or complex non-linear functions.

## **Installation**

1. Clone the repository:  
   git clone https://github.com/yourusername/svgp-kan.git 
   cd svgp-kan

2. Install the library (automatically installs PyTorch dependencies):  
   pip install .

## **Quick Start**

### **1\. Scientific Discovery (Feature Selection)**

Train a model to identify which inputs matter and which are noise.

```python

import numpy as np  
from svgp_kan import GPKANRegressor

# 1. Generate Data (Sine + Linear + Noise)  
X = np.random.rand(1000, 3)  
y = np.sin(X[:, 0]*5) + 2*X[:, 1] + 0*X[:, 2] # x2 is irrelevant noise

# 2. Train Scientist Model  
# We use RBF kernels as universal approximators  
model = GPKANRegressor(hidden_layers=[3, 5, 1], kernel='rbf')  
model.fit(X, y, epochs=500, sparsity_weight=0.05)

# 3. Interpret Results  
model.explain()  
# Output:  
# Feature 0: [ACTIVE] Type: Non-Linear (Sine)  
# Feature 1: [ACTIVE] Type: Linear (Large Lengthscale)  
# Feature 2: [PRUNED] (Irrelevant)

# 4. Predict with Uncertainty  
mu, std = model.predict(X)
```

### **2\. Running the Benchmarks**

To reproduce the results from our paper (Friedman \#1 dataset), run the included example script:

python examples/benchmark\_friedman.py

This will generate high-resolution diagnostic plots in your current directory:

* friedman\_feature\_importance.png  
* friedman\_interaction\_surface.png

## **Citation**

If you use this code or concepts in your research, please cite our arXiv manuscript:

@article{ju2025svgpkan,  
  title={Scalable and Interpretable Scientific Discovery via Sparse Variational Gaussian Process Kolmogorov-Arnold Networks},  
  author={Ju, Y. S.},  
  journal={arXiv preprint},  
  year={2025}  
}

## 👁️ U-Net Support (New in v0.2)
SVGP-KAN now natively supports U-Net for 2D Computer Vision tasks (Segmentation, Reconstruction) and other applications via the SVGPUNet module. This allows you to generate calibrated uncertainty maps for images, essential for medical imaging and safety-critical applications.

Why use SVGP-KAN for Vision?
Unlike standard U-Nets, SVGP-UNet replaces the bottleneck with a Probabilistic Gaussian Process.

Uncertainty: Returns a pixel-wise "confusion map" where the model is unsure (e.g., ambiguous tumor boundaries).

Data Efficiency: GP inductive biases often perform better on small datasets (common in scientific imaging).

### Usage Example

```python
import torch
from svgp_kan import SVGPUNet

# 1. Initialize the Probabilistic U-Net
# Input: 1 Channel (Grayscale), Output: 2 Classes (Background, Object)
model = SVGPUNet(in_channels=1, num_classes=2, base=32).cuda()

# 2. Forward Pass
# Input: [Batch, Channel, Height, Width]
images = torch.randn(4, 1, 128, 128).cuda()
logits, uncertainty = model(images)

# logits:      [4, 2, 128, 128] -> Standard segmentation mask
# uncertainty: [4, 1, 128, 128] -> Variance map (High value = High uncertainty)

# 3. Visualization
import matplotlib.pyplot as plt
plt.imshow(uncertainty[0, 0].cpu().detach(), cmap='hot')
plt.title("Model Uncertainty / Confusion Map")
plt.show()
```

### Custom CNN Integration
You can also drop GPKANLayer into any existing CNN. It automatically handles 4D tensors [B, C, H, W].

```python

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