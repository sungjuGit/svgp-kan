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
   git clone \[https://github.com/yourusername/svgp-kan.git\](https://github.com/yourusername/svgp-kan.git)  
   cd svgp-kan

2. Install the library (automatically installs PyTorch dependencies):  
   pip install .

## **Quick Start**

### **1\. Scientific Discovery (Feature Selection)**

Train a model to identify which inputs matter and which are noise.

import numpy as np  
from svgp\_kan import GPKANRegressor

\# 1\. Generate Data (Sine \+ Linear \+ Noise)  
X \= np.random.rand(1000, 3\)  
y \= np.sin(X\[:, 0\]\*5) \+ 2\*X\[:, 1\] \+ 0\*X\[:, 2\] \# x2 is irrelevant noise

\# 2\. Train Scientist Model  
\# We use RBF kernels as universal approximators  
model \= GPKANRegressor(hidden\_layers=\[3, 5, 1\], kernel='rbf')  
model.fit(X, y, epochs=500, sparsity\_weight=0.05)

\# 3\. Interpret Results  
model.explain()  
\# Output:  
\# Feature 0: \[ACTIVE\] Type: Non-Linear (Sine)  
\# Feature 1: \[ACTIVE\] Type: Linear (Large Lengthscale)  
\# Feature 2: \[PRUNED\] (Irrelevant)

\# 4\. Predict with Uncertainty  
mu, std \= model.predict(X)

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
