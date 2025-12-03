# svgp_kan/__init__.py

__version__ = "0.3.0"  # Bumped for major bug fixes
__author__ = "sungju"

# 1. Expose the High-Level API
from .regressor import GPKANRegressor

# 2. Expose the Low-Level Components
from .model import GPKAN, gaussian_nll_loss
from .layers import GPKANLayer
from .kl_divergence import compute_kl_divergence
from .kernels import rbf_moment_matching

# 3. Expose Vision Module
from .unet import SVGPUNet, SVGPUNet_Fluid

# 4. Define what gets imported with "from svgp_kan import *"
__all__ = [
    "GPKANRegressor",
    "GPKAN",
    "GPKANLayer",
    "SVGPUNet",
    "SVGPUNet_Fluid",
    "gaussian_nll_loss",
    "compute_kl_divergence",
    "rbf_moment_matching",
]