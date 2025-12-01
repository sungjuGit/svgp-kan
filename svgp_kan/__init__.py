# svgp_kan/__init__.py

__version__ = "0.2.0"  # Bumped for Vision Support
__author__ = "Open Source Contributor"

# 1. Expose the High-Level API
from .regressor import GPKANRegressor

# 2. Expose the Low-Level Components
from .model import GPKAN, gaussian_nll_loss
from .layers import GPKANLayer

# 3. Expose Vision Module
from .unet import SVGPUNet, SVGPUNet_Fluid

# 4. Define what gets imported with "from svgp_kan import *"
__all__ = [
    "GPKANRegressor",
    "GPKAN",
    "GPKANLayer",
    "SVGPUNet",
    "SVGPUNet_Fluid",
    "gaussian_nll_loss"
]