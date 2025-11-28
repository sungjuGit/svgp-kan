# svgp_kan/__init__.py

__version__ = "0.1.0"
__author__ = "Open Source Contributor"

# 1. Expose the High-Level API
from .regressor import GPKANRegressor

# 2. Expose the Low-Level Components
from .model import GPKAN, gaussian_nll_loss
from .layers import GPKANLayer

# 3. Define what gets imported with "from svgp_kan import *"
__all__ = [
    "GPKANRegressor",
    "GPKAN",
    "GPKANLayer",
    "gaussian_nll_loss"
]
