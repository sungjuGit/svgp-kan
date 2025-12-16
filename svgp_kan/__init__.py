# svgp_kan/__init__.py

__version__ = "0.4.2"  # Bumped for orthogonal variance normalization bugfix
__author__ = "sungjuGit"

# 1. Expose the High-Level API
from .regressor import GPKANRegressor

# 2. Expose the Low-Level Components
from .model import GPKAN, gaussian_nll_loss
from .layers import GPKANLayer
from .kl_divergence import compute_kl_divergence
from .kernels import rbf_moment_matching, check_mean_field_assumptions
from .training_utils import (
    EarlyStopping, ModelCheckpoint, KLAnnealingSchedule, 
    GradientClipper, create_lr_scheduler
)
from .svgpkanpod import SVGPKanPOD, train_svgpkanpod


# 3. Expose Vision Module
from .unet import SVGPUNet, SVGPUNet_Fluid

# 4. Define what gets imported with "from svgp_kan import *"
__all__ = [
    "GPKANRegressor",
    "GPKAN",
    "GPKANLayer",
    "SVGPUNet",
    "SVGPUNet_Fluid",
    "SVGPKanPOD",
    "gaussian_nll_loss",
    "compute_kl_divergence",
    "rbf_moment_matching",
    "check_mean_field_assumptions",
]
