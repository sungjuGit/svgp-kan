# svgp_kan/kanpod.py

import torch
import torch.nn as nn
from .model import GPKAN

class SVGPKanPOD(nn.Module):
    """
    SVGPKanPOD: Probabilistic Neural Operator via SVGP-KANs.
    
    A novel architecture for Reduced-Order Modeling (ROM) that replaces 
    Discrete Matrix Factorization (POD/SVD) with Continuous Operator Learning.
    
    Topology (Branch-Trunk):
        u(params, x) = Sum_k [ Branch_k(params) * Trunk_k(x) ]
    
    Where:
        Branch: SVGP-KAN mapping Physics Params -> Latent Coefficients
        Trunk:  SVGP-KAN mapping Spatial Coords -> Basis Functions
        
    Attributes:
        branch (GPKAN): The physics encoder.
        trunk (GPKAN): The spatial basis generator.
    """
    def __init__(self, param_dim, spatial_dim, latent_dim=10, 
                 branch_depth=[20, 20], trunk_depth=[20, 20], 
                 num_inducing=20, kernel='rbf'):
        super().__init__()
        
        # --- Branch Net (Physics) ---
        # Maps Parameters -> Coefficients alpha
        self.branch = GPKAN(
            layers_hidden=[param_dim] + branch_depth + [latent_dim],
            num_inducing=num_inducing,
            kernel_type=kernel
        )
        
        # --- Trunk Net (Spatial) ---
        # Maps Coordinates (x, y, t) -> Basis functions phi(x)
        self.trunk = GPKAN(
            layers_hidden=[spatial_dim] + trunk_depth + [latent_dim],
            num_inducing=num_inducing,
            kernel_type=kernel
        )

    def forward(self, params, coords):
        """
        Probabilistic Forward Pass with Analytic Moment Matching.
        
        Args:
            params: [Batch, param_dim]
            coords: [Grid_Size, spatial_dim]
            
        Returns:
            mean: [Batch, Grid_Size] - Predicted Field Mean
            var:  [Batch, Grid_Size] - Predicted Field Variance
        """
        # 1. Get distributions from both KANs
        # alpha: [Batch, Latent]
        alpha_mu, alpha_var = self.branch(params)
        
        # phi: [Grid_Size, Latent]
        phi_mu, phi_var = self.trunk(coords)
        
        # 2. Probabilistic Fusion (Dot Product of Independent Gaussians)
        # Broadcast to [Batch, Grid_Size, Latent]
        a_mu = alpha_mu.unsqueeze(1)
        a_var = alpha_var.unsqueeze(1)
        p_mu = phi_mu.unsqueeze(0)
        p_var = phi_var.unsqueeze(0)
        
        # Expectation of Product: E[XY] = E[X]E[Y]
        product_mean = a_mu * p_mu
        
        # Variance of Product: Var(XY) = Var(X)E[Y]^2 + Var(Y)E[X]^2 + Var(X)Var(Y)
        product_var = (a_var * p_mu.pow(2)) + \
                      (p_var * a_mu.pow(2)) + \
                      (a_var * p_var)
        
        # 3. Sum over Latent Dimension (The Operator Summation)
        # We assume latent modes are independent, so variances sum.
        pred_mean = product_mean.sum(dim=-1)
        pred_var = product_var.sum(dim=-1)
        
        return pred_mean, pred_var

    def compute_total_kl(self, jitter=1e-4):
        """Sum of KL divergence from both Physics and Spatial networks."""
        return self.branch.compute_total_kl(jitter) + self.trunk.compute_total_kl(jitter)