import torch
import math

def rbf_moment_matching(x_mean, x_var, z, q_mu, q_var, lengthscale, variance):
    """
    Analytic moment matching for RBF Kernel.
    Propagates Gaussian distributions N(x_mean, x_var) through univariate GPs
    defined by inducing points z and variational parameters q_mu, q_var.
    """
    # Broadcast inputs to match edge dimensions [Batch, Out, In, Inducing]
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    
    # Kernel parameters broadcast
    ell_sq = lengthscale.unsqueeze(0).unsqueeze(-1).pow(2)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)

    # --- 1. Compute Psi_1 (Expected Kernel Matrix) ---
    # Denominator handles input uncertainty blurring the kernel
    denom = ell_sq + x_var_exp
    
    # Exponent term: -(x-z)^2 / (2 * (L^2 + var))
    diff = x_mu_exp - z.unsqueeze(0)
    exponent = -0.5 * (diff.pow(2) / denom)
    
    # Scaling factor
    scale = var_kern * torch.sqrt(ell_sq / denom)
    
    psi_1 = scale * torch.exp(exponent)

    # --- 2. Compute Output Moments ---
    # Mean = Psi_1 * weights
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # Variance (Mean-Field Approx)
    # Var(y) = E[y^2] - E[y]^2
    # We use the independent variance approximation for speed and stability
    edge_vars = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    edge_vars = edge_vars - edge_means.pow(2)
    
    # Clamp for numerical stability
    return edge_means, torch.clamp(edge_vars, min=1e-6)

def cosine_moment_matching(x_mean, x_var, z, q_mu, q_var, period, variance):
    """
    Analytic moment matching for Cosine Kernel.
    Used for discovering periodic patterns.
    """
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)
    
    omega = (2 * math.pi) / period.unsqueeze(0).unsqueeze(-1)

    # Damping due to input uncertainty (Uncertain inputs kill the oscillation)
    damping = torch.exp(-0.5 * omega.pow(2) * x_var_exp)
    dist = x_mu_exp - z.unsqueeze(0)
    cosine_term = torch.cos(omega * dist)
    
    psi_1 = var_kern * damping * cosine_term
    
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # Mean-Field Variance
    edge_vars = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    edge_vars = edge_vars - edge_means.pow(2)
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)
