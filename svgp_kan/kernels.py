"""
kernels.py - Moment matching for SVGP-KAN with orthogonal variance

This version includes:
- Original mean computation (unchanged)
- Projected variance (mean-field approximation)
- Orthogonal variance (Nyström approximation error)

For OOD detection:
- Uncertainty naturally saturates at prior variance (correct GP behavior)

For unbounded OOD growth, apply post-hoc scaling at the OUTPUT level
using PostHocOODScaler in the experiment script (not in kernels).
"""

import torch
import math


def rbf_moment_matching(x_mean, x_var, z, q_mu, q_var, lengthscale, variance):
    """
    Analytic moment matching for RBF Kernel with orthogonal variance correction.
    
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
    denom = ell_sq + x_var_exp
    diff = x_mu_exp - z.unsqueeze(0)
    exponent = -0.5 * (diff.pow(2) / denom)
    scale = var_kern * torch.sqrt(ell_sq / denom)
    psi_1 = scale * torch.exp(exponent)

    # --- 2. Compute Mean ---
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # --- 3. Compute Variance Components ---
    
    # Component A: Projected variance (standard mean-field)
    var_projected = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    var_projected = var_projected - edge_means.pow(2)
    var_projected = torch.clamp(var_projected, min=0)
    
    # Component B: Orthogonal variance (Nyström approximation error)
    # V_orth = k(x,x) * (1 - trace_approx)
    # When x is near inducing points: trace_approx ≈ 1, V_orth ≈ 0
    # When x is far from inducing points: trace_approx → 0, V_orth → k(x,x)
    k_xx = var_kern.squeeze(-1)  # [1, Out, In]
    trace_approx = (psi_1.pow(2)).sum(dim=-1) / (k_xx + 1e-8)
    trace_approx = torch.clamp(trace_approx, min=0.0, max=1.0)
    var_ortho = k_xx * (1.0 - trace_approx)
    
    # Total variance = projected + orthogonal
    edge_vars = var_projected + var_ortho
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)


def cosine_moment_matching(x_mean, x_var, z, q_mu, q_var, period, variance):
    """
    Analytic moment matching for Cosine Kernel with orthogonal variance correction.
    """
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)
    
    omega = (2 * math.pi) / period.unsqueeze(0).unsqueeze(-1)

    damping = torch.exp(-0.5 * omega.pow(2) * x_var_exp)
    dist = x_mu_exp - z.unsqueeze(0)
    cosine_term = torch.cos(omega * dist)
    
    psi_1 = var_kern * damping * cosine_term
    
    # Mean (unchanged)
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # Projected variance
    var_projected = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    var_projected = var_projected - edge_means.pow(2)
    var_projected = torch.clamp(var_projected, min=0)
    
    # Orthogonal variance
    k_xx = var_kern.squeeze(-1)
    trace_approx = (psi_1.pow(2)).sum(dim=-1) / (k_xx + 1e-8)
    trace_approx = torch.clamp(trace_approx, min=0.0, max=1.0)
    var_ortho = k_xx * (1.0 - trace_approx)
    
    edge_vars = var_projected + var_ortho
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)