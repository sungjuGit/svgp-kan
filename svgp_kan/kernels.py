import torch
import math

def rbf_moment_matching(x_mean, x_var, z, q_mu, q_var, lengthscale, variance):
    """
    Analytic moment matching for RBF Kernel with Orthogonal Variance correction.
    
    Backward compatible with existing SVGP-KAN code.
    """
    # 1. Broadcast inputs to match edge dimensions [Batch, Out, In, Inducing]
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    
    # Kernel parameters broadcast
    ell_sq = lengthscale.unsqueeze(0).unsqueeze(-1).pow(2)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)

    # 2. Compute Psi_1 (Expected Kernel Matrix)
    # Denominator handles input uncertainty blurring the kernel
    denom = ell_sq + x_var_exp
    
    # Exponent term: -(x-z)^2 / (2 * (L^2 + var))
    diff = x_mu_exp - z.unsqueeze(0)
    exponent = -0.5 * (diff.pow(2) / denom)
    
    # Scaling factor
    scale = var_kern * torch.sqrt(ell_sq / denom)
    
    psi_1 = scale * torch.exp(exponent)

    # 3. Compute Output Moments
    # Mean is unchanged
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # --- VARIANCE CALCULATION ---
    
    # Term A: Projected Variance (The "Inside" Uncertainty)
    # This captures uncertainty transmitted through the inducing points.
    # It tends to go to 0 as we move far away from Z.
    var_projected = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1) - edge_means.pow(2)
    
    # Term B: Orthogonal Variance (The "Outside" Uncertainty)
    # This captures the information lost by the sparse approximation.
    # It tends to 'variance' as we move far away from Z.
    
    # We approximate the trace term k(x,Z)K(Z,Z)^-1 k(Z,x) using psi_1 squares
    # This is a standard approximation in sparse GPs (Titsias 2009)
    trace_approx = (psi_1.pow(2)).sum(dim=-1) / (variance.unsqueeze(0) + 1e-8)
    trace_approx = torch.clamp(trace_approx, min=0.0, max=1.0)
    
    k_xx = variance.unsqueeze(0) # The maximum possible variance (signal variance)
    var_ortho = k_xx * (1.0 - trace_approx)
    
    # Total Variance
    # Even if kl_weight=0, var_ortho ensures we don't collapse to zero far from data.
    edge_vars = var_projected + var_ortho
    
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
    
    # Cosine arguments
    arg = omega * dist
    
    # Psi_1 calculation for Cosine
    psi_1 = var_kern * damping * torch.cos(arg)

    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # Simplified variance for cosine (assuming mostly projected)
    edge_vars = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    edge_vars = edge_vars - edge_means.pow(2)
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)