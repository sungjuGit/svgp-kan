"""
kernels.py - Moment matching for SVGP-KAN with orthogonal variance

This version includes:
- Original mean computation (unchanged)
- Projected variance (mean-field approximation)
- Orthogonal variance (Nyström approximation error) with CORRECTED normalization

For OOD detection:
- Uncertainty naturally saturates at prior variance (correct GP behavior)

The orthogonal variance formula is derived from:
    σ²_⊥ = ψ₀ - tr(K_ZZ⁻¹ Ψ₂)

Under mean-field approximation (K_ZZ ≈ σ_f² I, Ψ₂ ≈ ψ₁ψ₁ᵀ):
    σ²_⊥ ≈ σ_f² - ||ψ₁||² / σ_f² = σ_f² (1 - ||ψ₁||² / σ_f⁴)

The normalized trace ||ψ₁||² / σ_f⁴ ∈ [0,1] represents the fraction of 
prior variance explained by the inducing points.

BUGFIX (v0.4.2): Previous versions incorrectly normalized by σ_f² instead of σ_f⁴,
which caused trace_approx to exceed 1 whenever σ_f² > 1, leading to 
underestimated orthogonal variance.
"""

import torch
import math


def rbf_moment_matching(x_mean, x_var, z, q_mu, q_var, lengthscale, variance,
                        return_diagnostics=False):
    """
    Analytic moment matching for RBF Kernel with orthogonal variance correction.
    
    Propagates Gaussian distributions N(x_mean, x_var) through univariate GPs
    defined by inducing points z and variational parameters q_mu, q_var.
    
    Args:
        x_mean: [Batch, In] input means
        x_var: [Batch, In] input variances
        z: [Out, In, M] inducing point locations
        q_mu: [Out, In, M] variational means
        q_var: [Out, In, M] variational variances
        lengthscale: [Out, In] kernel lengthscales
        variance: [Out, In] kernel signal variances (σ_f²)
        return_diagnostics: if True, also return diagnostic dict
        
    Returns:
        edge_means: [Batch, Out, In] output means
        edge_vars: [Batch, Out, In] output variances
        diagnostics: (optional) dict with approximation quality metrics
    """
    # Broadcast inputs to match edge dimensions [Batch, Out, In, Inducing]
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    
    # Kernel parameters broadcast
    ell_sq = lengthscale.unsqueeze(0).unsqueeze(-1).pow(2)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)  # This is σ_f²

    # --- 1. Compute Psi_1 (Expected Kernel Matrix) ---
    # ψ₁,j = σ_f² √(ℓ²/(ℓ²+σ_x²)) exp(-(Z_j - μ_x)² / (2(ℓ²+σ_x²)))
    denom = ell_sq + x_var_exp
    diff = x_mu_exp - z.unsqueeze(0)
    exponent = -0.5 * (diff.pow(2) / denom)
    scale = var_kern * torch.sqrt(ell_sq / denom)
    psi_1 = scale * torch.exp(exponent)

    # --- 2. Compute Mean ---
    # μ_f = Σ_j ψ₁,j · m_j  (under mean-field with normalized variational params)
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # --- 3. Compute Variance Components ---
    
    # Component A: Projected variance (standard mean-field)
    # σ²_proj = Σ_j ψ₁,j² (s_j + m_j²) - μ_f²
    var_projected = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    var_projected = var_projected - edge_means.pow(2)
    var_projected = torch.clamp(var_projected, min=0)
    
    # Component B: Orthogonal variance (Nyström approximation error)
    # 
    # Exact: σ²_⊥ = ψ₀ - tr(K_ZZ⁻¹ Ψ₂)
    # 
    # Mean-field approximation (K_ZZ ≈ σ_f² I, Ψ₂ ≈ ψ₁ψ₁ᵀ):
    #   tr(K_ZZ⁻¹ Ψ₂) ≈ tr((1/σ_f²)I · ψ₁ψ₁ᵀ) = ||ψ₁||² / σ_f²
    #   σ²_⊥ ≈ σ_f² - ||ψ₁||² / σ_f² = σ_f² (1 - ||ψ₁||² / σ_f⁴)
    #
    # The normalized trace ||ψ₁||² / σ_f⁴ should be in [0, 1]:
    #   - At inducing point: ψ₁,j ≈ σ_f², so ||ψ₁||² ≈ σ_f⁴, trace ≈ 1, σ²_⊥ ≈ 0
    #   - Far from data: ψ₁,j → 0, trace → 0, σ²_⊥ → σ_f²
    
    k_xx = var_kern.squeeze(-1)  # [1, Out, In], this is σ_f²
    
    # CORRECTED: Normalize by σ_f⁴ (k_xx²), not σ_f² (k_xx)
    # Previous bug: trace_approx = psi_1.pow(2).sum(-1) / k_xx  <-- WRONG
    psi1_sq_sum = psi_1.pow(2).sum(dim=-1)  # ||ψ₁||²
    k_xx_sq = k_xx.pow(2)  # σ_f⁴
    
    trace_approx_raw = psi1_sq_sum / (k_xx_sq + 1e-8)
    
    # Clamp to [0, 1] - values outside indicate approximation breakdown
    trace_approx = torch.clamp(trace_approx_raw, min=0.0, max=1.0)
    var_ortho = k_xx * (1.0 - trace_approx)
    
    # Total variance = projected + orthogonal
    edge_vars = var_projected + var_ortho
    
    if return_diagnostics:
        diagnostics = {
            'trace_raw_mean': trace_approx_raw.mean().item(),
            'trace_raw_max': trace_approx_raw.max().item(),
            'trace_raw_min': trace_approx_raw.min().item(),
            'n_clamped_high': (trace_approx_raw > 1.0).sum().item(),
            'n_clamped_low': (trace_approx_raw < 0.0).sum().item(),
            'max_clamp_excess': (trace_approx_raw - 1.0).clamp(min=0).max().item(),
            'var_ortho_mean': var_ortho.mean().item(),
            'var_projected_mean': var_projected.mean().item(),
            'psi1_max': psi_1.max().item(),
            'k_xx_mean': k_xx.mean().item(),
        }
        return edge_means, torch.clamp(edge_vars, min=1e-6), diagnostics
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)


def cosine_moment_matching(x_mean, x_var, z, q_mu, q_var, period, variance,
                           return_diagnostics=False):
    """
    Analytic moment matching for Cosine Kernel with orthogonal variance correction.
    
    Uses the same corrected normalization as rbf_moment_matching.
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
    
    # Orthogonal variance with CORRECTED normalization
    k_xx = var_kern.squeeze(-1)
    k_xx_sq = k_xx.pow(2)
    psi1_sq_sum = psi_1.pow(2).sum(dim=-1)
    
    trace_approx_raw = psi1_sq_sum / (k_xx_sq + 1e-8)
    trace_approx = torch.clamp(trace_approx_raw, min=0.0, max=1.0)
    var_ortho = k_xx * (1.0 - trace_approx)
    
    edge_vars = var_projected + var_ortho
    
    if return_diagnostics:
        diagnostics = {
            'trace_raw_mean': trace_approx_raw.mean().item(),
            'trace_raw_max': trace_approx_raw.max().item(),
            'n_clamped_high': (trace_approx_raw > 1.0).sum().item(),
            'max_clamp_excess': (trace_approx_raw - 1.0).clamp(min=0).max().item(),
        }
        return edge_means, torch.clamp(edge_vars, min=1e-6), diagnostics
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)


# =============================================================================
# Diagnostic utilities for monitoring approximation quality
# =============================================================================

def check_mean_field_assumptions(z, lengthscale, variance, x_var=None, verbose=True):
    """
    Check whether the mean-field approximation assumptions are satisfied.
    
    Assumptions:
    1. Inducing points are well-separated: |Z_i - Z_j| > 2ℓ for i ≠ j
    2. Input variance is small: σ_x² < ℓ²
    3. Signal variance is reasonable: σ_f² ∈ [0.1, 10] typical
    
    Args:
        z: [Out, In, M] inducing point locations
        lengthscale: [Out, In] kernel lengthscales
        variance: [Out, In] kernel signal variances
        x_var: [Batch, In] input variances (optional)
        verbose: print diagnostic messages
        
    Returns:
        dict with diagnostic information
    """
    out_features, in_features, M = z.shape
    
    diagnostics = {
        'separation_ok': True,
        'input_var_ok': True,
        'variance_ok': True,
        'details': []
    }
    
    # Check inducing point separation
    min_separations = []
    for i in range(out_features):
        for j in range(in_features):
            z_edge = z[i, j]  # [M]
            ell = lengthscale[i, j].item()
            
            # Pairwise distances
            dists = (z_edge.unsqueeze(0) - z_edge.unsqueeze(1)).abs()
            # Mask diagonal
            mask = ~torch.eye(M, dtype=torch.bool, device=z.device)
            if mask.sum() > 0:
                min_sep = dists[mask].min().item()
                min_separations.append(min_sep / ell)
                
                if min_sep < 2 * ell:
                    diagnostics['separation_ok'] = False
                    diagnostics['details'].append(
                        f"Edge ({i},{j}): min separation = {min_sep:.3f}, "
                        f"threshold = {2*ell:.3f} (2ℓ)"
                    )
    
    if min_separations:
        diagnostics['min_separation_ratio'] = min(min_separations)
        diagnostics['mean_separation_ratio'] = sum(min_separations) / len(min_separations)
    
    # Check signal variance
    var_min = variance.min().item()
    var_max = variance.max().item()
    diagnostics['variance_range'] = (var_min, var_max)
    
    if var_max > 10 or var_min < 0.01:
        diagnostics['variance_ok'] = False
        diagnostics['details'].append(
            f"Signal variance outside typical range: [{var_min:.3f}, {var_max:.3f}]"
        )
    
    # Check input variance if provided
    if x_var is not None:
        ell_sq = lengthscale.pow(2)
        # Average check across edges
        x_var_mean = x_var.mean(dim=0)  # [In]
        
        for j in range(in_features):
            ell_sq_avg = ell_sq[:, j].mean().item()
            x_var_j = x_var_mean[j].item()
            
            if x_var_j > ell_sq_avg:
                diagnostics['input_var_ok'] = False
                diagnostics['details'].append(
                    f"Input {j}: σ_x² = {x_var_j:.4f} > ℓ² = {ell_sq_avg:.4f}"
                )
    
    diagnostics['all_ok'] = (
        diagnostics['separation_ok'] and 
        diagnostics['input_var_ok'] and 
        diagnostics['variance_ok']
    )
    
    if verbose:
        status = "✓ PASS" if diagnostics['all_ok'] else "✗ FAIL"
        print(f"Mean-field approximation check: {status}")
        if diagnostics['details']:
            for detail in diagnostics['details']:
                print(f"  - {detail}")
        if 'min_separation_ratio' in diagnostics:
            print(f"  Min separation ratio (want > 2): {diagnostics['min_separation_ratio']:.2f}")
    
    return diagnostics
