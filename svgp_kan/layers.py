"""
layers.py - SVGP-KAN Layer implementation (v0.4.4)

Each layer implements a Sparse Variational GP on each edge of a KAN,
enabling probabilistic inference with uncertainty propagation.

MODES:
- strict_mean_field=False (DEFAULT): Uses clamping heuristic for orthogonal variance.
  Backward compatible. Works well in practice even when approximation assumptions
  are not strictly satisfied. Inducing points and lengthscales are fully learnable.
  
- strict_mean_field=True: Enforces mean-field approximation validity DURING TRAINING:
  1. Inducing points are FIXED (not learnable) - evenly spaced in [z_min, z_max]
  2. Lengthscales are BOUNDED above by spacing/2 - guarantees separation
  
  This sacrifices some flexibility in exchange for guaranteed mathematical validity.
"""

import torch
import torch.nn as nn
import warnings
from .kernels import (
    rbf_moment_matching, 
    cosine_moment_matching, 
    check_mean_field_assumptions,
    check_mean_field_validity,
    compute_recommended_num_inducing
)


class GPKANLayer(nn.Module):
    """
    A single SVGP-KAN layer.
    
    Each edge (i,j) is an independent univariate GP with:
    - Inducing points z[i,j,:]
    - Variational mean q_mu[i,j,:]  
    - Variational variance exp(q_log_var[i,j,:])
    - Kernel lengthscale exp(log_scale[i,j])
    - Kernel signal variance exp(log_variance[i,j])
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_inducing: Number of inducing points per edge
        kernel_type: 'rbf' or 'cosine'
        strict_mean_field: If True, enforce mean-field approximation:
            - Inducing points are FIXED (not learnable)
            - Lengthscales are BOUNDED by max_lengthscale = spacing / 2
            Default False (backward compatible).
        z_min: Minimum inducing point location (default: -1.5)
        z_max: Maximum inducing point location (default: 1.5)
        initial_lengthscale: Initial lengthscale (default: 1.0). In strict mode,
            this is clamped to max_lengthscale if necessary.
    """
    
    def __init__(self, in_features, out_features, num_inducing=20, kernel_type='rbf',
                 strict_mean_field=False, z_min=-1.5, z_max=1.5, initial_lengthscale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_type = kernel_type.lower()
        self.strict_mean_field = strict_mean_field
        self.z_min = z_min
        self.z_max = z_max
        self.num_inducing = num_inducing
        
        # Store original request for reference
        self._requested_num_inducing = num_inducing
        
        # Compute spacing and constraints
        domain_size = z_max - z_min
        self.fixed_spacing = domain_size / max(num_inducing - 1, 1)
        
        # Maximum lengthscale for mean-field validity: spacing / 2
        self.max_lengthscale = self.fixed_spacing / 2.0
        self.min_lengthscale = 0.01  # Minimum for numerical stability
        
        if strict_mean_field:
            # Print configuration info
            self._print_strict_mode_info(initial_lengthscale)
            
            # FIXED inducing points - register as buffer, not parameter
            z_fixed = (
                torch.linspace(z_min, z_max, num_inducing)
                .reshape(1, 1, num_inducing)
                .repeat(out_features, in_features, 1)
            )
            self.register_buffer('z', z_fixed)
            
            # Clamp initial lengthscale to valid range
            init_ls = min(initial_lengthscale, self.max_lengthscale * 0.9)
            init_ls = max(init_ls, self.min_lengthscale)
        else:
            # LEARNABLE inducing points (original behavior)
            self.z = nn.Parameter(
                torch.linspace(z_min, z_max, num_inducing)
                .reshape(1, 1, num_inducing)
                .repeat(out_features, in_features, 1)
            )
            init_ls = initial_lengthscale
        
        # Variational parameters
        self.q_mu = nn.Parameter(torch.randn(out_features, in_features, num_inducing) * 0.05)
        self.q_log_var = nn.Parameter(torch.ones(out_features, in_features, num_inducing) * -3.0)
        
        # Kernel hyperparameters (in log space for positivity)
        # Initialize log_scale based on initial_lengthscale
        init_log_scale = torch.log(torch.tensor(init_ls))
        self.log_scale = nn.Parameter(torch.ones(out_features, in_features) * init_log_scale)
        self.log_variance = nn.Parameter(torch.zeros(out_features, in_features) - 1.0)
        
        # For strict mode, we use a raw parameter that gets transformed
        if strict_mean_field:
            # Store raw parameter for sigmoid transformation
            # Initialize so that sigmoid(raw) * range + min = init_ls
            range_ls = self.max_lengthscale - self.min_lengthscale
            target_sigmoid = (init_ls - self.min_lengthscale) / range_ls
            target_sigmoid = max(0.01, min(0.99, target_sigmoid))  # Clamp for valid logit
            init_raw = torch.log(torch.tensor(target_sigmoid / (1 - target_sigmoid)))
            self.log_scale_raw = nn.Parameter(torch.ones(out_features, in_features) * init_raw)
        
        # Diagnostic tracking
        self._last_diagnostics = None
        self._mean_field_warning_issued = False
    
    def _print_strict_mode_info(self, initial_lengthscale):
        """Print information about strict mean-field configuration."""
        print(f"\n{'='*70}")
        print(f"STRICT MEAN-FIELD MODE ENABLED")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Inducing points: {self.num_inducing} (FIXED, not learnable)")
        print(f"  - Domain: [{self.z_min:.2f}, {self.z_max:.2f}]")
        print(f"  - Fixed spacing: {self.fixed_spacing:.4f}")
        print(f"  - Max lengthscale: {self.max_lengthscale:.4f} (= spacing / 2)")
        print(f"  - Min lengthscale: {self.min_lengthscale:.4f}")
        if initial_lengthscale > self.max_lengthscale:
            print(f"  - Initial lengthscale {initial_lengthscale:.4f} → clamped to {self.max_lengthscale * 0.9:.4f}")
        else:
            print(f"  - Initial lengthscale: {initial_lengthscale:.4f}")
        print(f"\nGuarantees:")
        print(f"  ✓ Separation ratio ≥ 1.0 (spacing / (2 × lengthscale))")
        print(f"  ✓ Mean-field approximation mathematically valid")
        print(f"  ✓ No clamping needed in orthogonal variance")
        print(f"{'='*70}\n")

    def get_constrained_scale(self):
        """
        Get the constrained lengthscale.
        
        In strict mode: bounded to [min_lengthscale, max_lengthscale]
        In default mode: just clamped to minimum for stability
        
        Returns:
            Tensor of constrained lengthscales [out_features, in_features]
        """
        if self.strict_mean_field:
            # Use sigmoid to bound between min and max
            sigmoid_val = torch.sigmoid(self.log_scale_raw)
            range_ls = self.max_lengthscale - self.min_lengthscale
            constrained = self.min_lengthscale + sigmoid_val * range_ls
            return constrained
        else:
            # Original behavior: just clamp to minimum
            min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
            return torch.exp(self.log_scale).clamp(min=min_scale)

    def forward(self, x, return_diagnostics=False):
        """
        Forward pass with uncertainty propagation.
        
        Args:
            x: Either a tensor (deterministic input) or tuple (mean, var)
            return_diagnostics: If True, also return approximation diagnostics
            
        Returns:
            (y_mean, y_var): Output mean and variance tensors
            diagnostics: (optional) Dict with approximation quality metrics
        """
        # --- 1. Input Parsing ---
        if isinstance(x, torch.Tensor):
            x_mean, x_var = x, torch.zeros_like(x) + 1e-6
        else:
            x_mean, x_var = x
            x_var = torch.clamp(x_var, min=1e-6)

        original_shape = x_mean.shape
        ndim = x_mean.ndim

        # --- 2. Universal Shape Handling ---
        if ndim == 2:
            x_mean_flat = x_mean
            x_var_flat = x_var
        elif ndim == 4:
            # 4D Vision Support [B, C, H, W]
            b, c, h, w = original_shape
            if c != self.in_features: 
                raise ValueError(f"Channel mismatch: {c} vs {self.in_features}")
            x_mean_flat = x_mean.permute(0, 2, 3, 1).contiguous().view(-1, c)
            x_var_flat = x_var.permute(0, 2, 3, 1).contiguous().view(-1, c)
        else:
            x_mean_flat = x_mean.reshape(-1, self.in_features)
            x_var_flat = x_var.reshape(-1, self.in_features)

        # --- 3. Core GP Logic ---
        constrained_q_var = torch.exp(self.q_log_var).clamp(min=1e-5)
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        constrained_scale = self.get_constrained_scale()

        # Verify mean-field in strict mode (should always pass now)
        if self.strict_mean_field and not self._mean_field_warning_issued:
            is_valid, min_ratio, _ = check_mean_field_validity(self.z, constrained_scale)
            if not is_valid:
                # This should not happen with the new implementation
                warnings.warn(
                    f"GPKANLayer: Unexpected mean-field violation in strict mode. "
                    f"Separation ratio = {min_ratio:.2f}. This is a bug - please report.",
                    UserWarning
                )
                self._mean_field_warning_issued = True

        if self.kernel_type == 'rbf':
            result = rbf_moment_matching(
                x_mean_flat, x_var_flat, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var,
                return_diagnostics=return_diagnostics
            )
        elif self.kernel_type == 'cosine':
            result = cosine_moment_matching(
                x_mean_flat, x_var_flat, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var,
                return_diagnostics=return_diagnostics
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        if return_diagnostics:
            edge_means, edge_vars, diagnostics = result
            self._last_diagnostics = diagnostics
        else:
            edge_means, edge_vars = result
            diagnostics = None

        # Aggregation across input dimensions
        y_mean_flat = edge_means.sum(dim=2) 
        y_var_flat = edge_vars.sum(dim=2)

        # --- 4. Output Reshaping ---
        if ndim == 2:
            y_mean, y_var = y_mean_flat, y_var_flat
        elif ndim == 4:
            y_mean = y_mean_flat.view(b, h, w, self.out_features).permute(0, 3, 1, 2)
            y_var = y_var_flat.view(b, h, w, self.out_features).permute(0, 3, 1, 2)
        else:
            out_shape = original_shape[:-1] + (self.out_features,)
            y_mean = y_mean_flat.reshape(out_shape)
            y_var = y_var_flat.reshape(out_shape)

        if return_diagnostics:
            return (y_mean, y_var), diagnostics
        return y_mean, y_var

    def compute_kl(self, jitter=1e-4):
        """
        Compute KL divergence KL[q(u)||p(u)] for this layer.
        
        This is the proper Bayesian regularization term for variational inference.
        
        Returns:
            kl: scalar KL divergence
        """
        try:
            from .kl_divergence import compute_kl_divergence
        except ImportError:
            warnings.warn(
                "kl_divergence module not found. KL term will be zero. "
                "To enable proper Bayesian inference, ensure kl_divergence.py is in the package."
            )
            return torch.tensor(0.0, device=self.q_mu.device)
        
        # Get constrained parameters (same as used in forward pass)
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        constrained_scale = self.get_constrained_scale()
        
        if self.kernel_type == 'rbf':
            kl = compute_kl_divergence(
                q_mu=self.q_mu,
                q_log_var=self.q_log_var,
                z=self.z,
                lengthscale=constrained_scale,
                variance=constrained_var,
                jitter=jitter
            )
        else:
            warnings.warn(f"KL divergence not implemented for kernel type '{self.kernel_type}'. Returning zero.")
            kl = torch.tensor(0.0, device=self.q_mu.device)
        
        return kl

    def get_relevance(self):
        """Returns the ARD variance magnitude for feature selection."""
        return torch.exp(self.log_variance)
    
    def get_lengthscale_info(self):
        """
        Get information about current lengthscales.
        
        Returns:
            dict with 'current', 'min_allowed', 'max_allowed', 'is_constrained'
        """
        constrained_scale = self.get_constrained_scale()
        return {
            'current': constrained_scale.detach(),
            'min': constrained_scale.min().item(),
            'max': constrained_scale.max().item(),
            'mean': constrained_scale.mean().item(),
            'min_allowed': self.min_lengthscale,
            'max_allowed': self.max_lengthscale if self.strict_mean_field else float('inf'),
            'is_constrained': self.strict_mean_field
        }
    
    def check_approximation(self, x_var=None, verbose=True):
        """
        Check whether mean-field approximation assumptions are satisfied.
        
        Args:
            x_var: [Batch, In] input variances (optional)
            verbose: Print diagnostic messages
            
        Returns:
            dict with diagnostic information
        """
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        constrained_scale = self.get_constrained_scale()
        
        result = check_mean_field_assumptions(
            self.z, constrained_scale, constrained_var, x_var, verbose
        )
        
        # Add lengthscale info
        if verbose:
            ls_info = self.get_lengthscale_info()
            print(f"Lengthscale range: [{ls_info['min']:.4f}, {ls_info['max']:.4f}]")
            if self.strict_mean_field:
                print(f"Bounded to: [{ls_info['min_allowed']:.4f}, {ls_info['max_allowed']:.4f}]")
        
        return result
    
    def get_last_diagnostics(self):
        """Return diagnostics from last forward pass (if computed)."""
        return self._last_diagnostics
    
    def get_recommended_num_inducing(self):
        """
        Get recommended number of inducing points based on current lengthscales.
        
        Returns:
            recommended: Recommended number of inducing points
            current: Current number of inducing points
        """
        constrained_scale = self.get_constrained_scale()
        
        # Use minimum lengthscale across all edges
        min_lengthscale = constrained_scale.min().item()
        
        recommended = compute_recommended_num_inducing(
            self.z_min, self.z_max, min_lengthscale, min_separation_factor=2.0
        )
        
        return recommended, self.num_inducing
    
    def reset_mean_field_warning(self):
        """Reset the mean-field warning flag (useful for re-checking after adjustments)."""
        self._mean_field_warning_issued = False
    
    def is_inducing_points_learnable(self):
        """Check if inducing points are learnable."""
        return isinstance(self.z, nn.Parameter)
