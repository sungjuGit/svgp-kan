import torch
import torch.nn as nn
from .kernels import rbf_moment_matching, cosine_moment_matching

class GPKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_inducing=20, kernel_type='rbf'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_type = kernel_type.lower()
        
        # --- Learnable Parameters ---

        # 1. Inducing Locations (z)
        # Initialized uniformly to cover the standard data range [-1, 1] with buffer
        self.z = nn.Parameter(
            torch.linspace(-1.5, 1.5, num_inducing)
            .reshape(1, 1, num_inducing)
            .repeat(out_features, in_features, 1)
        )
        
        # 2. Variational Parameters (q_mu, q_log_var)
        self.q_mu = nn.Parameter(torch.randn(out_features, in_features, num_inducing) * 0.05)
        # Initialize variance small (confident) but not zero        
        self.q_log_var = nn.Parameter(torch.ones(out_features, in_features, num_inducing) * -3.0)
        
        # 3. Kernel Hyperparameters
        self.log_scale = nn.Parameter(torch.zeros(out_features, in_features))
        self.log_variance = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        """
        Universal Forward Pass.
        Handles:
          - x: Tensor (First layer) or Tuple(Mean, Var) (Hidden layers)
          - Shapes: 
              2D (Tabular): [Batch, Features]
              3D (Sequence): [Batch, Time, Features] 
              4D (Image): [Batch, Channels, Height, Width]
        """
        
        # --- 1. Input Parsing (Tuple vs Tensor) ---
        # Guarantees x_mean and x_var are both Tensors matching input shape
        if isinstance(x, tuple):
            x_mean, x_var = x
            x_var = torch.clamp(x_var, min=1e-6)
        else:
            x_mean = x
            x_var = torch.zeros_like(x_mean) # Zero uncertainty for raw inputs

        # --- 2. Shape Handling & Flattening ---
        original_shape = x_mean.shape
        ndim = x_mean.ndim
        
        if ndim == 4:
            # CNN Case: [Batch, Channels, Height, Width]
            # Standard PyTorch image format (Channels First)
            b, c, h, w = original_shape
            
            if c != self.in_features:
                 raise ValueError(f"Input channels {c} mismatch layer features {self.in_features}")

            # Flatten spatial dims: [B, C, H, W] -> [B, H, W, C] -> [N_pixels, C]
            x_mean_flat = x_mean.permute(0, 2, 3, 1).contiguous().view(-1, c)
            x_var_flat = x_var.permute(0, 2, 3, 1).contiguous().view(-1, c)
            
        elif ndim > 2:
            # Sequence/Video Case: [Batch, ..., Features]
            # Assumes Features are LAST (Standard RNN/Transformer format)
            if original_shape[-1] != self.in_features:
                 raise ValueError(f"Last dim {original_shape[-1]} mismatch layer features {self.in_features}")
                 
            # Flatten everything except the last feature dimension
            x_mean_flat = x_mean.reshape(-1, self.in_features)
            x_var_flat = x_var.reshape(-1, self.in_features)
            
        else:
            # Standard Tabular: [Batch, Features] (Legacy Mode)
            x_mean_flat = x_mean
            x_var_flat = x_var

        # --- 3. Core GP Logic (Unchanged) ---

        # --- Clamping for Stability ---
        constrained_q_var = torch.exp(self.q_log_var).clamp(min=1e-5)
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)

        # Scale constraints depend on kernel type        
        min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
        constrained_scale = torch.exp(self.log_scale).clamp(min=min_scale)

        # --- Dispatch Kernel ---
        if self.kernel_type == 'rbf':
            edge_means, edge_vars = rbf_moment_matching(
                x_mean_flat, x_var_flat, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var
            )
        elif self.kernel_type == 'cosine':
            edge_means, edge_vars = cosine_moment_matching(
                x_mean_flat, x_var_flat, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Aggregation (KAN Summation)
        y_mean_flat = edge_means.sum(dim=2) 
        y_var_flat = edge_vars.sum(dim=2)

        # --- 4. Output Reshaping (Restore Spatial Structure) ---
        if ndim == 4:
            # [N_pixels, Out_C] -> [B, H, W, Out_C] -> [B, Out_C, H, W]
            b, c, h, w = original_shape
            
            y_mean = y_mean_flat.view(b, h, w, self.out_features).permute(0, 3, 1, 2)
            y_var = y_var_flat.view(b, h, w, self.out_features).permute(0, 3, 1, 2)
            
        elif ndim > 2:
            # Restore Generic Shape [Batch, ..., Out_Features]
            out_shape = original_shape[:-1] + (self.out_features,)
            y_mean = y_mean_flat.reshape(out_shape)
            y_var = y_var_flat.reshape(out_shape)
            
        else:
            # Standard Tabular
            y_mean = y_mean_flat
            y_var = y_var_flat

        return y_mean, y_var

    def get_relevance(self):
        """
        Returns the Automatic Relevance Determination (ARD) scores.
        High value = Important feature. Low value = Pruned.
        """
        return 1.0 / torch.exp(self.log_scale).mean(dim=0)