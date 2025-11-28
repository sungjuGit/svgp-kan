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
        # log_scale handles Lengthscale (RBF) or Period (Cosine)
        self.log_scale = nn.Parameter(torch.zeros(out_features, in_features))
        # log_variance handles ARD (Pruning). Start at -1.0 to encourage sparsity.
        self.log_variance = nn.Parameter(torch.zeros(out_features, in_features) - 1.0)

    def forward(self, x):
        """
        Args:
            x: Tensor [Batch, In] (Deterministic) OR Tuple(Mean, Var) (Probabilistic)
        """
        # Handle deterministic vs probabilistic input
        if isinstance(x, torch.Tensor):
            x_mean, x_var = x, torch.zeros_like(x) + 1e-6
        else:
            x_mean, x_var = x
            x_var = torch.clamp(x_var, min=1e-6)

        # --- Clamping for Stability ---
        # Prevents "Variance Collapse" and "White Noise Overfitting"
        constrained_q_var = torch.exp(self.q_log_var).clamp(min=1e-5)
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        
        # Scale constraints depend on kernel type
        min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
        constrained_scale = torch.exp(self.log_scale).clamp(min=min_scale)

        # --- Dispatch Kernel ---
        if self.kernel_type == 'rbf':
            edge_means, edge_vars = rbf_moment_matching(
                x_mean, x_var, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var
            )
        elif self.kernel_type == 'cosine':
            edge_means, edge_vars = cosine_moment_matching(
                x_mean, x_var, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # --- Aggregation (KAN Summation) ---
        # Sum of Gaussians = Gaussian of Sums
        return edge_means.sum(dim=2), edge_vars.sum(dim=2)

    def get_relevance(self):
        """Returns the ARD variance magnitude for interpretability."""
        return torch.exp(self.log_variance).detach().cpu()
