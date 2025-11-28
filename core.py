import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# ==============================================================================
# PART 1: KERNELS
# ==============================================================================

def rbf_moment_matching(x_mean, x_var, z, q_mu, q_var, lengthscale, variance):
    """
    Analytic moment matching for RBF Kernel.
    Propagates Gaussian distributions through univariate GPs.
    """
    # Broadcast inputs to match edge dimensions [Batch, Out, In, Inducing]
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    
    # Kernel parameters
    ell_sq = lengthscale.unsqueeze(0).unsqueeze(-1).pow(2)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)

    # 1. Compute Psi_1 (Expected Kernel Matrix)
    # Denominator handles input uncertainty blurring the kernel
    denom = ell_sq + x_var_exp
    
    # Exponent term: -(x-z)^2 / (2 * (L^2 + var))
    diff = x_mu_exp - z.unsqueeze(0)
    exponent = -0.5 * (diff.pow(2) / denom)
    
    # Scaling factor
    scale = var_kern * torch.sqrt(ell_sq / denom)
    
    psi_1 = scale * torch.exp(exponent)

    # 2. Compute Output Moments
    # Mean = Psi_1 * weights
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # Variance (Mean-Field Approx)
    edge_vars = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    edge_vars = edge_vars - edge_means.pow(2)
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)

def cosine_moment_matching(x_mean, x_var, z, q_mu, q_var, period, variance):
    """Analytic moment matching for Cosine Kernel."""
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)
    
    omega = (2 * math.pi) / period.unsqueeze(0).unsqueeze(-1)

    # Damping due to input uncertainty
    damping = torch.exp(-0.5 * omega.pow(2) * x_var_exp)
    dist = x_mu_exp - z.unsqueeze(0)
    cosine_term = torch.cos(omega * dist)
    
    psi_1 = var_kern * damping * cosine_term
    
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    edge_vars = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    edge_vars = edge_vars - edge_means.pow(2)
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)

# ==============================================================================
# PART 2: LAYERS & MODEL
# ==============================================================================

class GPKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_inducing=20, kernel_type='rbf'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_type = kernel_type.lower()
        
        # --- Learnable Parameters ---
        
        # 1. Inducing Locations (z) - Initialized uniformly
        self.z = nn.Parameter(
            torch.linspace(-1.5, 1.5, num_inducing)
            .reshape(1, 1, num_inducing)
            .repeat(out_features, in_features, 1)
        )
        
        # 2. Variational Parameters (q_mu, q_log_var)
        self.q_mu = nn.Parameter(torch.randn(out_features, in_features, num_inducing) * 0.05)
        self.q_log_var = nn.Parameter(torch.ones(out_features, in_features, num_inducing) * -3.0)
        
        # 3. Kernel Hyperparameters
        # log_scale handles Lengthscale (RBF) or Period (Cosine)
        self.log_scale = nn.Parameter(torch.zeros(out_features, in_features))
        # log_variance handles ARD (Pruning)
        self.log_variance = nn.Parameter(torch.zeros(out_features, in_features) - 1.0)

    def forward(self, x):
        # Handle deterministic vs probabilistic input
        if isinstance(x, torch.Tensor):
            x_mean, x_var = x, torch.zeros_like(x) + 1e-6
        else:
            x_mean, x_var = x
            x_var = torch.clamp(x_var, min=1e-6)

        # --- Clamping for Stability (The "Crash Fix") ---
        constrained_q_var = torch.exp(self.q_log_var).clamp(min=1e-5)
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        
        # Scale constraints depend on kernel
        min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
        constrained_scale = torch.exp(self.log_scale).clamp(min=min_scale)

        # Dispatch Kernel
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

        # Aggregation (KAN Summation)
        return edge_means.sum(dim=2), edge_vars.sum(dim=2)

    def get_relevance(self):
        """Returns the ARD variance magnitude for interpretability."""
        return torch.exp(self.log_variance).detach().cpu()

class GPKAN(nn.Module):
    def __init__(self, layers_hidden, num_inducing=20, kernel_type='rbf'):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                GPKANLayer(
                    layers_hidden[i], layers_hidden[i+1], 
                    num_inducing=num_inducing, kernel_type=kernel_type
                )
            )

    def forward(self, x):
        state = x
        for layer in self.layers:
            state = layer(state)
        return state

def gaussian_nll_loss(pred_mean, pred_var, target):
    """Negative Log Likelihood Loss for Gaussian distributions."""
    pred_var = torch.clamp(pred_var, min=1e-6)
    loss_term = 0.5 * (torch.log(pred_var) + (target - pred_mean).pow(2) / pred_var)
    return loss_term.mean()

# ==============================================================================
# PART 3: REGRESSOR API
# ==============================================================================

class GPKANRegressor:
    """
    High-level 'Scikit-Learn style' API for GP-KAN.
    """
    def __init__(self, hidden_layers=[1, 5, 1], kernel='rbf', num_inducing=20, device='cpu'):
        self.hidden_layers = hidden_layers
        self.kernel = kernel
        self.num_inducing = num_inducing
        self.device = device
        
        self.model = GPKAN(
            layers_hidden=hidden_layers, 
            num_inducing=num_inducing, 
            kernel_type=kernel
        ).to(device)
        
        self.history = {'loss': [], 'sparsity': []}
        self.trained = False

    def fit(self, X, y, epochs=1000, lr=0.02, sparsity_weight=0.05, verbose=True):
        X_ten = self._to_tensor(X)
        y_ten = self._to_tensor(y)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.8)
        
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            mu, var = self.model(X_ten)
            
            # Loss: Accuracy (NLL) + Pruning (L1 on Input Variance)
            nll = gaussian_nll_loss(mu, var, y_ten)
            
            l1_loss = 0
            first_layer = self.model.layers[0]
            l1_loss += torch.exp(first_layer.log_variance).sum() * sparsity_weight
            
            loss = nll + l1_loss
            loss.backward()
            
            # Clip Gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            self.history['loss'].append(nll.item())
            self.history['sparsity'].append(l1_loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | NLL: {nll.item():.3f} | Sparsity: {l1_loss.item():.3f}")

        self.trained = True
        return self

    def predict(self, X):
        if not self.trained: raise RuntimeError("Call fit() first.")
        self.model.eval()
        X_ten = self._to_tensor(X)
        with torch.no_grad():
            mu, var = self.model(X_ten)
            std = torch.sqrt(var)
        return mu.cpu().numpy(), std.cpu().numpy()

    def explain(self, threshold=0.01):
        print("\n=== GP-KAN Model Explanation ===")
        layer0 = self.model.layers[0]
        relevance = layer0.get_relevance().mean(dim=0)
        scales = torch.exp(layer0.log_scale).mean(dim=0)
        
        for i in range(self.hidden_layers[0]):
            score, s = relevance[i].item(), scales[i].item()
            status = "[PRUNED]" if score < threshold else "[ACTIVE]"
            shape = "Linear" if s > 3.0 else ("High Freq" if s < 0.5 else "Smooth/Non-Linear")
            if status == "[ACTIVE]":
                print(f"Feature {i}: {status} importance={score:.3f} | Type: {shape} (scale={s:.2f})")
            else:
                print(f"Feature {i}: {status} (Irrelevant)")

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray): return torch.from_numpy(x).float().to(self.device)
        return x.float().to(self.device)
