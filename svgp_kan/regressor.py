import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from .model import GPKAN, gaussian_nll_loss

class GPKANRegressor:
    """
    High-level API for GP-KAN (Scikit-Learn style).
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
        
        self.likelihood_log_var = torch.nn.Parameter(
            torch.tensor(-2.0, device=device) 
        )
        
        self.history = {'loss': [], 'sparsity': []}
        self.trained = False
        self.noise_lower_bound = 1e-4 

    def fit(self, X, y, epochs=1000, lr=0.02, sparsity_weight=0.05, noise_lower_bound=None, verbose=True):
        X_ten = self._to_tensor(X)
        y_ten = self._to_tensor(y)
        
        data_var = y_ten.var().item()
        if noise_lower_bound is None:
            self.noise_lower_bound = 1e-4
        else:
            if noise_lower_bound > 0.5 * data_var:
                safe_bound = 0.5 * data_var
                if verbose:
                    print(f"WARNING: noise_lower_bound ({noise_lower_bound:.4f}) is too high.")
                    print(f"Auto-adjusting bound to {safe_bound:.4f}.")
                self.noise_lower_bound = safe_bound
            else:
                self.noise_lower_bound = noise_lower_bound

        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': [self.likelihood_log_var], 'lr': 0.05} 
        ], lr=lr)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.8)
        
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            f_mu, f_var = self.model(X_ten)
            
            obs_noise = torch.exp(self.likelihood_log_var).clamp(min=self.noise_lower_bound)
            y_var = f_var + obs_noise
            
            nll = gaussian_nll_loss(f_mu, y_var, y_ten)
            
            l1_loss = 0
            first_layer = self.model.layers[0]
            l1_loss += torch.exp(first_layer.log_variance).sum() * sparsity_weight
            
            total_loss = nll + l1_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            self.history['loss'].append(nll.item())
            self.history['sparsity'].append(l1_loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | NLL: {nll.item():.3f} | Sparsity: {l1_loss.item():.3f} | Noise: {obs_noise.item():.4f}")

        self.trained = True
        return self

    def predict(self, X, return_std=True, include_likelihood=True):
        """
        Universal Predict: Handles both legacy scripts.
        """
        if not self.trained:
            raise RuntimeError("Model is not trained yet. Call .fit() first.")
            
        self.model.eval()
        X_ten = self._to_tensor(X)
        
        with torch.no_grad():
            f_mu, f_var = self.model(X_ten)
            
            if not return_std:
                return f_mu.cpu().numpy()

            if include_likelihood:
                obs_noise = torch.exp(self.likelihood_log_var).clamp(min=self.noise_lower_bound)
                y_var = f_var + obs_noise
                std = torch.sqrt(y_var)
            else:
                std = torch.sqrt(f_var)
            
        return f_mu.cpu().numpy(), std.cpu().numpy()

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

    def plot_diagnosis(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'])
        plt.title("NLL Loss")
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(self.history['sparsity'])
        plt.title("Sparsity Loss")
        plt.grid(True, alpha=0.3)
        plt.show()

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        return x.float().to(self.device)