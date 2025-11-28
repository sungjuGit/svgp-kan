import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from .model import GPKAN, gaussian_nll_loss

class GPKANRegressor:
    """
    High-level API for GP-KAN (Scikit-Learn style).
    Handles training loops, data conversion, ARD pruning, and visualization.
    """
    def __init__(self, hidden_layers=[1, 5, 1], kernel='rbf', num_inducing=20, device='cpu'):
        self.hidden_layers = hidden_layers
        self.kernel = kernel
        self.num_inducing = num_inducing
        self.device = device
        
        # Initialize the model
        self.model = GPKAN(
            layers_hidden=hidden_layers, 
            num_inducing=num_inducing, 
            kernel_type=kernel
        ).to(device)
        
        self.history = {'loss': [], 'sparsity': []}
        self.trained = False

    def fit(self, X, y, epochs=1000, lr=0.02, sparsity_weight=0.05, verbose=True):
        """
        Trains the model.
        Args:
            sparsity_weight: Strength of ARD (L1 penalty) to prune irrelevant features.
        """
        X_ten = self._to_tensor(X)
        y_ten = self._to_tensor(y)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.8)
        
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            mu, var = self.model(X_ten)
            
            # 1. Accuracy Loss (NLL)
            nll = gaussian_nll_loss(mu, var, y_ten)
            
            # 2. Sparsity Loss (L1 on Input Layer Variance)
            l1_loss = 0
            first_layer = self.model.layers[0]
            l1_loss += torch.exp(first_layer.log_variance).sum() * sparsity_weight
            
            total_loss = nll + l1_loss
            
            total_loss.backward()
            
            # Clip gradients to prevent variational explosion
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
        """Returns (Mean, StdDev) as numpy arrays."""
        if not self.trained:
            raise RuntimeError("Model is not trained yet. Call .fit() first.")
            
        self.model.eval()
        X_ten = self._to_tensor(X)
        
        with torch.no_grad():
            mu, var = self.model(X_ten)
            std = torch.sqrt(var)
            
        return mu.cpu().numpy(), std.cpu().numpy()

    def explain(self, threshold=0.01):
        """
        Prints the Scientific Discovery report.
        Identifies pruned features and linear vs non-linear relationships.
        """
        print("\n=== GP-KAN Model Explanation ===")
        layer0 = self.model.layers[0]
        
        # Average relevance over output neurons
        relevance = layer0.get_relevance().mean(dim=0)
        # Get learned scales (Lengthscale or Period)
        scales = torch.exp(layer0.log_scale).mean(dim=0)
        
        n_features = self.hidden_layers[0]
        
        for i in range(n_features):
            score = relevance[i].item()
            s = scales[i].item()
            
            if score < threshold:
                status = "[PRUNED] (Irrelevant)"
                print(f"Feature {i}: {status}")
            else:
                # Interpret Shape based on Scale
                if s > 3.0:
                    shape = "Linear / Monotonic"
                elif s < 0.5:
                    shape = "High Frequency / Complex"
                else:
                    shape = "Non-Linear / Smooth"
                    
                status = "[ACTIVE]"
                print(f"Feature {i}: {status} importance={score:.3f} | Type: {shape} (scale={s:.2f})")

    def plot_diagnosis(self):
        """Plots loss curves."""
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'])
        plt.title("NLL Loss (Accuracy)")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['sparsity'])
        plt.title("Sparsity Loss (Pruning)")
        plt.grid(True, alpha=0.3)
        plt.show()

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        return x.float().to(self.device)
