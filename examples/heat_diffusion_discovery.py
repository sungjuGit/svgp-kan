# install pysr first via: pip3 install pysr
from pysr import PySRRegressor

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

from svgp_kan.layers import GPKANLayer
from svgp_kan.model import gaussian_nll_loss

# ==========================================
# 1. Physics & Model Definitions
# ==========================================
def heat_source_solution(x, t, alpha=1.0):
    t = torch.clamp(t, min=1e-4)
    term1 = 1.0 / torch.sqrt(4 * np.pi * alpha * t)
    term2 = torch.exp(-(x**2) / (4 * alpha * t))
    return term1 * term2

class HeatAutoEncoder(nn.Module):
    def __init__(self, num_inducing=20):
        super().__init__()
        self.enc1 = GPKANLayer(2, 5, num_inducing=num_inducing, kernel_type='rbf')
        self.enc2 = GPKANLayer(5, 1, num_inducing=num_inducing, kernel_type='rbf')
        self.dec1 = GPKANLayer(1, 5, num_inducing=num_inducing, kernel_type='rbf')
        self.dec2 = GPKANLayer(5, 1, num_inducing=num_inducing, kernel_type='rbf')
        self.likelihood_log_var = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x_in):
        h1_mu, h1_var = self.enc1(x_in)
        z_mu, z_var = self.enc2((h1_mu, h1_var))
        h2_mu, h2_var = self.dec1((z_mu, z_var))
        u_mu, u_var = self.dec2((h2_mu, h2_var))
        return u_mu, u_var, z_mu

# ==========================================
# 2. Symbolic Regression Engines
# ==========================================


def symbolic_verification(eta, z):
        
    print("\n--- PySR Symbolic Regression ---")
    print("Searching for equation z = f(eta)...")
        
    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["square", "exp"],
        model_selection="best",
        verbosity=0,
        temp_equation_file=None
    )
        
    # Reshape for sklearn-like API
    X = eta.reshape(-1, 1)
    model.fit(X, z)
        
    print("\nBest PySR Model Found:")
    print(f"z = {model.sympy()}")

    return model.predict(X)
        
    

# ==========================================
# 3. Main
# ==========================================
def main():
    print("=== Heat Diffusion Discovery (Symbolic) ===")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data
    N = 1000

    t = torch.rand(N_SAMPLES := N, 1) * 1.9 + 0.1

    eta_max = 2.0 / np.sqrt(2.0 * 0.1)
    eta_sampled = (torch.rand(N, 1) * 2.0 - 1.0) * eta_max

    x = eta_sampled * torch.sqrt(2.0 * t)

    u_true = heat_source_solution(x, t, alpha=0.5)
    u_noisy = u_true + torch.randn_like(u_true) * 0.02
    X_train = torch.cat([x, t], dim=1)
    
    # Train
    model = HeatAutoEncoder(num_inducing=30)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Autoencoder...")
    for epoch in range(1500):
        optimizer.zero_grad()
        u_mu, u_var, _ = model(X_train)
        obs_noise = torch.exp(model.likelihood_log_var).clamp(min=1e-6)
        nll = gaussian_nll_loss(u_mu, u_var + obs_noise, u_noisy)
        nll.backward()
        optimizer.step()
        if epoch % 500 == 0: print(f"Epoch {epoch}: NLL={nll.item():.3f}")

    # Analysis
    model.eval()
    with torch.no_grad():
        _, _, z_pred = model(X_train)
    
    # We sampled eta directly above, so reuse that (numeric equality holds)
    eta = eta_sampled.numpy().flatten()
    z = z_pred.numpy().flatten()
    
    # Symbolic Regression Step
    z_sym = symbolic_verification(eta, z)
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=120)
    plt.scatter(eta, z, alpha=0.3, label='Learned Latent (NN)')
    
    # Sort for clean line plot
    sort_idx = np.argsort(eta)
    plt.plot(eta[sort_idx], z_sym[sort_idx], 'r--', lw=2, label='Symbolic Law')
    
    plt.xlabel(r"True Similarity $\eta$")
    plt.ylabel("Latent $z$")
    plt.title("Discovery: Latent Variable vs Physics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("examples/heat_symbolic.png")
    plt.show()

if __name__ == "__main__":
    main()