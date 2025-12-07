import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

from svgp_kan import gaussian_nll_loss
from svgp_kan import SVGPKanPOD

# GPKANLayers expect inputs in [-1.5, 1.5] to cover inducing points well.
# ==========================================
# 1. Specialized Normalizers
# ==========================================
class FeatureScaler:
    """Normalizes each feature column independently (e.g., Location vs Intensity)."""
    def __init__(self, data, feature_range=(-1, 1)):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        self.range = self.max - self.min + 1e-6
        self.feature_range = feature_range
        self.scale = (feature_range[1] - feature_range[0]) / self.range
        self.shift = feature_range[0] - self.min * self.scale

    def transform(self, x):
        return x * self.scale + self.shift

class FieldScaler:
    """Normalizes the entire physical field globally (Scalar normalization)."""
    def __init__(self, data, feature_range=(-1, 1)):
        self.min = data.min() # Global Scalar
        self.max = data.max() # Global Scalar
        self.range = self.max - self.min + 1e-6
        self.feature_range = feature_range
        self.scale = (feature_range[1] - feature_range[0]) / self.range
        self.shift = feature_range[0] - self.min * self.scale

    def transform(self, x):
        return x * self.scale + self.shift

    def inverse_transform(self, x_norm):
        # Scalars broadcast to any shape (1, 200)
        return (x_norm - self.shift) / self.scale

# ==========================================
# 2. Physics & Data
# ==========================================
def heat_equation_solution(x, source_loc, intensity, width=0.1):
    return intensity * np.exp(- (x - source_loc)**2 / (2 * width**2))

def generate_data(n_samples=200, n_sensors=32):
    params = np.random.rand(n_samples, 2)
    params[:, 1] = params[:, 1] * 4 + 1 
    
    x = np.linspace(0, 1, n_sensors).reshape(-1, 1)
    
    Y = np.zeros((n_samples, n_sensors))
    for i in range(n_samples):
        Y[i, :] = heat_equation_solution(x.flatten(), params[i, 0], params[i, 1])
        
    return params, x, Y

# ==========================================
# 3. Training
# ==========================================
def run_study_d():
    print("--- KanPOD 1D Steady-State Heat Conduction with Point Source ---")
    
    # 1. Generate Raw Data
    raw_params, raw_coords, raw_y = generate_data(n_samples=500, n_sensors=32)
    
    # 2. Scaling Strategy
    # Params: Column-wise (Loc is 0-1, Int is 1-5)
    p_scaler = FeatureScaler(raw_params, (-1, 1))
    # Coords: Column-wise (x is 0-1)
    x_scaler = FeatureScaler(raw_coords, (-1, 1))
    # Field: GLOBAL (Temperature is the same unit everywhere)
    y_scaler = FieldScaler(raw_y, (-1, 1)) 
    
    # Apply Transforms
    train_params = torch.FloatTensor(p_scaler.transform(raw_params))
    train_coords = torch.FloatTensor(x_scaler.transform(raw_coords))
    train_y = torch.FloatTensor(y_scaler.transform(raw_y))
    
    # 3. Define KanPOD Model
    model = SVGPKanPOD(
        param_dim=2, 
        spatial_dim=1, 
        latent_dim=8,
        branch_depth=[16,16], #[32, 32],
        trunk_depth=[16,16], #[32, 32],
        num_inducing=30
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 4. Train
    print("Training...")
    for epoch in range(801):
        optimizer.zero_grad()
        mu, var = model(train_params, train_coords)
        nll = gaussian_nll_loss(mu, var, train_y)
        kl = model.compute_total_kl()
        
        loss = nll + min(0.01, epoch/500 * 0.01) * kl
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.3f} | NLL: {nll.item():.3f}")

    # ==========================================
    # 5. Super-Resolution Test (The crash happened here)
    # ==========================================
    test_loc = 0.65
    test_int = 3.5
    
    # Inputs
    raw_p_test = np.array([[test_loc, test_int]])
    test_p_norm = torch.FloatTensor(p_scaler.transform(raw_p_test))
    
    # Fine Grid (200 points)
    raw_x_fine = np.linspace(0, 1, 200).reshape(-1, 1)
    test_x_norm = torch.FloatTensor(x_scaler.transform(raw_x_fine))
    
    # Predict
    with torch.no_grad():
        # Output is (1, 200)
        mu_norm, var_norm = model(test_p_norm, test_x_norm)
        std_norm = torch.sqrt(var_norm)
    
    # Inverse Transform
    # NOW THIS WORKS: Scalars broadcast to (1, 200)
    mu_phys = y_scaler.inverse_transform(mu_norm.numpy())
    std_phys = std_norm.numpy() * y_scaler.scale
    
    # ==========================================
    # 6. Visualization
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    true_y = heat_equation_solution(raw_x_fine.flatten(), test_loc, test_int)
    plt.plot(raw_x_fine, true_y, 'k--', linewidth=1.5, label='Ground Truth')
    
    # Plot Training Sensors
    sensor_x = raw_coords.flatten()
    sensor_y = heat_equation_solution(sensor_x, test_loc, test_int)
    plt.plot(sensor_x, sensor_y, 'ro', markersize=4, label='Training Sensors')
    
    x_plot = raw_x_fine.flatten()
    mu_plot = mu_phys.flatten()
    std_plot = std_phys.flatten()
    
    plt.plot(x_plot, mu_plot, 'b-', linewidth=2, label='KanPOD Prediction')
    plt.fill_between(x_plot, mu_plot - 2*std_plot, mu_plot + 2*std_plot, color='b', alpha=0.2, label='95% Conf')
    
    plt.title(f"KanPOD Super-Resolution (Corrected)\nLoc={test_loc}, Int={test_int}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('kanpod_.png')
    print("Success! Saved kanpod_.png")

if __name__ == "__main__":
    run_study_d()