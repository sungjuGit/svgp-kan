import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure library is importable
import sys
sys.path.append('.') 
from svgp_kan import gaussian_nll_loss, SVGPKanPOD

# ==========================================
# 0. PUBLICATION STYLE CONFIGURATION
# ==========================================
def set_pub_style():
    plt.rcParams.update({
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold',
        'axes.grid': False,      # No grid lines
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })

set_pub_style()
OUTPUT_DIR = "examples/KanPOD/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. PHYSICS & DATA UTILS
# ==========================================
class FeatureScaler:
    def __init__(self, data, feature_range=(-1, 1)):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        self.range = self.max - self.min + 1e-6
        self.feature_range = feature_range
        self.scale = (feature_range[1] - feature_range[0]) / self.range
        self.shift = feature_range[0] - self.min * self.scale

    def transform(self, x): return x * self.scale + self.shift

class FieldScaler:
    def __init__(self, data, feature_range=(-1, 1)):
        self.min = data.min()
        self.max = data.max()
        self.range = self.max - self.min + 1e-6
        self.feature_range = feature_range
        self.scale = (feature_range[1] - feature_range[0]) / self.range
        self.shift = feature_range[0] - self.min * self.scale

    def transform(self, x): return x * self.scale + self.shift
    def inverse_transform(self, x): return (x - self.shift) / self.scale

def heat_equation_solution(x, source_loc, intensity, width=0.1):
    return intensity * np.exp(- (x - source_loc)**2 / (2 * width**2))

def generate_data(n_samples=500, n_sensors=20):
    # Params: [Source Loc (0-1), Intensity (1-5)]
    params = np.random.rand(n_samples, 2)
    params[:, 1] = params[:, 1] * 4 + 1 
    x = np.linspace(0, 1, n_sensors).reshape(-1, 1)
    Y = np.zeros((n_samples, n_sensors))
    for i in range(n_samples):
        Y[i, :] = heat_equation_solution(x.flatten(), params[i, 0], params[i, 1])
    return params, x, Y

# ==========================================
# 2. MODEL TRAINING
# ==========================================
print("--- 1. Generating Data & Training KanPOD ---")

# A. Data Generation
raw_params, raw_coords, raw_y = generate_data(n_samples=500, n_sensors=32)

# B. Normalization (Crucial for GPs)
p_scaler = FeatureScaler(raw_params)
x_scaler = FeatureScaler(raw_coords)
y_scaler = FieldScaler(raw_y)

train_params = torch.FloatTensor(p_scaler.transform(raw_params))
train_coords = torch.FloatTensor(x_scaler.transform(raw_coords))
train_y = torch.FloatTensor(y_scaler.transform(raw_y))

# C. Model Definition
model = SVGPKanPOD(
    param_dim=2, 
    spatial_dim=1, 
    latent_dim=8,
    branch_depth=[32, 32],
    trunk_depth=[32, 32],
    num_inducing=40
)

optimizer = optim.Adam(model.parameters(), lr=0.01)

# D. Training Loop
print("   Training for 1500 epochs...")
for epoch in range(1501):
    optimizer.zero_grad()
    mu, var = model(train_params, train_coords)
    nll = gaussian_nll_loss(mu, var, train_y)
    kl = model.compute_total_kl()
    
    # Annealing KL weight
    loss = nll + min(1e-4, epoch/500 * 1e-4) * kl
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"   Epoch {epoch} | Loss: {loss.item():.3f}")

# ==========================================
# 3. EXPERIMENT 1: SUPER-RESOLUTION
# ==========================================
print("\n--- 2. Experiment: Mesh-Independent Super-Resolution ---")

# Test Case: In-Distribution (Loc=0.65, Int=3.5)
test_loc, test_int = 0.65, 3.5
raw_p_test = np.array([[test_loc, test_int]])
test_p_norm = torch.FloatTensor(p_scaler.transform(raw_p_test))

# Fine Grid (200 points - Never seen during training)
raw_x_fine = np.linspace(0, 1, 200).reshape(-1, 1)
test_x_norm = torch.FloatTensor(x_scaler.transform(raw_x_fine))

with torch.no_grad():
    mu_norm, var_norm = model(test_p_norm, test_x_norm)
    std_norm = torch.sqrt(var_norm)

# Inverse Transform
mu_phys = y_scaler.inverse_transform(mu_norm.numpy()).flatten()
std_phys = (std_norm.numpy() * y_scaler.scale).flatten()
true_y = heat_equation_solution(raw_x_fine.flatten(), test_loc, test_int)

# --- METRIC CALCULATION (NEW) ---
# 1. MSE (Mean Squared Error)
mse = np.mean((mu_phys - true_y)**2)

# 2. RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# 3. NMSE (Normalized MSE) = ||y_pred - y_true||^2 / ||y_true||^2
# This is the standard metric for comparing fields of different magnitudes
nmse = np.linalg.norm(mu_phys - true_y)**2 / np.linalg.norm(true_y)**2

print(f"\n   [RESULTS]")
print(f"   Test Config: Loc={test_loc}, Int={test_int}")
print(f"   Grid Size:   200 points (Super-Resolution)")
print(f"   MSE:         {mse:.2e}")
print(f"   RMSE:        {rmse:.2e}")
print(f"   NMSE:        {nmse:.2e}  <-- Compares to Manuscript Claim")

# --- PLOT 1: RECONSTRUCTION ---
plt.figure(figsize=(8, 5))
# 1. Ground Truth
plt.plot(raw_x_fine, true_y, 'k-', linewidth=2.5, label='Ground Truth')
# 2. Prediction
plt.plot(raw_x_fine, mu_phys, 'b--', linewidth=2.5, label=f'KanPOD (NMSE={nmse:.1e})')
# 3. Uncertainty
plt.fill_between(raw_x_fine.flatten(), mu_phys - 2*std_phys, mu_phys + 2*std_phys, 
                 color='blue', alpha=0.2, label='95% Confidence')
# 4. Training Data (Scatter)
sensor_x = raw_coords.flatten()
sensor_y = heat_equation_solution(sensor_x, test_loc, test_int)
plt.scatter(sensor_x, sensor_y, color='red', s=40, zorder=5, label='Training Sensors (N=32)')

plt.xlabel('Spatial Position (x)')
plt.ylabel('Temperature (u)')
plt.legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_super_resolution.png", dpi=300)
print(f"   Saved {OUTPUT_DIR}/fig1_super_resolution.png")


# ==========================================
# 4. EXPERIMENT 2: OOD DETECTION
# ==========================================
print("\n--- 3. Experiment: OOD Parameter Detection ---")

# Test Case: Out-of-Distribution (Loc=0.5, Int=10.0) -> Training max was 5.0
ood_loc, ood_int = 0.5, 10.0
raw_p_ood = np.array([[ood_loc, ood_int]])
ood_p_norm = torch.FloatTensor(p_scaler.transform(raw_p_ood))

with torch.no_grad():
    mu_ood_norm, var_ood_norm = model(ood_p_norm, test_x_norm)
    std_ood_norm = torch.sqrt(var_ood_norm)

mu_ood = y_scaler.inverse_transform(mu_ood_norm.numpy()).flatten()
std_ood = (std_ood_norm.numpy() * y_scaler.scale).flatten()
true_ood = heat_equation_solution(raw_x_fine.flatten(), ood_loc, ood_int)

# --- PLOT 2: OOD UNCERTAINTY ---
plt.figure(figsize=(8, 5))
# 1. Ground Truth
plt.plot(raw_x_fine, true_ood, 'k-', linewidth=2.5, label='Ground Truth (OOD)')
# 2. Prediction
plt.plot(raw_x_fine, mu_ood, 'r--', linewidth=2.5, label='KanPOD Mean')
# 3. Uncertainty (Should be huge)
plt.fill_between(raw_x_fine.flatten(), mu_ood - 2*std_ood, mu_ood + 2*std_ood, 
                 color='red', alpha=0.2, label='Epistemic Uncertainty')

plt.xlabel('Spatial Position (x)')
plt.ylabel('Temperature (u)')
# Note: No training data scatter plot here as this parameter setting was never seen
plt.legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_ood_detection.png", dpi=300)
print(f"   Saved {OUTPUT_DIR}/fig2_ood_detection.png")

print("\nAll experiments completed.")