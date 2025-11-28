print("Setting up Data...")
torch.manual_seed(42)
np.random.seed(42)
    
# 1. Generate Hybrid Data (Sine + Linear + Noise)
N = 600
X_np = np.random.rand(N, 3) * 2 - 1
y_np = np.sin(3 * np.pi * X_np[:, 0]) + 1.5 * X_np[:, 1] # Truth
y_np = y_np + np.random.randn(N) * 0.1 # Add Noise
y_np = y_np.reshape(-1, 1)

# 2. Train Model
print("Initializing GP-KAN (RBF Mode)...")
# Architecture: 3 inputs -> 5 hidden GPs -> 1 output
model = GPKANRegressor(hidden_layers=[3, 5, 1], kernel='rbf', num_inducing=20)
    
print("Fitting...")
model.fit(X_np, y_np, epochs=1200, sparsity_weight=0.05)
    
# 3. Explain Findings
model.explain(threshold=0.05)

# 4. Plot Comparison
print("\nPlotting Results...")

N_test = 100
x_grid = np.linspace(-1, 1, N_test)
zeros = np.zeros(N_test)

# Prepare Slices
X_slice1 = np.column_stack((x_grid, zeros, zeros)) # Vary x1
X_slice2 = np.column_stack((zeros, x_grid, zeros)) # Vary x2
X_slice3 = np.column_stack((zeros, zeros, x_grid)) # Vary x3

# Predict
mu1, std1 = model.predict(X_slice1)
mu2, std2 = model.predict(X_slice2)
mu3, std3 = model.predict(X_slice3)
    
# Flatten
mu1, std1 = mu1.flatten(), std1.flatten()
mu2, std2 = mu2.flatten(), std2.flatten()
mu3, std3 = mu3.flatten(), std3.flatten()

# Visualize
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Sine
ax[0].plot(x_grid, mu1, 'b-', label='GP-KAN')
ax[0].plot(x_grid, np.sin(3*np.pi*x_grid), 'r--', label='Truth')
ax[0].fill_between(x_grid, mu1-2*std1, mu1+2*std1, color='b', alpha=0.2)
ax[0].set_title("Feature 1: Periodic Discovery")
ax[0].legend()

# Panel 2: Linear
ax[1].plot(x_grid, mu2, 'b-', label='GP-KAN')
ax[1].plot(x_grid, 1.5*x_grid, 'r--', label='Truth')
ax[1].fill_between(x_grid, mu2-2*std2, mu2+2*std2, color='b', alpha=0.2)
ax[1].set_title("Feature 2: Linear Discovery")

# Panel 3: Noise
ax[2].plot(x_grid, mu3, 'b-', label='GP-KAN')
ax[2].fill_between(x_grid, mu3-2*std3, mu3+2*std3, color='b', alpha=0.2)
ax[2].set_ylim(-2, 2)
ax[2].set_title("Feature 3: Pruned Noise")

plt.tight_layout()
plt.show()
print("Done. Notice how Feature 3 is flat (pruned) and Feature 2 is straight (linearized RBF).")
