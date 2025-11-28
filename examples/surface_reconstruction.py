import numpy as np
import matplotlib.pyplot as plt
import torch

# Clean import from your new library
from svgp_kan import GPKANRegressor

def main():
    # ==========================================
    # 1. Synthetic Data Generation (2D Surface)
    # ==========================================
    print("Generating 2D Wave Data...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    N_SAMPLES = 600
    
    # Input: 2 Dimensions [Batch, 2] in range [-1, 1]
    X_train = np.random.rand(N_SAMPLES, 2) * 2 - 1
    
    # Ground Truth: y = sin(2*pi*x1) + cos(2*pi*x2)
    # This creates a "checkerboard" of peaks and valleys
    y_true = np.sin(2 * np.pi * X_train[:, 0]) + np.cos(2 * np.pi * X_train[:, 1])
    
    # Add noise
    noise_level = 0.2
    y_train = y_true + np.random.randn(N_SAMPLES) * noise_level
    y_train = y_train.reshape(-1, 1)

    print(f"Data Shape: X={X_train.shape}, y={y_train.shape}")

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    print("\nInitializing SVGP-KAN...")
    
    # Architecture: 2 Inputs -> 5 Hidden GPs -> 1 Output
    # We use RBF because the surface is smooth
    model = GPKANRegressor(
        hidden_layers=[2, 5, 1], 
        kernel='rbf', 
        num_inducing=20, 
        device='cpu'
    )

    # ==========================================
    # 3. Training
    # ==========================================
    print("Training...")
    # We don't need heavy sparsity_weight here because both inputs are relevant.
    # A small weight helps clean up the internal routing.
    model.fit(X_train, y_train, epochs=1200, lr=0.02, sparsity_weight=0.01)

    # ==========================================
    # 4. Visualization (The 3-Panel Plot)
    # ==========================================
    print("\nVisualizing Surface...")
    
    # Create a dense grid for plotting
    res = 50
    x1_grid = np.linspace(-1.2, 1.2, res) # Go slightly outside [-1, 1] to see uncertainty
    x2_grid = np.linspace(-1.2, 1.2, res)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
    
    # Flatten for prediction
    X_test_grid = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])

    # Predict
    mu_pred, std_pred = model.predict(X_test_grid)
    
    # Reshape back to grid
    Mean_mesh = mu_pred.reshape(res, res)
    Std_mesh = std_pred.reshape(res, res)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: The Noisy Training Data
    # We color points by their Z-value to visualize the structure
    sc = axes[0].scatter(X_train[:,0], X_train[:,1], c=y_train.ravel(), cmap='viridis', s=20, edgecolors='k', linewidth=0.5)
    axes[0].set_title("1. Training Data (Noisy Input)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    plt.colorbar(sc, ax=axes[0])

    # Panel 2: The Learned Mean Surface
    cp1 = axes[1].contourf(X1_mesh, X2_mesh, Mean_mesh, levels=20, cmap='viridis')
    axes[1].set_title("2. GP-KAN Predicted Mean")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.colorbar(cp1, ax=axes[1])

    # Panel 3: The Predictive Uncertainty
    # This is the "Plaid" plot. Look for yellow bands outside the [-1, 1] box.
    cp2 = axes[2].contourf(X1_mesh, X2_mesh, Std_mesh, levels=20, cmap='plasma')
    axes[2].set_title("3. Predictive Uncertainty (Std Dev)")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    plt.colorbar(cp2, ax=axes[2])

    plt.tight_layout()
    plt.show()
    print("Notice in Plot 3 how uncertainty increases (becomes brighter) outside the [-1, 1] data range.")

if __name__ == "__main__":
    main()
