import numpy as np
import matplotlib.pyplot as plt
import torch

from svgp_kan import GPKANRegressor

def main():
    # ==========================================
    # 1. Synthetic Data Generation
    # ==========================================
    print("Generating Synthetic 'Physics' Data...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    N_SAMPLES = 600
    
    # Input: 3 Dimensions (Batch, 3) in range [-1, 1]
    X = np.random.rand(N_SAMPLES, 3) * 2 - 1
    
    # Ground Truth: y = Sin(x1) + Linear(x2) + Noise(x3)
    f1 = np.sin(3 * np.pi * X[:, 0])  # Feature 0: Periodic
    f2 = 1.5 * X[:, 1]                # Feature 1: Linear
    f3 = 0.0 * X[:, 2]                # Feature 2: Noise
    
    y = f1 + f2 + f3 + np.random.randn(N_SAMPLES) * 0.1
    y = y.reshape(-1, 1)

    # ==========================================
    # 2. Model Initialization & Training
    # ==========================================
    print("\nInitializing SVGP-KAN...")
    
    # Architecture: 3 Inputs -> 5 Hidden GPs -> 1 Output
    model = GPKANRegressor(
        hidden_layers=[3, 5, 1], 
        kernel='rbf', 
        num_inducing=20, 
        device='cpu' 
    )

    print("Training...")
    model.fit(X, y, epochs=1000, lr=0.02, sparsity_weight=0.05)

    # ==========================================
    # 3. Scientific Discovery (Explanation)
    # ==========================================
    model.explain(threshold=0.01)

    # ==========================================
    # 4. Visualization
    # ==========================================
    print("\nVisualizing results...")
    
    N_TEST = 100
    grid = np.linspace(-1, 1, N_TEST)
    zeros = np.zeros(N_TEST)

    # Prepare Slices
    X_test_1 = np.column_stack([grid, zeros, zeros]) # Vary Feat 0
    X_test_2 = np.column_stack([zeros, grid, zeros]) # Vary Feat 1
    X_test_3 = np.column_stack([zeros, zeros, grid]) # Vary Feat 2

    # Predict
    mu1, std1 = model.predict(X_test_1)
    mu2, std2 = model.predict(X_test_2)
    mu3, std3 = model.predict(X_test_3)

    # Flatten
    mu1, std1 = mu1.flatten(), std1.flatten()
    mu2, std2 = mu2.flatten(), std2.flatten()
    mu3, std3 = mu3.flatten(), std3.flatten()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1
    axes[0].plot(grid, mu1, 'b-', label='Prediction')
    axes[0].plot(grid, np.sin(3 * np.pi * grid), 'r--', label='Truth')
    axes[0].fill_between(grid, mu1 - 2*std1, mu1 + 2*std1, color='b', alpha=0.2)
    axes[0].set_title("Feature 0: Periodic")
    axes[0].legend()

    # Panel 2
    axes[1].plot(grid, mu2, 'b-', label='Prediction')
    axes[1].plot(grid, 1.5 * grid, 'r--', label='Truth')
    axes[1].fill_between(grid, mu2 - 2*std2, mu2 + 2*std2, color='b', alpha=0.2)
    axes[1].set_title("Feature 1: Linear")

    # Panel 3
    axes[2].plot(grid, mu3, 'b-', label='Prediction')
    axes[2].fill_between(grid, mu3 - 2*std3, mu3 + 2*std3, color='b', alpha=0.2)
    axes[2].set_ylim(-2, 2)
    axes[2].set_title("Feature 2: Noise (Pruned)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
