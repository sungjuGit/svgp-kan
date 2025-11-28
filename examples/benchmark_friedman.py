import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from svgp_kan import GPKANRegressor

# ==========================================
# Helper Functions
# ==========================================

def compute_permutation_importance(model, X_val, y_val):
    """
    Computes Feature Importance by measuring error increase when a feature is shuffled.
    Robust against overfitting because it tests generalization power.
    """
    # Baseline Performance
    mu_base, _ = model.predict(X_val)
    mse_base = np.mean((mu_base - y_val)**2)
    
    importances = []
    X_val_np = X_val if isinstance(X_val, np.ndarray) else X_val.cpu().numpy()
    
    for i in range(X_val.shape[1]):
        # Shuffle one column to destroy its information
        X_perturbed = X_val_np.copy()
        np.random.shuffle(X_perturbed[:, i])
        
        # Predict on perturbed data
        mu_pert, _ = model.predict(X_perturbed)
        mse_pert = np.mean((mu_pert - y_val)**2)
        
        # Importance = Drop in Accuracy
        # If > 0, the feature was useful. If <= 0, it was noise.
        importances.append(max(0, mse_pert - mse_base))
        
    return np.array(importances)

def plot_diagnostics(model, X_test, y_test, importances, rmse):
    """Generates and saves publication-quality diagnostic figures."""
    
    print("Generating diagnostic plots...")
    
    # --- Plot 1: Feature Importance ---
    plt.figure(figsize=(10, 6), dpi=300)
    # Ground Truth: 0-4 are signal, 5-9 are noise
    colors = ['#2ca02c' if i < 5 else '#d62728' for i in range(10)]
    
    bars = plt.bar(range(10), importances, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8)
    
    # Labeling
    plt.xlabel("Input Feature Index")
    plt.ylabel("Importance (MSE Increase upon Permutation)")
    plt.title(f"Scientific Discovery: Feature Relevance | Test RMSE: {rmse:.3f}")
    plt.xticks(range(10))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ca02c', label='True Signal (Features 0-4)'),
                       Patch(facecolor='#d62728', label='True Noise (Features 5-9)')]
    plt.legend(handles=legend_elements)
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("examples/friedman_feature_importance.png", bbox_inches='tight')
    plt.close()

    # --- Plot 2: Regression Fit ---
    mu, std = model.predict(X_test, include_likelihood=True) # Include noise for calibration check
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Scatter
    subset = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
    ax[0].errorbar(y_test.flatten()[subset], mu.flatten()[subset], 
                   yerr=std.flatten()[subset]*2, fmt='o', alpha=0.5, color='blue', label='Predictions (95% CI)')
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    ax[0].set_xlabel("True Value")
    ax[0].set_ylabel("Predicted Value")
    ax[0].set_title("Regression Accuracy")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test.flatten() - mu.flatten()
    ax[1].scatter(mu.flatten(), residuals, alpha=0.5, color='purple')
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_xlabel("Predicted Value")
    ax[1].set_ylabel("Residual (True - Pred)")
    ax[1].set_title("Residual Analysis")
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/friedman_regression_fit.png", bbox_inches='tight')
    plt.close()

    # --- Plot 3: Discovery Scan (Univariate Shapes) ---
    scan_grid = np.linspace(-2, 2, 100)
    zeros = np.zeros((100, 10)) 
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=300)
    axes = axes.flatten()
    
    for i in range(10):
        X_scan = zeros.copy()
        X_scan[:, i] = scan_grid
        
        # Predict pure function (exclude noise) to see shape clearly
        mu_scan, std_scan = model.predict(X_scan, include_likelihood=False)
        
        ax = axes[i]
        is_signal = i < 5
        color = '#2ca02c' if is_signal else '#d62728'
        
        ax.plot(scan_grid, mu_scan, color=color, linewidth=2)
        ax.fill_between(scan_grid, 
                        (mu_scan - 2*std_scan).flatten(), 
                        (mu_scan + 2*std_scan).flatten(), 
                        color=color, alpha=0.1)
        
        status = "(Signal)" if is_signal else "(Noise)"
        ax.set_title(f"Feature {i} {status}")
        ax.set_ylim(-3, 3) 
        ax.grid(True, alpha=0.3)
        
        # Visual cue for pruned features
        if not is_signal:
            ax.set_facecolor('#fff5f5') 
            
    plt.suptitle("Learned Functional Shapes (Discovery Scan)", fontsize=16)
    plt.tight_layout()
    plt.savefig("examples/friedman_learned_functions.png", bbox_inches='tight')
    plt.close()

def plot_interaction_analysis(model, scaler_x, scaler_y):
    """Visualizes the x0*x1 interaction surface."""
    print("Generating Interaction Surface...")
    res = 50
    x0_grid = np.linspace(0, 1, res)
    x1_grid = np.linspace(0, 1, res)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)
    
    # Create input batch (fix others to 0.5)
    X_raw = np.ones((res*res, 10)) * 0.5
    X_raw[:, 0] = X0.ravel()
    X_raw[:, 1] = X1.ravel()
    
    # Transform to model space
    X_input = scaler_x.transform(X_raw)
    
    # Predict
    mu_inter, _ = model.predict(X_input, include_likelihood=False)
    
    # Transform back to real units
    Z_pred = scaler_y.inverse_transform(mu_inter).reshape(res, res)
    
    # Ground Truth: y = 10sin(pi*x0*x1) + Constants
    # Constants from other fixed features (at 0.5): 
    # 20*(0.5-0.5)^2 + 10*0.5 + 5*0.5 = 7.5
    Z_true = 10 * np.sin(np.pi * X0 * X1) + 7.5
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    extent = [0, 1, 0, 1]
    
    # 1. Ground Truth
    im0 = axes[0].imshow(Z_true, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title("Ground Truth: 10*sin(pi*x0*x1)")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")
    plt.colorbar(im0, ax=axes[0])
    
    # 2. Prediction
    im1 = axes[1].imshow(Z_pred, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title("SVGP-KAN Prediction")
    axes[1].set_xlabel("x0")
    plt.colorbar(im1, ax=axes[1])
    
    # 3. Error
    err = np.abs(Z_true - Z_pred)
    im2 = axes[2].imshow(err, extent=extent, origin='lower', cmap='plasma', aspect='auto')
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x0")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("examples/friedman_interaction_surface.png", bbox_inches='tight')
    plt.close()

# ==========================================
# Main Script
# ==========================================

def main():
    print("=== Friedman #1 Benchmark (Final Publication Run) ===")
    
    # 1. Setup Data
    np.random.seed(42)
    torch.manual_seed(42)
    
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42)
    y = y.reshape(-1, 1)
    
    # Calculate Noise Target for Stability
    target_noise = 1.0 / np.var(y)
    
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y)
    
    print(f"Target Scaled Noise Variance: {target_noise:.4f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # 2. Initialize Model
    # 50 Inducing points are critical for capturing the Sine wave and Interaction
    model = GPKANRegressor(
        hidden_layers=[10, 10, 1], 
        kernel='rbf', 
        num_inducing=50, 
        device='cpu'
    )
    
    # 3. Phase 1: Unconstrained Training (Discovery)
    # We use NO sparsity here to let the model find the complex interaction signal.
    # We rely on the noise_lower_bound to prevent overfitting.
    print("\n[Phase 1] Training (Discovery)...")
    model.fit(
        X_train, y_train, 
        epochs=2000, 
        lr=0.03,
        sparsity_weight=0.0, 
        noise_lower_bound=target_noise * 0.95, # Strict noise floor
        verbose=True
    )
    
    # 4. Phase 2: Identification (Permutation Importance)
    print("\n[Phase 2] Identifying Signal vs Noise...")
    importances = compute_permutation_importance(model, X_test, y_test)
    
    # Normalize
    importances = np.maximum(importances, 0)
    imp_norm = importances / importances.max()
    
    # Identify survivors (> 5% impact)
    features_to_keep = [i for i in range(10) if imp_norm[i] > 0.05]
    
    print("-" * 40)
    print(f"Detected Signal Features: {features_to_keep}")
    print(f"Ground Truth Features:    [0, 1, 2, 3, 4]")
    print("-" * 40)

    # 5. Phase 3: Refinement (Hard Pruning)
    print(f"\n[Phase 3] Refinement (Pruning Noise Features)...")
    
    # Manually kill the variances of pruned features
    with torch.no_grad():
        for i in range(10):
            if i not in features_to_keep:
                # model.input_gates[i] = 0.0  <-- REMOVED THIS LINE
                model.model.layers[0].log_variance[:, i] = -10.0 # Hard kill variance
    
    # Fine-tune the survivors
    model.fit(
        X_train, y_train, 
        epochs=500, 
        lr=0.01,
        sparsity_weight=0.0, 
        noise_lower_bound=target_noise,
        verbose=False
    )

    # 6. Final Evaluation & Plots
    mu, std = model.predict(X_test)
    rmse = np.sqrt(np.mean((mu - y_test)**2))
    rmse_orig = rmse * scaler_y.scale_[0]
    
    print(f"\nFinal Test RMSE (Scaled): {rmse:.4f}")
    print(f"Final Test RMSE (Original Units): {rmse_orig:.4f}")
    print("\nGenerating Plots...")
    
    plot_diagnostics(model, X_test, y_test, imp_norm, rmse)
    plot_interaction_analysis(model, scaler_x, scaler_y)
    
    print("Done! Plots saved to current directory.")

if __name__ == "__main__":
    main()