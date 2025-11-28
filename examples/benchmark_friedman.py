import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from svgp_kan import GPKANRegressor

# ==========================================
# Helper Functions
# ==========================================

def compute_permutation_importance(model, X_val, y_val):
    """Computes Feature Importance via Permutation."""
    mu_base, _ = model.predict(X_val)
    mse_base = np.mean((mu_base - y_val)**2)
    
    importances = []
    X_val_np = X_val if isinstance(X_val, np.ndarray) else X_val.cpu().numpy()
    
    for i in range(X_val.shape[1]):
        X_perturbed = X_val_np.copy()
        np.random.shuffle(X_perturbed[:, i])
        
        mu_pert, _ = model.predict(X_perturbed)
        mse_pert = np.mean((mu_pert - y_val)**2)
        
        importances.append(max(0, mse_pert - mse_base))
        
    return np.array(importances)

def plot_diagnostics(model, X_test, y_test, importances, rmse, save_dir="."):
    """Generates diagnostic plots."""
    print(f"Generating plots in {save_dir}...")
    
    # 1. Feature Importance
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ['#2ca02c' if i < 5 else '#d62728' for i in range(10)]
    plt.bar(range(10), importances, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance (MSE Increase)")
    plt.title(f"Feature Relevance | Test RMSE: {rmse:.3f}")
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "examples/friedman_feature_importance.png"), bbox_inches='tight')
    plt.close()

    # 2. Regression Fit
    mu, std = model.predict(X_test, include_likelihood=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    subset = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
    ax[0].errorbar(y_test.flatten()[subset], mu.flatten()[subset], 
                   yerr=std.flatten()[subset]*2, fmt='o', alpha=0.5, color='blue', label='Preds (95% CI)')
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax[0].set_title("Regression Accuracy")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    residuals = y_test.flatten() - mu.flatten()
    ax[1].scatter(mu.flatten(), residuals, alpha=0.5, color='purple')
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_title("Residual Analysis")
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "examples/friedman_regression_fit.png"), bbox_inches='tight')
    plt.close()

    # 3. Discovery Scan
    scan_grid = np.linspace(-2, 2, 100)
    zeros = np.zeros((100, 10)) 
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=300)
    axes = axes.flatten()
    
    for i in range(10):
        X_scan = zeros.copy()
        X_scan[:, i] = scan_grid
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
        if not is_signal: ax.set_facecolor('#fff5f5') 
            
    plt.suptitle("Learned Functional Shapes", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "examples/friedman_learned_functions.png"), bbox_inches='tight')
    plt.close()

def plot_interaction_analysis(model, scaler_x, scaler_y, save_dir="."):
    """Visualizes x0*x1 interaction."""
    print("Generating Interaction Surface...")
    res = 50
    x0_grid = np.linspace(0, 1, res)
    x1_grid = np.linspace(0, 1, res)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)
    
    X_raw = np.ones((res*res, 10)) * 0.5
    X_raw[:, 0] = X0.ravel()
    X_raw[:, 1] = X1.ravel()
    
    X_input = scaler_x.transform(X_raw)
    mu_inter, _ = model.predict(X_input, include_likelihood=False)
    Z_pred = scaler_y.inverse_transform(mu_inter).reshape(res, res)
    
    Z_true = 10 * np.sin(np.pi * X0 * X1) + 7.5
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    extent = [0, 1, 0, 1]
    
    im0 = axes[0].imshow(Z_true, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(Z_pred, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title("Prediction")
    plt.colorbar(im1, ax=axes[1])
    
    err = np.abs(Z_true - Z_pred)
    im2 = axes[2].imshow(err, extent=extent, origin='lower', cmap='plasma', aspect='auto')
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "examples/friedman_interaction_surface.png"), bbox_inches='tight')
    plt.close()

# ==========================================
# Main Script (Robust 5-Trial Run)
# ==========================================

def run_single_trial(seed):
    """Runs one complete training cycle."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=seed)
    y = y.reshape(-1, 1)
    
    target_noise = 1.0 / np.var(y)
    
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)
    
    model = GPKANRegressor(hidden_layers=[10, 10, 1], kernel='rbf', num_inducing=50, device='cpu')
    
    # Phase 1: Discovery
    model.fit(X_train, y_train, epochs=2000, lr=0.03, sparsity_weight=0.0, 
              noise_lower_bound=target_noise * 0.95, verbose=False)
    
    # Phase 2: Permutation
    importances = compute_permutation_importance(model, X_test, y_test)
    importances = np.maximum(importances, 0)
    imp_norm = importances / (importances.max() + 1e-6)
    
    features_to_keep = [i for i in range(10) if imp_norm[i] > 0.05]
    
    # Phase 3: Refinement
    with torch.no_grad():
        for i in range(10):
            if i not in features_to_keep:
                model.model.layers[0].log_variance[:, i] = -10.0
    
    model.fit(X_train, y_train, epochs=500, lr=0.01, sparsity_weight=0.0, 
              noise_lower_bound=target_noise, verbose=False)
    
    mu, _ = model.predict(X_test)
    rmse = np.sqrt(np.mean((mu - y_test)**2))
    
    return {
        'rmse': rmse,
        'features': features_to_keep,
        'model': model,
        'data': (X_test, y_test, scaler_x, scaler_y),
        'importances': importances
    }

def main():
    print("=== Friedman #1 Benchmark (5-Trial Robustness Test) ===")
    
    n_trials = 5
    results = []
    feature_counts = np.zeros(10)
    
    for i in range(n_trials):
        print(f"\n--- Trial {i+1}/{n_trials} ---")
        res = run_single_trial(seed=42+i)
        results.append(res)
        
        print(f"RMSE: {res['rmse']:.4f} | Features: {res['features']}")
        for f in res['features']:
            feature_counts[f] += 1

    # --- Final Stats ---
    rmses = [r['rmse'] for r in results]
    print("\n=== Robustness Report ===")
    print(f"Mean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    
    print("\nFeature Stability:")
    print(f"{'Feat':<5} | {'Truth':<5} | {'Selection Rate':<15} | {'Status'}")
    print("-" * 50)
    ground_truth = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    for i in range(10):
        rate = feature_counts[i] / n_trials
        status = "✅ Robust" if (rate > 0.8 and ground_truth[i]) or (rate < 0.2 and not ground_truth[i]) else "⚠️ Unstable"
        print(f"{i:<5} | {ground_truth[i]:<5} | {rate*100:.0f}%            | {status}")

    # --- Generate Plots from Best Model ---
    best_idx = np.argmin(rmses)
    print(f"\nGenerating plots from best trial (Trial {best_idx+1})...")
    best_res = results[best_idx]
    X_test, y_test, sx, sy = best_res['data']
    
    plot_diagnostics(best_res['model'], X_test, y_test, best_res['importances'], best_res['rmse'])
    plot_interaction_analysis(best_res['model'], sx, sy)
    print("Done.")

if __name__ == "__main__":
    main()