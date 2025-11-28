import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from svgp_kan import GPKANRegressor

def calculate_calibration_curve(y_true, y_mean, y_std, n_bins=10):
    """
    Computes empirical probability for confidence intervals.
    If perfectly calibrated, x% of data should fall within the x% confidence interval.
    """
    confidences = np.linspace(0.1, 0.9, n_bins)
    empirical_coverage = []
    
    for conf in confidences:
        z_score = np.abs(np.percentile(np.random.standard_normal(10000), (1 + conf) / 2 * 100))
        lower = y_mean - z_score * y_std
        upper = y_mean + z_score * y_std
        
        in_bounds = np.logical_and(y_true >= lower, y_true <= upper)
        empirical_coverage.append(np.mean(in_bounds))
        
    return confidences, np.array(empirical_coverage)

def main():
    print("=== Friedman #1 Benchmark (Real-World Proxy) ===")
    print("Goal: Test if SVGP-KAN can identify the 5 real features out of 10.")
    
    # 1. Load Data
    # Friedman 1: 10 inputs. Only first 5 are used. x0*x1 interaction exists.
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42)
    y = y.reshape(-1, 1)
    
    # Scale data (Important for GPs)
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X = scaler_x.transform(X)
    y = scaler_y.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Deep SVGP-KAN
    # We need depth=2 or 3 to capture the x0*x1 interaction via addition
    print("Training Deep SVGP-KAN (10 -> 8 -> 1)...")
    model = GPKANRegressor(
        hidden_layers=[10, 8, 1], 
        kernel='rbf', 
        num_inducing=20, 
        device='cpu'
    )
    
    model.fit(X_train, y_train, epochs=800, lr=0.02, sparsity_weight=0.02, verbose=False)
    
    # 3. Predict & Metrics
    mu, std = model.predict(X_test)
    
    # Metric 1: RMSE
    rmse = np.sqrt(np.mean((mu - y_test)**2))
    print(f"\nTest RMSE: {rmse:.4f} (Lower is better)")
    
    # Metric 2: NLL (Uncertainty Quality)
    # NLL = 0.5 * log(var) + 0.5 * (y-mu)^2 / var
    var = std**2
    nll = 0.5 * np.log(var) + 0.5 * ((y_test - mu)**2) / var
    print(f"Test NLL:  {np.mean(nll):.4f} (Lower is better)")
    
    # Metric 3: Feature Discovery (ARD)
    print("\n--- Feature Discovery Analysis ---")
    layer0 = model.model.layers[0]
    relevance = layer0.get_relevance().mean(dim=0).numpy()
    
    # Normalize relevance score 0-1
    relevance = relevance / relevance.max()
    
    ground_truth = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] # First 5 are relevant
    
    print(f"{'Feat':<5} | {'Truth':<5} | {'Relevance':<10} | {'Status'}")
    print("-" * 40)
    for i in range(10):
        is_relevant = relevance[i] > 0.1 # Threshold
        status = "✅ Found" if (is_relevant and ground_truth[i]) else ""
        status = "✅ Pruned" if (not is_relevant and not ground_truth[i]) else status
        status = "❌ Missed" if (not is_relevant and ground_truth[i]) else status
        status = "❌ False Pos" if (is_relevant and not ground_truth[i]) else status
        
        print(f"{i:<5} | {ground_truth[i]:<5} | {relevance[i]:.4f}     | {status}")

    # 4. Calibration Plot
    expected, observed = calculate_calibration_curve(y_test, mu, std)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(expected, observed, 'b-o', label="SVGP-KAN")
    plt.xlabel("Expected Confidence Level")
    plt.ylabel("Observed Coverage")
    plt.title("Uncertainty Calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
