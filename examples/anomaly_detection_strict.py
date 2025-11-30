import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

from svgp_kan import GPKANLayer

'''
Reconstruction:

"0": Should look like a "0".

"7": Should look like a weird, blurry "0" (or a blob). It should not look like a sharp "7". This is the goal.

Uncertainty (Right Panel):

The Bar for "7" should be significantly higher than for "0".

This demonstrates OOD Detection: "I can't reconstruct this '7' properly, AND my internal variance is spiking because I don't recognize these features."
'''

class StrictAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Encoder ---
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*7*7, 64),
            nn.ReLU()
        )
        
        # --- Bottleneck ---
        self.bottleneck = GPKANLayer(
            in_features=64, 
            out_features=6,  
            num_inducing=20, 
            kernel_type='rbf'
        )
        
        # --- TUNING ---
        # 1. Lengthscale: Keep it somewhat large to ensure '0's are connected
        nn.init.constant_(self.bottleneck.log_scale, 0.0) 
        # 2. Variance Amplitude: Start high (1.0) so OOD samples have high uncertainty
        nn.init.constant_(self.bottleneck.log_variance, 0.0) 
        
        # --- Decoder ---
        self.dec_linear = nn.Linear(6, 16*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid() 
        )

    def forward(self, x):
        feat = self.enc(x)
        mu, var = self.bottleneck(feat)
        
        if self.training:
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            z = mu + torch.randn_like(std) * std
        else:
            z = mu 
            
        z_spatial = self.dec_linear(z).view(-1, 16, 7, 7)
        recon = self.dec(z_spatial)
        
        # Uncertainty: Sum variance
        uncertainty_scalar = var.sum(dim=1).view(-1, 1, 1, 1)
        
        return recon, uncertainty_scalar

def get_data(normal_digit=0, batch_size=64): 
    print(f"Preparing Data: Normal Class = '{normal_digit}'")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Train only on '0'
    train_idx = [i for i, (img, label) in enumerate(dataset) if label == normal_digit]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train(model, loader, device, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            recon, _ = model(x)
            
            # --- ONLY RECONSTRUCTION LOSS ---
            # We removed the sparsity loss. 
            # We want the GP to maintain its natural variance for OOD data.
            loss = F.mse_loss(recon, x)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

def evaluate(model, test_loader, device, normal_digit=0, anomaly_digit=7):
    model.eval()
    img_normal, img_anomaly = None, None
    
    # 1. Grab Samples
    for x, y in test_loader:
        if img_normal is None and (y == normal_digit).any():
            img_normal = x[y == normal_digit][0].unsqueeze(0)
        if img_anomaly is None and (y == anomaly_digit).any():
            img_anomaly = x[y == anomaly_digit][0].unsqueeze(0)
        if img_normal is not None and img_anomaly is not None: break
            
    imgs = torch.cat([img_normal, img_anomaly]).to(device)
    
    # 2. Forward Pass
    with torch.no_grad():
        recons, uncs = model(imgs)
        
    imgs, recons = imgs.cpu(), recons.cpu()
    uncs = uncs.cpu() 
    
    # 3. NUMERICAL LOGGING (The Fix)
    u_norm = uncs[0].item()
    u_anom = uncs[1].item()
    ratio = u_anom / (u_norm + 1e-6)
    
    print("\n=== Anomaly Detection Results ===")
    print(f"Normal  ('{normal_digit}') Uncertainty: {u_norm:.4f}")
    print(f"Anomaly ('{anomaly_digit}') Uncertainty: {u_anom:.4f}")
    print(f"Anomaly Score Ratio: {ratio:.2f}x")
    print("=================================")

    # 4. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True, dpi=300)

    labels = [f"Normal ({normal_digit})", f"Anomaly ({anomaly_digit})"]
    
    # Global scaling for fair visual comparison
    u_max = max(u_norm, u_anom)
    
    for i in range(2):
        # Input
        axes[i, 0].imshow(imgs[i, 0], cmap='gray')
        axes[i, 0].set_title(f"Input: {labels[i]}")
        axes[i, 0].axis('off')
        
        # Reconstruction
        axes[i, 1].imshow(recons[i, 0], cmap='gray')
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis('off')
        
        # Uncertainty Bar
        u_val = uncs[i].item()
        color = 'green' if i == 0 else 'red'
        
        axes[i, 2].bar([0], [u_val], color=color, width=0.5)
        axes[i, 2].set_ylim(0, u_max * 1.2)
        axes[i, 2].set_title(f"Uncertainty: {u_val:.4f}")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks(np.linspace(0, u_max, 5))
        
    plt.suptitle(f"Anomaly Detection (Ratio: {ratio:.1f}x)")
    plt.savefig("examples/anomaly_detection_strict.png", dpi=300)
    #plt.show()

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StrictAutoencoder().to(DEVICE)
    train_loader, test_loader = get_data(normal_digit=0)
    train(model, train_loader, DEVICE, epochs=40)
    evaluate(model, test_loader, DEVICE, normal_digit=0, anomaly_digit=7)