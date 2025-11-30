import torch
import torch.nn as nn
from svgp_kan import GPKANLayer

class SVGPUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=32):
        super().__init__()
        
        # --- Encoder ---
        # Level 1
        self.enc1 = self.conv_block(in_channels, base)      # -> [B, 32, H, W]
        self.pool1 = nn.MaxPool2d(2)                        # -> [B, 32, H/2, W/2]
        
        # Level 2
        self.enc2 = self.conv_block(base, base*2)           # -> [B, 64, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2)                        # -> [B, 64, H/4, W/4]
        
        # --- Bottleneck (The SVGP Innovation) ---
        # Input: [B, 64, H/4, W/4]
        # The GPKANLayer now natively handles this 4D shape!
        self.bottleneck_gp = GPKANLayer(
            in_features=base*2,   # 64
            out_features=base*2,  # 64
            num_inducing=32,      # Inducing points in "Feature Space"
            kernel_type='rbf'
        )
        
        # --- Decoder ---
        # Level 2 Upsample
        self.up2 = nn.ConvTranspose2d(base*2, base*2, kernel_size=2, stride=2) 
        # Skip connection: enc2 has 64 channels. up2 has 64 channels.
        # Cat = 128 channels.
        self.dec2 = self.conv_block(base*2 + base*2, base*2)# 128 -> 64
        
        # Level 1 Upsample
        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        # Skip connection: enc1 has 32 channels. up1 has 32 channels.
        # Cat = 64 channels.
        self.dec1 = self.conv_block(base + base, base)      # 64 -> 32
        
        # Final Head
        self.final = nn.Conv2d(base, num_classes, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # --- Bottleneck (Probabilistic) ---
        # The layer automatically permutes [B, C, H, W] -> [N, C] -> GP -> [B, C, H, W]
        mu, var = self.bottleneck_gp(p2)
        
        # Reparameterization (Sampling)
        # We perform sampling in the latent feature space
        if self.training:
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            z = mu + torch.randn_like(std) * std
        else:
            z = mu # Deterministic mean at test time (or sample manually for MC)
            
        # --- Decoder ---
        # 1. Upsample Bottleneck
        d2 = self.up2(z)
        # 2. Concatenate with Encoder Level 2
        d2 = torch.cat([d2, e2], dim=1) 
        d2 = self.dec2(d2)
        
        # 3. Upsample Level 2
        d1 = self.up1(d2)
        # 4. Concatenate with Encoder Level 1
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final(d1)
        
        # Extract the Uncertainty Map (Bottleneck Variance)
        # We sum variance across the 64 feature channels to get a spatial "Confusion Map"
        uncertainty_map = var.sum(dim=1, keepdim=True) # [B, 1, H/4, W/4]
        
        # Ideally, we upsample the uncertainty map to match image size for visualization
        uncertainty_map = torch.nn.functional.interpolate(
            uncertainty_map, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        
        return logits, uncertainty_map