import torch
import torch.nn as nn
from .layers import GPKANLayer

class SVGPUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck (Uses GPKANLayer which now supports 4D)
        self.bottleneck_gp = GPKANLayer(
            in_features=base*2,
            out_features=base*2,
            num_inducing=32,
            kernel_type='rbf'
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base*2, base*2, kernel_size=2, stride=2) 
        self.dec2 = self.conv_block(base*2 + base*2, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base + base, base)
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
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        mu, var = self.bottleneck_gp(p2)
        
        if self.training:
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
            
        d2 = self.up2(z)
        d2 = torch.cat([d2, e2], dim=1) 
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1), var.sum(dim=1, keepdim=True)
    

class SVGPUNet_Fluid(SVGPUNet):
    def __init__(self, in_channels=1, base=32):
        # Initialize parent with dummy classes
        super().__init__(in_channels=in_channels, num_classes=2, base=base)
        
        # REPLACE the final layer
        # Old: self.final = nn.Conv2d(base, num_classes, kernel_size=1)   
        # New: Output 2 channels (Mean, Log_Variance)
        self.final = nn.Conv2d(base, 2, kernel_size=1)
        
        # Optional: Initialize log_var to be low (stable start)
        torch.nn.init.constant_(self.final.bias[1], -5.0)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # GP Bottleneck (Latent Uncertainty)
        # This injects stochasticity into the latent space
        z_mu, z_var = self.bottleneck_gp(e2)
        
        # Use z_mu for reconstruction flow
        # (In a full Bayesian Network, we would sample z ~ N(mu, var))
        # For efficiency, we pass the mean, but the GP regularization 
        # constrains how this mean is learned.
        
        d2 = self.up2(z_mu)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final Output: [Batch, 2, H, W]
        out = self.final(d1)
        
        pred_mean = out[:, 0:1, :, :]      # Channel 0: Fluid State
        pred_log_var = out[:, 1:2, :, :]   # Channel 1: Uncertainty
        pred_var = torch.exp(pred_log_var)
        
        return pred_mean, pred_var