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