import torch
import torch.nn as nn
from .layers import GPKANLayer

class GPKAN(nn.Module):
    """
    The core Neural Network class.
    Stacks multiple GPKANLayers to form a deep probabilistic network.
    """
    def __init__(self, layers_hidden, num_inducing=20, kernel_type='rbf'):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Build layer stack
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                GPKANLayer(
                    in_features=layers_hidden[i], 
                    out_features=layers_hidden[i+1], 
                    num_inducing=num_inducing, 
                    kernel_type=kernel_type
                )
            )

    def forward(self, x):
        state = x
        for layer in self.layers:
            state = layer(state)
        # Returns tuple (Mean, Variance)
        return state

def gaussian_nll_loss(pred_mean, pred_var, target):
    """
    Negative Log Likelihood Loss for Gaussian distributions.
    Optimizes for both accuracy (Mean) and calibration (Variance).
    """
    # Ensure positive variance
    pred_var = torch.clamp(pred_var, min=1e-6)
    
    # Loss = 0.5 * ( log(var) + (y-mu)^2 / var )
    loss_term = 0.5 * (torch.log(pred_var) + (target - pred_mean).pow(2) / pred_var)
    
    return loss_term.mean()
