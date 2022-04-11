import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

class TestTimeIN(nn.BatchNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, target_input):
        exponential_average_factor = self.momentum        
        n = target_input.numel() / target_input.size(1)
        
        with torch.no_grad():
            # source_var = self.running_var[None, :, None, None]
            # source_mean = self.running_mean[None, :, None, None]
            
            target_instance_var = target_input.var([2, 3], unbiased=False)[:, :, None, None]
            target_instance_mean = target_input.mean([2, 3])[:, :, None, None]
            
            # target_mean = target_input.mean([0, 2, 3])[None, :, None, None]
            # target_var = target_input.var([0, 2, 3], unbiased=False)[None, :, None, None]
            
            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            
            target_input = weight * (target_input - target_instance_mean) / (torch.sqrt(target_instance_var + self.eps)) + bias
            
            target_input = torch.clamp(target_input, max=1)            
            return target_input

