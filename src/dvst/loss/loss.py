
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torchvision.transforms.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=torch.ones(10, dtype=torch.float32)):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        self.transforms = weights.transforms()
        self.transforms.resize_size=224
        self.layers = list(model.features) + [lambda x: model.classifier(model.avgpool(x))]
        self.model = model
        
        self.layer_weights = layer_weights
    
    def forward_layer(self, x1, x2):
        offset = len(x1.shape) - 3
        
        # C * H * W
        N = x1.shape[offset:].numel()

        # TODO test with norm 1 instead of 2 (other works like LVSM use norm 1)
        return torch.norm(x1 - x2, p=2, dim=-1).sum() / N
    
    def forward(self, input, target):
        losses = []
        
        x1 = F.center_crop(input, max(input.shape[-1], input.shape[-2]))
        x2 = F.center_crop(target, max(target.shape[-1], target.shape[-2]))
        losses.append(self.forward_layer(x1, x2))
        
        x1, x2 = self.transforms(x1), self.transforms(x2)
        for l in self.layers:
            x1, x2 = l(x1), l(x2)
            losses.append(self.forward_layer(x1, x2))
        
        weights = self.layer_weights.to(input.device)
        weights = weights / weights.sum() # Normalizes weights
        
        loss = (torch.stack(losses) * weights).sum()
        return loss

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['layer_weights'] = self.layer_weights
        
        return state_dict
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        self.layer_weights = state_dict['layer_weights']
        super().load_state_dict(state_dict, *args, **kwargs)
