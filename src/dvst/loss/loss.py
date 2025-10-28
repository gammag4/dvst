import torch
import torch.nn as nn
import einx
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
        self.layers = list(model.features)
        # TODO test without classifier layer
        self.classifier_layer = lambda x: model.classifier(model.avgpool(x))
        self.model = model
        
        self.layer_weights = nn.Buffer(layer_weights)
    
    def distance(self, x1, x2):
        # TODO test with norm 1 instead of 2 (other works like LVSM use norm 1)
        # TODO check which is better l1 or l2 (l1 is more like a mean of the per-pixel errors and l2 is more like a distance in the space of possible images)
        # Using l1 for now so that it is equivalent to a score where 1 is furthest image possible (image of zeros vs image of ones) and 0 is closest possible (exact match)
        # Then that score is used to know how well it is still maintaining information from previous frames in current latent embeds
        return torch.norm(x1 - x2, p=1, dim=-1) / x1.shape[-1] # norm / C * H * W
    
    def forward_layer(self, x1, x2):
        x1, x2 = [einx.rearrange('... c h w -> ... (c h w)', k) for k in (x1, x2)]
        
        return self.distance(x1, x2)
    
    def forward(self, input, target):
        losses = []
        
        original_shape = input.shape[:-3]
        input, target = [einx.rearrange('... c h w -> (...) c h w', k) for k in (input, target)]
        
        x1 = F.center_crop(input, max(input.shape[-1], input.shape[-2]))
        x2 = F.center_crop(target, max(target.shape[-1], target.shape[-2]))
        losses.append(self.forward_layer(x1, x2))
        
        x1, x2 = self.transforms(x1), self.transforms(x2)
        for l in self.layers:
            x1, x2 = l(x1), l(x2)
            losses.append(self.forward_layer(x1, x2))
        
        losses.append(self.distance(self.classifier_layer(x1), self.classifier_layer(x2)))
        
        weights = self.layer_weights.to(input.device, non_blocking=True)
        weights = weights / weights.sum() # Normalizes weights
        
        losses = einx.dot('s b, s -> b', torch.stack(losses), weights)
        losses = losses.reshape(original_shape)
        return losses
