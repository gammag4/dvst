
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=torch.ones(10, dtype=torch.float32)):
        super().__init__()
        # TODO Fix so that the implementation is right (use 1 norm instead of 2 and use hyperparameters for each layer)
        # TODO add layer_weights scheduler
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.transforms = weights.transforms()
        self.layers = [lambda x: x] + list(model.features) + [lambda x: model.classifier(model.avgpool(x))]
        self.model = model
        
        self.layer_weights = layer_weights # TODO put to weight scheduler

    def forward(self, input, target):
        x1, x2 = self.transforms(input), self.transforms(target)
        offset = len(x1.shape) - 3
        losses = []

        for l in self.layers:
            x1, x2 = l(x1), l(x2)
            
            # C * H * W
            N = x1.shape[offset:].numel()
            losses.append(torch.norm(x1 - x2, p=2, dim=-1).sum() / N)

        weights = self.layer_weights / self.layer_weights.sum() # Normalizes weights
        weights = weights.to(input.device) # TODO put to weight scheduler
        loss = (torch.stack(losses) * weights).sum()
        return loss
