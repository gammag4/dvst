import torch
from src.base.providers import ModelProvider

from src.dvst.config import DVSTModelConfig
from .model import DVST


def init_weights(module):
    std = 0.02
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    if isinstance(module, torch.nn.Parameter):
        torch.nn.init.normal_(module.data, mean=0, std=std)


class DVSTModelProvider(ModelProvider[DVSTModelConfig]):
    def create_model(self, config, loss):
        model = DVST(config, loss)
        model.encoder.apply(init_weights)
        model.decoder.apply(init_weights)
        
        return model
