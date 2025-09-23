import torch

from src.base.providers import OptimizerProvider

from src.dvst.config import DVSTOptimizerConfig
from src.dvst.model import DVST


class DVSTOptimizerProvider(OptimizerProvider[DVSTOptimizerConfig, DVST]):
    def create_optimizer(self, config, model):
        # Removing parameters that are not optimized
        params = [p for p in model.parameters() if p.requires_grad]
        
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            fused=config.fused
        )
