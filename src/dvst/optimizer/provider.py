from typing import cast
import torch
from torch.nn import Module

from src.base.config import BaseOptimizerConfig
from src.base.providers import OptimizerProvider

from src.dvst.config import DVSTOptimizerConfig
from src.dvst.model import DVST


class DVSTOptimizerProvider(OptimizerProvider):
    def create_optimizer(self, config: BaseOptimizerConfig, model: Module):
        config = cast(DVSTOptimizerConfig, config)
        model = cast(DVST, model)
        
        # Removing parameters that are not optimized
        params = [p for p in model.parameters() if p.requires_grad]
        
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            fused=config.fused
        )
