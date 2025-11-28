import torch
from torch.optim.lr_scheduler import ConstantLR, ChainedScheduler, LinearLR, CosineAnnealingLR

from src.base.providers import OptimizerProvider

from src.dvst.config import DVSTOptimizerConfig
from src.dvst.model import DVST


class DVSTOptimizerProvider(OptimizerProvider[DVSTOptimizerConfig, DVST]):
    def create_optimizer(self, config, model, n_scheduler_steps):
        # Removing parameters that are not optimized
        params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            fused=config.fused
        )
        
        if n_scheduler_steps is None:
            n_scheduler_steps = config.n_warmup_steps
        
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-12, end_factor=1, total_iters=config.n_warmup_steps)
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=n_scheduler_steps - config.n_warmup_steps)
        lr_scheduler = ChainedScheduler([warmup_scheduler, decay_scheduler], optimizer=optimizer)
        # lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=n_scheduler_steps)
        
        return (optimizer, lr_scheduler)
