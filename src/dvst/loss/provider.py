import torch

from src.base.providers import LossProvider

from src.dvst.config import DVSTLossConfig
from .loss import PerceptualLoss
from .scheduler import PerceptualLossScheduler


class DVSTLossProvider(LossProvider[DVSTLossConfig]):
    def create_loss(self, config, n_scheduler_steps):
        if config.loss == 'mse':
            return (torch.nn.MSELoss(), None)
        elif config.loss == 'perceptual':
            assert n_scheduler_steps is not None, 'n_scheduler_steps cannot be None for perceptual loss'
            loss = PerceptualLoss()
            scheduler = PerceptualLossScheduler(loss, n_scheduler_steps, config.scheduler.beta, config.scheduler.regime)
            return (loss, scheduler)
        else:
            raise Exception(f'Invlid loss "f{config.loss}"')
