import torch

from src.base.providers import LossProvider

from src.dvst.config import DVSTLossConfig
from .loss import PerceptualLoss


class DVSTLossProvider(LossProvider[DVSTLossConfig]):
    def create_loss(self, config):
        # return torch.nn.MSELoss()
        return PerceptualLoss()
