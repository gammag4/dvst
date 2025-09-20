from typing import cast
from torch.nn import Module

from src.base.config import BaseModelConfig
from src.base.providers import ModelProvider

from src.dvst.config import DVSTModelConfig
from .model import DVST


class DVSTModelProvider(ModelProvider):
    def create_model(self, config: BaseModelConfig) -> Module:
        config = cast(DVSTModelConfig, config)
        return DVST(config)
