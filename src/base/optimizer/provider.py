from abc import ABC, abstractmethod
from torch.nn import Module
from torch.optim import Optimizer

from src.base.config import BaseOptimizerConfig


class OptimizerProvider(ABC):
    @abstractmethod
    def create_optimizer(self, config: BaseOptimizerConfig, model: Module) -> Optimizer:
        pass
