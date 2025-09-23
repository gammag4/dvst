from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from torch.nn import Module
from torch.optim import Optimizer

from src.base.config import TOptimizerConfig


TModel = TypeVar('Model', bound=Module)


class OptimizerProvider(ABC, Generic[TOptimizerConfig, TModel]):
    @abstractmethod
    def create_optimizer(self, config: TOptimizerConfig, model: TModel) -> Optimizer:
        pass
