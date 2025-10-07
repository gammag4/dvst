from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Tuple
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.base.config import TOptimizerConfig


TModel = TypeVar('Model', bound=Module)


class OptimizerProvider(ABC, Generic[TOptimizerConfig, TModel]):
    @abstractmethod
    def create_optimizer(self, config: TOptimizerConfig, model: TModel, n_scheduler_steps: int | None) -> Tuple[Optimizer, LRScheduler | None]:
        pass
