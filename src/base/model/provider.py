from abc import ABC, abstractmethod
from typing import Generic, Callable
from torch.nn import Module

from src.base.config import TModelConfig


class ModelProvider(ABC, Generic[TModelConfig]):
    @abstractmethod
    def create_model(self, config: TModelConfig, loss: Callable | None) -> Module:
        pass
