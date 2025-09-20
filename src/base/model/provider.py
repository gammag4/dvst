from abc import ABC, abstractmethod
from torch.nn import Module

from src.base.config import BaseModelConfig


class ModelProvider(ABC):
    @abstractmethod
    def create_model(self, config: BaseModelConfig) -> Module:
        pass
