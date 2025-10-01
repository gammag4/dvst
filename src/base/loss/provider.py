from abc import ABC, abstractmethod
from typing import Generic, Callable

from src.base.config import TLossConfig


class LossProvider(ABC, Generic[TLossConfig]):
    @abstractmethod
    def create_loss(self, config: TLossConfig) -> Callable:
        pass
