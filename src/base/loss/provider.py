from abc import ABC, abstractmethod
from typing import Generic, Tuple, Callable

from src.base.config import TLossConfig
from .scheduler import LossScheduler


class LossProvider(ABC, Generic[TLossConfig]):
    @abstractmethod
    def create_loss(self, config: TLossConfig, n_scheduler_steps: int | None) -> Tuple[Callable, LossScheduler | None]:
        pass
