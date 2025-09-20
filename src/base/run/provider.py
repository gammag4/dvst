from abc import ABC, abstractmethod

from src.base.config import Config
from src.base.datasets.provider import DatasetProvider
from src.base.model.provider import ModelProvider
from src.base.optimizer.provider import OptimizerProvider
from .trainer import DistributedTrainer


class RunProvider(ABC):
    def __init__(self,
        dataset_provider: DatasetProvider,
        model_provider: ModelProvider,
        optimizer_provider: OptimizerProvider
    ):
        self.dataset_provider = dataset_provider
        self.model_provider = model_provider
        self.optimizer_provider = optimizer_provider
    
    @abstractmethod
    def create_trainer(self, config: Config) -> DistributedTrainer:
        pass
