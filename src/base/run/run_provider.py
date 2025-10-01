from abc import ABC, abstractmethod
from typing import Generic

from src.base.config import Config, TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig
from src.base.datasets.provider import DatasetProvider
from src.base.model.provider import ModelProvider
from src.base.optimizer.provider import OptimizerProvider, TModel
from src.base.loss.provider import LossProvider
from .log_provider import LogProvider
from .trainer import DistributedTrainer

class RunProvider(ABC, Generic[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig, TModel]):
    def __init__(self,
        dataset_provider: DatasetProvider[TDatasetConfig],
        model_provider: ModelProvider[TModelConfig],
        optimizer_provider: OptimizerProvider[TOptimizerConfig, TModel],
        loss_provider: LossProvider[TLossConfig],
        log_provider: LogProvider
    ):
        self.dataset_provider = dataset_provider
        self.model_provider = model_provider
        self.optimizer_provider = optimizer_provider
        self.loss_provider = loss_provider
        self.log_provider = log_provider
    
    @abstractmethod
    def create_trainer(self, config: Config[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig]) -> DistributedTrainer[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig, TModel]:
        pass
