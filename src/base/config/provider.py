from abc import ABC, abstractmethod
from typing import Generic

from .config import Config, TDatasetConfig, TModelConfig, TOptimizerConfig


class ConfigProvider(ABC, Generic[TDatasetConfig, TModelConfig, TOptimizerConfig]):
    @abstractmethod
    def create_default_config(self) -> Config[TDatasetConfig, TModelConfig, TOptimizerConfig]:
        pass
