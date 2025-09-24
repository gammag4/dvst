from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Generic

from .config import Config, TDatasetConfig, TModelConfig, TOptimizerConfig


class ConfigProvider(ABC, Generic[TDatasetConfig, TModelConfig, TOptimizerConfig]):
    def __init__(self):
        self.config = None
    
    @abstractmethod
    def _create_default_config(self) -> Config[TDatasetConfig, TModelConfig, TOptimizerConfig]:
        pass
    
    def get_default_config(self) -> Config[TDatasetConfig, TModelConfig, TOptimizerConfig]:
        if self.config is None:
            self.config = self._create_default_config()
        
        return replace(self.config)
