from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Generic

from .config import Config, TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig


class ConfigProvider(ABC, Generic[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig]):
    def __init__(self):
        self.config = None
    
    @abstractmethod
    def _create_default_config(self) -> Config[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig]:
        pass
    
    def get_default_config(self) -> Config[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig]:
        if self.config is None:
            self.config = self._create_default_config()
        
        return replace(self.config)
