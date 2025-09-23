from abc import ABC, abstractmethod
from typing import Generic
from torch.utils.data import Dataset

from src.base.config import TDatasetConfig


class DatasetProvider(ABC, Generic[TDatasetConfig]):
    @abstractmethod
    async def download_dataset(self, config: TDatasetConfig):
        pass
    
    @abstractmethod
    def create_train_dataset(self, config: TDatasetConfig) -> Dataset:
        pass
    
    @abstractmethod
    def create_val_dataset(self, config: TDatasetConfig) -> Dataset:
        pass
    
    @abstractmethod
    def create_test_dataset(self, config: TDatasetConfig) -> Dataset:
        pass
