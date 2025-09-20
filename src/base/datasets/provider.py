from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from src.base.config import BaseDatasetConfig


class DatasetProvider(ABC):
    @abstractmethod
    async def download_dataset(self, config: BaseDatasetConfig):
        pass
    
    @abstractmethod
    def create_train_dataset(self, config: BaseDatasetConfig) -> Dataset:
        pass
    
    @abstractmethod
    def create_val_dataset(self, config: BaseDatasetConfig) -> Dataset:
        pass
    
    @abstractmethod
    def create_test_dataset(self, config: BaseDatasetConfig) -> Dataset:
        pass
