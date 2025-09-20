from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.nn import Module
from torch.utils.data import Dataset
from torch.optim import Optimizer

from src.base.config import Config


class SetupFactory(ABC):
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    async def download_dataset(self):
        pass
    
    @abstractmethod
    def create_train_dataset(self) -> Dataset:
        pass
    
    @abstractmethod
    def create_val_dataset(self) -> Dataset:
        pass
    
    @abstractmethod
    def create_test_dataset(self) -> Dataset:
        pass
    
    @abstractmethod
    def create_model(self) -> Module:
        pass
    
    @abstractmethod
    def create_optimizer(self, model: Module) -> Optimizer:
        pass
