from abc import ABC, abstractmethod

from .config import Config


class ConfigProvider(ABC):
    @abstractmethod
    def create_default_config(self) -> Config:
        pass
