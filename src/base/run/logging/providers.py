import os
from abc import ABC, abstractmethod
import torch

from .logging import Logger, PrintLogger, Stateful


class LogProvider(ABC):
    @abstractmethod
    def create_logger(self) -> Logger:
        pass


class PrintLogProvider(LogProvider):
    def create_logger(self):
        return PrintLogger()
