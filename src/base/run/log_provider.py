from abc import ABC, abstractmethod

from .logger import Logger, PrintLogger


class LogProvider(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def create_logger(self) -> Logger:
        pass


class PrintLogProvider(LogProvider):
    def create_logger(self):
        return PrintLogger()
