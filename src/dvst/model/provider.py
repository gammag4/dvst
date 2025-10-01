from src.base.providers import ModelProvider

from src.dvst.config import DVSTModelConfig
from .model import DVST


class DVSTModelProvider(ModelProvider[DVSTModelConfig]):
    def create_model(self, config, loss):
        return DVST(config, loss)
