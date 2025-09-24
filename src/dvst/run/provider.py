from src.base.providers import RunProvider

from src.dvst.config import *
from src.dvst.model import DVST
from .trainer import DVSTTrainer


class DVSTRunProvider(RunProvider[DVSTDatasetConfig, DVSTModelConfig, DVSTOptimizerConfig, DVST]):
    def create_trainer(self, config):
        return DVSTTrainer(
            config,
            dataset_provider=self.dataset_provider,
            model_provider=self.model_provider,
            optimizer_provider=self.optimizer_provider,
            log_provider=self.log_provider
        )
