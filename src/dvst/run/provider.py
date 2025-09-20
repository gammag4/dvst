from src.base.config import Config
from src.base.providers import RunProvider
from src.base.run import DistributedTrainer

from .trainer import DVSTTrainer


class DVSTRunProvider(RunProvider):
    def create_trainer(self, config: Config) -> DistributedTrainer:
        return DVSTTrainer(
            config,
            dataset_provider=self.dataset_provider,
            model_provider=self.model_provider,
            optimizer_provider=self.optimizer_provider
        )
