import os
import torch

from src.base.config import Config
from src.base.run import SetupFactory
from src.base.datasets import FullDataset

from src.dvst.datasets import PanopticDataset, PanopticDownloader, RawPlenopticDataset
from src.dvst.model import DVST


class DVSTSetupFactory(SetupFactory):
    def __init__(self, config: Config):
        super().__init__(config)
    
    async def download_dataset(self):
        config = self.config.train.data
        
        downloaders = [
            PanopticDownloader(
                path=os.path.join(config.path, 'panoptic'),
                use_cuda=True,
                cq_amount=23,
                resize_to=(-1, 256),
                n_scenes=None,
                n_views=8
            ),
        ]
        
        for d in downloaders:
            await d.download()
    
    def _create_datasets(self):
        config = self.config.train.data
        
        # TODO Add MultiShapeNet dataset https://srt-paper.github.io/#dataset
        # TODO Add RealEstate10K dataset processed by pixelsplat https://github.com/dcharatan/pixelsplat/blob/main/README.md#acquiring-datasets
        #   Original: google.github.io/realestate10k/download.html
        #   Easier script to download (supposedly): https://github.com/Findeton/real-state-10k
        # when adding image datasets, concat them into tensor and return it as if it was the video
        datasets = [
            PanopticDataset(
                os.path.join(config.path, 'panoptic'),
                (64, 114), 2, None
            ),
            # RawPlenopticDataset(
            #     os.path.join(config.path, 'plenoptic'),
            #     (64, 114), 2, None
            # ),
        ]
        
        # TODO split dataset in a way where each distributed process receives roughly the same amount of batches
        return FullDataset(datasets)
    
    def create_train_dataset(self):
        return self._create_datasets()
    
    def create_val_dataset(self):
        return [] # TODO
    
    def create_test_dataset(self):
        return [] # TODO
    
    def create_model(self):
        config = self.config.model
        return DVST(config)
    
    def create_optimizer(self, model):
        config = self.config.train.optimizer
        
        # Removing parameters that are not optimized
        params = [p for p in model.parameters() if p.requires_grad]
        
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            fused=config.fused
        )
