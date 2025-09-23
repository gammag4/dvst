import os
from torch.utils.data import Dataset

from src.base.datasets import FullDataset
from src.base.providers import DatasetProvider

from src.dvst.config import DVSTDatasetConfig
from .panoptic import PanopticDataset, PanopticDownloader
from .plenoptic import RawPlenopticDataset


class DVSTDatasetProvider(DatasetProvider[DVSTDatasetConfig]):
    async def download_dataset(self, config):
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
    
    def _create_datasets(self, config):
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
    
    def create_train_dataset(self, config):
        return self._create_datasets(config)
    
    def create_val_dataset(self, config):
        return FullDataset([]) # TODO
    
    def create_test_dataset(self, config):
        return FullDataset([]) # TODO
