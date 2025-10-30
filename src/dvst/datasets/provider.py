import os
import torch.distributed as dist

from src.dvst.datasets.scene_dataset import IterableCollectionSceneDataset, DistIterableSceneDataset
from src.base.providers import DatasetProvider

from src.dvst.config import DVSTDatasetConfig
from .panoptic import PanopticDataset, PanopticDownloader
from .realestate10k import PixelSplatRealEstate10KDataset, PixelSplatRealEstate10KDownloader
from .kubric import DySODataset


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
            PixelSplatRealEstate10KDownloader(
                path=os.path.join(config.path, 're10k/train')
            )
        ]
        
        for d in downloaders:
            await d.download()
    
    def create_long_sequence_train_dataset(self, config):
        datasets = [
            PanopticDataset(
                path=os.path.join(config.path, 'panoptic'),
                resize_to=(32, 56),
                n_sources=2,
                n_target_frames=16
            ),
            # RawPlenopticDataset(
            #     os.path.join(config.path, 'plenoptic'),
            #     (64, 114), 2, None
            # ),
        ]
        
        return DistIterableSceneDataset(IterableCollectionSceneDataset(datasets), dist.get_rank(), dist.get_world_size())
    
    def create_train_dataset(self, config):
        # TODO Add MultiShapeNet dataset https://srt-paper.github.io/#dataset
        #   Also add DySO dataset
        # TODO Add original RealEstate10K (google.github.io/realestate10k/download.html)
        #   Processing like pixelSplat: https://github.com/dcharatan/pixelsplat/blob/main/README.md#acquiring-datasets
        #   Script to download more easily (supposedly): https://github.com/Findeton/real-state-10k
        # when adding image datasets, concat them into tensor and return it as if it was the video
        datasets = [
            # TODO missing credentials
            # DySODataset(
            #     resize_to=(48, 48),
            #     n_sources=2,
            #     n_target_frames=16
            # ),
            PixelSplatRealEstate10KDataset(
                path=os.path.join(config.path, 're10k/train'),
                resize_to=(32, 56), # TODO Check if this resize is respecting aspect ratios of images
                n_sources=2,
                n_target_frames=16
            )
        ]
        
        # TODO split dataset in a way where each distributed process receives roughly the same amount of batches
        return DistIterableSceneDataset(IterableCollectionSceneDataset(datasets), dist.get_rank(), dist.get_world_size())
    
    def create_val_dataset(self, config):
        return DistIterableSceneDataset(IterableCollectionSceneDataset([]), dist.get_rank(), dist.get_world_size()) # TODO
    
    def create_test_dataset(self, config):
        return DistIterableSceneDataset(IterableCollectionSceneDataset([]), dist.get_rank(), dist.get_world_size()) # TODO
