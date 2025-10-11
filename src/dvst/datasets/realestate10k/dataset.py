import os
import math
import torch
from torchvision.io import decode_image

from src.base.utils import json_load

from src.dvst.datasets.scene_dataset import TensorViewData, SceneData, SceneDataset


class PixelSplatRealEstate10KDataset(SceneDataset):
    def __init__(self, path, resize_to, n_sources, n_targets):
        self.path = path
        self.resize_to = resize_to
        self.n_sources = n_sources
        self.n_targets = n_targets
        
        scenes = json_load(os.path.join(self.path, 'data.json')).scenes
        
        self.scenes = scenes
        self._n_frames = sum([s.n_frames for s in scenes])
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, i):
        scene_data = self.scenes[i]
        scene = torch.load(os.path.join(self.path, scene_data.file))[scene_data.file_index]
        
        scene_name = scene['key']
        times = scene['timestamps'] / 1000000 # From microseconds to seconds
        images = torch.stack([decode_image(i) for i in scene['images']])
        cameras = scene['cameras']
        
        fx, fy, px, py = [cameras[:, i] for i in range(4)]
        K = torch.zeros((fx.shape[0], 3, 3))
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, px, py, torch.ones(K.shape[0])
        
        T = cameras[:, 6:].reshape((-1, 3, 4))
        R, t = T[:, :, :3], T[:, :, 3:]
        
        # TODO randomly choose views
        sources_mask = torch.arange(images.shape[0])[None if self.n_sources is None else -self.n_sources:]
        targets_mask = torch.arange(images.shape[0])[:self.n_targets]
        
        view_datas = [
            TensorViewData(
                view=images[targets_mask],
                K=K[0], # TODO
                # K=K[targets_mask],
                R=R[targets_mask],
                t=t[targets_mask],
                time=times[targets_mask],
                fps=None,
                shape=targets_mask.shape,
                resize_to=self.resize_to
            ),
            TensorViewData(
                view=images[sources_mask],
                K=K[0], # TODO
                # K=K[sources_mask],
                R=R[sources_mask],
                t=t[sources_mask],
                time=times[sources_mask],
                fps=None,
                shape=sources_mask.shape,
                resize_to=self.resize_to
            )
        ]
        
        scene = SceneData.from_sources_targets_split(
            dataset_name='realestate10k',
            scene_name=scene_name,
            view_datas=view_datas,
            n_frames=scene_data.n_frames,
            n_sources=1,
            n_targets=1,
            shuffle=True,
            shuffle_before_splitting=True
        )
        return self.scenes[i]
    
    @property
    def n_frames(self):
        return self._n_frames
    
    @property
    def n_scenes(self):
        return len(self.scenes)
    
    def get_n_batches(self, batch_size):
        return sum([math.ceil(s.n_frames / batch_size) for s in self.scenes])
