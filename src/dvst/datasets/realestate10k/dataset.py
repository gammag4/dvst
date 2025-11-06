import os
import random
import math
import torch
from torchvision.io import decode_image

from src.base.utils import json_load

from src.dvst.datasets.scene_dataset import StaticScene, SceneData, IterableSceneDataset, process_data


class PixelSplatRE10KSceneData(SceneData):
    def __init__(self, scene, n_sources, n_target_frames, resize_to):
        self.scene = scene
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        self.resize_to = resize_to
    
    def load(self, device):
        scene_name = self.scene['key']
        # times = self.scene['timestamps'] / 1000000 # From microseconds to seconds
        I = torch.stack([decode_image(i) for i in self.scene['images']]).to(device, non_blocking=True)
        cameras = self.scene['cameras'].to(device, non_blocking=True)
        
        fx, fy, px, py = [cameras[:, i] for i in range(4)]
        K = torch.zeros((fx.shape[0], 3, 3), device=device)
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, px, py, 1
        
        T = cameras[:, 6:].reshape((-1, 3, 4)).to(device, non_blocking=True)
        R, t = T[:, :, :3], T[:, :, 3]
        
        time = torch.zeros(I.shape[0], device=device)
        
        # (v f ...) where `f = 1`
        K, R, t, time, I = [k.unsqueeze(1) for k in (K, R, t, time, I)]
        
        hw = I.shape[-2:]
        K, R, t, time = process_data(K, R, t, time, hw, self.resize_to)
        
        return StaticScene.from_tensors(
            dataset_name='realestate10k',
            scene_name=scene_name,
            K=K,
            R=R,
            t=t,
            time=time,
            I=I,
            n_sources=self.n_sources,
            n_target_frames=self.n_target_frames,
            resize_to=self.resize_to
        )


class PixelSplatRealEstate10KDataset(IterableSceneDataset):
    def __init__(self, path, resize_to: tuple[int, int] | None, n_sources: int, n_target_frames: int):
        super().__init__()
        self.path = path
        self.resize_to = resize_to
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        
        self.scenes = json_load(os.path.join(self.path, 'data.json')).scenes
        self.batches = [i for i in os.listdir(self.path) if i.endswith('.torch')]
        random.shuffle(self.batches)
        self.current_batch = 0
        self.current_index = 0
    
    def __len__(self):
        return len(self.scenes)
    
    def __iter__(self):
        for b in self.batches:
            batch = torch.load(os.path.join(self.path, b))
            random.shuffle(batch)
            for s in batch:
                yield PixelSplatRE10KSceneData(s, self.n_sources, self.n_target_frames, self.resize_to)
    
    @property
    def n_frames(self):
        return len(self)
    
    @property
    def n_scenes(self):
        return len(self)
    
    def get_n_batches(self, batch_size):
        return sum([math.ceil(s.n_frames / batch_size) for s in self.scenes])
