import os
import math
import torch
from torchvision.io import decode_image

from src.base.utils import json_load

from src.dvst.datasets.scene_dataset import StaticScene, SceneData, IndexableSceneDataset, process_data


class PixelSplatRE10KSceneData(SceneData):
    def __init__(self, scene_data, path, n_sources, n_target_frames, resize_to):
        self.scene_data = scene_data
        self.path = path
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        self.resize_to = resize_to
    
    def load(self, device):
        scene = torch.load(os.path.join(self.path, self.scene_data.file))[self.scene_data.file_index]
        
        scene_name = scene['key']
        # times = scene['timestamps'] / 1000000 # From microseconds to seconds
        I = torch.stack([decode_image(i) for i in scene['images']]).to(device, non_blocking=True)
        cameras = scene['cameras'].to(device, non_blocking=True)
        
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


class PixelSplatRealEstate10KDataset(IndexableSceneDataset):
    def __init__(self, path, resize_to: tuple[int, int] | None, n_sources: int, n_target_frames: int):
        super().__init__()
        self.path = path
        self.resize_to = resize_to
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        
        self.scenes = json_load(os.path.join(self.path, 'data.json')).scenes
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, i):
        return PixelSplatRE10KSceneData(self.scenes[i], self.path, self.n_sources, self.n_target_frames, self.resize_to)
    
    @property
    def n_frames(self):
        return len(self)
    
    @property
    def n_scenes(self):
        return len(self)
    
    def get_n_batches(self, batch_size):
        return sum([math.ceil(s.n_frames / batch_size) for s in self.scenes])


# class BatchedDataset(IndexableSceneDataset):
#     def __init__(self, dataset: IndexableSceneDataset, batch_size):
#         super().__init__()
#         self.dataset = dataset
#         self.batch_size = batch_size
    
#     def __len__(self):
#         return self.get_n_batches(self.batch_size)
    
#     def __getitem__(self, i):
#         scenes = [self.dataset[j] for j in range(i * self.batch_size, min(len(self.dataset), (i + 1) * self.batch_size))]
        
#         sources, targets = zip(*[[s.sources, s.targets] for s in scenes])
#         sources, targets = [zip(*[[s.K, s.R, s.t, s.time, s.I] for s in ss]) for ss in (sources, targets)]
#         sources, targets = [SourceBatch(*[torch.concat(i) for i in ss]) for ss in (sources, targets)]
        
#         return edict(
#             sources=sources,
#             targets=targets,
#             scene_ids=[sid for s in scenes for sid in s.scene_ids],
#             start=0,
#             end=len(scenes)
#         )
    
#     @property
#     def n_frames(self):
#         return self.dataset.n_frames
    
#     @property
#     def n_scenes(self):
#         return self.dataset.n_scenes
    
#     def get_n_batches(self, batch_size):
#         return self.dataset.get_n_batches(batch_size)
