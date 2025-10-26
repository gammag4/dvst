import os
import math
from easydict import EasyDict as edict
import torch
from torchcodec.decoders import VideoDecoder

from src.base.utils import json_load

from src.dvst.datasets.scene_dataset import VideoDecoderScene, SceneData, SceneDataset, process_K


class PanopticSceneData(SceneData):
    def __init__(self, scene_name, views, fps, n_frames, n_sources, n_target_frames, resize_to):
        self.scene_name = scene_name
        self.views = views
        self.fps = fps
        self.n_frames = n_frames
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        self.resize_to = resize_to
    
    def load(self, device):
        I = [VideoDecoder(v.path, device=device) for v in self.views]
        
        # K: v f 3 3, R: v f 3 3, t: v f 3, time: v f
        K, R, t = zip(*[[v.K, v.R, v.t] for v in self.views])
        K, R, t = [torch.stack([torch.tensor(i, device=device).unsqueeze(0).repeat((self.n_frames, 1, 1)) for i in k]) for k in (K, R, t)]
        t = t.squeeze(-1)
        time = torch.arange(self.n_frames, device=device) / self.fps
        time = time.unsqueeze(0).repeat((len(I), 1))
        
        K = process_K(K, I[0].shape[-2:] if self.resize_to is None else self.resize_to)
        
        return VideoDecoderScene(
            dataset_name='panoptic',
            scene_name=self.scene_name,
            n_frames=self.n_frames,
            K=K,
            R=R,
            t=t,
            time=time,
            I=I,
            n_sources=self.n_sources,
            n_target_frames=self.n_target_frames,
            resize_to=self.resize_to
        )


class PanopticDataset(SceneDataset):
    def __init__(self, path, resize_to: tuple[int, int] | None, n_sources: int, n_target_frames: int):
        super().__init__()
        self.path = path
        self.fps = {'hd': 29.97, 'vga': 25.0, 'kinect-color': 30}
        self.resize_to = resize_to
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        
        scenes: list[SceneData] = []
        for sname in os.listdir(self.path):
            if not os.path.isdir(os.path.join(self.path, sname)):
                continue
            
            spath = os.path.join(self.path, sname)
            data = json_load(os.path.join(spath, f'data.json'))
            
            scenes.append(PanopticSceneData(
                scene_name=sname,
                views=[edict(
                    path=os.path.join(spath, v.path),
                    K=v.K,
                    R=v.R,
                    t=v.t,
                    shape=v.shape,
                ) for v in data.views],
                fps=self.fps['hd'],
                n_frames=data.n_frames,
                n_sources=self.n_sources,
                n_target_frames=self.n_target_frames,
                resize_to=self.resize_to
            ))
        
        self.scenes = scenes
        self._n_frames = sum([s.n_frames for s in scenes])
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, i):
        return self.scenes[i]
    
    @property
    def n_frames(self):
        return self._n_frames
    
    @property
    def n_scenes(self):
        return len(self.scenes)
    
    def get_n_batches(self, batch_size):
        return sum([math.ceil(s.n_frames / batch_size) for s in self.scenes])
