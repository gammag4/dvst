import math
import sunds
import torch
import einx

from src.dvst.datasets.scene_dataset import StaticScene, SceneData, IterableSceneDataset, process_data


class DySOSceneData(SceneData):
    def __init__(self, scene_data, n_sources, n_target_frames, resize_to):
        self.scene_data = scene_data
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        self.resize_to = resize_to
    
    def load(self, device):
        data = self.scene_data
        
        scene_name = data['scene_name'].numpy().decode('utf-8')
        
        cameras = [data['cameras'][f'camera_{i}'] for i in range(25)]
        
        # torch.Size([5, 5, 128, 128, 3])
        # (camera view, frame, C H W)
        I = einx.rearrange('(f v) h w c -> v f c h w', torch.stack([torch.tensor(c['color_image'].numpy(), device=device) for c in cameras]), f=5, v=5)
        
        dist = torch.stack([torch.concat([torch.tensor(c['intrinsics']['distortion'][k].numpy(), device=device) for k in ('radial', 'tangential')]) for c in cameras])
        dist[:, [4, 2]] = dist[:, [2, 4]]
        K = [c['intrinsics']['K'] for c in cameras]
        R, t = [[c['extrinsics'][k] for c in cameras] for k in ('R', 't')]
        K, R, t = [torch.stack([torch.tensor(i.numpy(), device=device) for i in k]) for k in (K, R, t)]
        
        # [torch.Size([5, 5, 3, 128, 128]), torch.Size([5, 5, 5]), torch.Size([5, 5, 3, 3]), torch.Size([5, 5, 3, 3]), torch.Size([5, 5, 3])]
        # v f c h w, v f d, v f i j, v f i j, v f i
        dist, K, R, t = [einx.rearrange('(f v) ... -> v f ...', k, f=5, v=5) for k in (dist, K, R, t)]
        time = einx.rearrange('f -> v f', torch.arange(5, device=device) * (1 + torch.rand(5, device=device)), v=5)
        
        hw = I.shape[-2:]
        K, R, t, time = process_data(K, R, t, time, hw, self.resize_to)
        
        return StaticScene.from_tensors(
            dataset_name='dyso',
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


class DySODataset(IterableSceneDataset):
    def __init__(self, resize_to: tuple[int, int] | None, n_sources: int, n_target_frames: int):
        super().__init__()
        self.resize_to = resize_to
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        
        builder = sunds.builder('kubric:kubric/msn_ms_v4.3a')
        self.scenes = builder.as_dataset(
            split='train', # TODO train val or test
            # Stack all camera of a scene together
            # task=sunds.tasks.Nerf(yield_mode='stacked'),
            task=sunds.tasks.Frames(),
        )
    
    def __len__(self):
        return len(self.scenes)
    
    def __iter__(self):
        for e in self.scenes:
            yield DySOSceneData(e, self.n_sources, self.n_target_frames, self.resize_to)
    
    @property
    def n_frames(self):
        return len(self) * 5
    
    @property
    def n_scenes(self):
        return len(self)
    
    def get_n_batches(self, batch_size):
        return math.ceil(5 / batch_size) * len(self)
