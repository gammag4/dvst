import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from src.dvst.utils import colmap_poses_to_intrinsics_extrinsics


# Download dataset from https://github.com/facebookresearch/Neural_3D_Video/releases
# unzip all in a folder and use it as root for the dataset


class RawPlenopticDataset(Dataset):
    def __init__(self, path, include_test=False):
        self.path = path
        self.fps = 30
        self.include_test = include_test

        scene_names = list(filter(lambda p: os.path.isdir(os.path.join(self.path, p)), os.listdir(self.path)))

        scenes = []
        for sname in scene_names:
            spath = os.path.join(self.path, sname)
            vids = sorted(filter(lambda n: not n.endswith('.npy'), os.listdir(spath)))
            vids = [os.path.join(spath, v) for v in vids]

            poses_raw = torch.from_numpy(np.load(os.path.join(spath, 'poses_bounds.npy')))
            K, T, _ = colmap_poses_to_intrinsics_extrinsics(poses_raw)
            R, t = T[:, :, :3], T[:, :, 3:]
            K, R, t = [list(i) for i in [K, R, t]]
            
            fpss = [self.fps] * len(vids)

            res = list(zip(vids, K, R, t, fpss))
            scenes.append(res)

        self.data = scenes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        scene = self.data[i]
        # First video is the test one
        scene = scene if self.include_test else scene[1:]
        
        return scene
