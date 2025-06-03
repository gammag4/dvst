import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from .utils import preprocess_scene_video


# Download dataset from https://github.com/facebookresearch/Neural_3D_Video/releases
# unzip all in a folder and use it as root for the dataset


def colmap_poses_to_intrinsics_extrinsics(data):
    mat, close, far = data[:, :-2].reshape((-1, 3, 5)), data[:, -2], data[:, -1]
    T, mat2 = mat[:, :, :-1], mat[:, :, -1:]
    h, w, f = mat2[:, 0, 0], mat2[:, 1, 0], mat2[:, 2, 0]

    # Since we only have width, height and focal point, the intrinsics matrix will give imprecise results
    K = torch.zeros((T.shape[0], 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2], K[:, 0, 2], K[:, 1, 2] = f, f, 1, w / 2, h / 2

    return K, T, (h, w)


class PlenopticDataset(Dataset):
    def __init__(self, path, is_test=False):
        self.path = path
        self.fps = 30
        self.is_test = is_test
        scene_names = list(filter(lambda p: os.path.isdir(os.path.join(path, p)), os.listdir(path)))
        
        scenes = []
        for sname in scene_names:
            spath = os.path.join(self.path, sname)
            vids = sorted(filter(lambda n: not n.endswith('.npy'), os.listdir(spath)))
            vids = [os.path.join(spath, v) for v in vids]

            poses_raw = torch.from_numpy(np.load(os.path.join(spath, 'poses_bounds.npy')))
            K, T, _ = colmap_poses_to_intrinsics_extrinsics(poses_raw)
            R, t = T[:, :, :3], T[:, :, 3:]
            K, R, t = [list(i) for i in [K, R, t]]
            
            fps = [self.fps] * len(vids)

            res = list(zip(vids, K, R, t, fps))
            # First video is the test one
            res = res[:1] if self.is_test else res[1:]

            scenes.append(res)

        self.data = scenes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        return [preprocess_scene_video(*v) for v in d]
