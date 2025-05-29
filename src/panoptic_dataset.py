import os
import json

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from .utils import preprocess_scene_video

#TODO create test dataset
class PanopticDataset(Dataset):
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, 'data.json')) as f:
            data = json.load(f)

        for d in data:
            for i in d[1]:
                i['filename'] = os.path.join(path, d[0], i['filename'])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i][1]

        return [
            preprocess_scene_video(
                v['filename'],
                v['K'],
                v['R'],
                v['t'],
                v['fps']
            ) for v in d
        ]
