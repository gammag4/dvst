import os
import json

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class PanopticDataset(Dataset):
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, 'data.json')) as f:
            data = json.load(f)

        for d in data:
            for i in d[1]:
                i['filename'] = os.path.join(path, d[0], i['filename'])

        # Coordinate convention transformation matrix
        # Computed in experiments/ponoptic_coordinate_conventions.ipynb
        self.Q = torch.tensor([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i][1]
        res = [(
            VideoDecoder(v['filename']),
            torch.tensor(v['K']),
            torch.tensor(v['R']),
            torch.tensor(v['t']),
            self.Q,
            v['fps'],
            torch.Size(v['shape'])
        ) for v in d]
        return res
